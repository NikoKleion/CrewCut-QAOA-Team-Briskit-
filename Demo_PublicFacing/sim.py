#!/usr/bin/env python3
"""
Exact state-vector simulation behind the CrewCut Lab demo (numpy + scipy only).

Two encodings of the same two-crew restoration problem, trained the same way:

  CrewCut   one qubit per repair (s_j = crew A or B), cost = sum over same-crew pairs of J_ij
            with J_ij = p_i p_j min(r_i, r_j), r_j = w_j / p_j   (Smith's rule), no penalty.
  Standard  one qubit per repair and crew (x_jc), same pair cost on each crew plus the penalty
            lambda * sum_j (x_j0 + x_j1 - 1)^2.  Bound = max_j sum_i J_ij; run at 1/8 ... 4 times it.

Trainer for both: X mixer, uniform start, minimise the expected normalised cost with COBYLA from
6 random starts, depth 1 and 2.  Metrics are scored on the true schedule cost.

This is NOT the research code behind the paper. It generates one illustrative 6-repair instance
(and a 12-instance replication) so the demo can draw exact probability maps.
"""
import argparse
import json
import time

import numpy as np
from scipy.optimize import minimize

N = 6
MULTS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]


# ---------------------------------------------------------------- instance
def make_instance(seed, n=N):
    rng = np.random.default_rng(seed)
    p = rng.integers(1, 9, size=n).astype(float)    # repair hours 1..8
    w = rng.integers(1, 11, size=n).astype(float)   # priority weight 1..10
    return p, w


def pair_matrix(p, w):
    r = w / p
    n = len(p)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                J[i, j] = p[i] * p[j] * min(r[i], r[j])
    return J


def schedule_cost_direct(p, w, plan):
    """Priority-weighted completion time, each crew in decreasing urgency (independent check)."""
    r = w / p
    total = 0.0
    for crew in (0, 1):
        jobs = [j for j in range(len(p)) if plan[j] == crew]
        jobs.sort(key=lambda j: -r[j])
        t = 0.0
        for j in jobs:
            t += p[j]
            total += w[j] * t
    return total


def crewcut_costs(p, w):
    """Cost of every plan; bit j of the index is 1 when repair j goes to crew B."""
    n = len(p)
    J = pair_matrix(p, w)
    const = float(np.sum(w * p))
    out = np.zeros(2 ** n)
    for s in range(2 ** n):
        bits = [(s >> j) & 1 for j in range(n)]
        c = const
        for i in range(n):
            for j in range(i + 1, n):
                if bits[i] == bits[j]:
                    c += J[i, j]
        out[s] = c
    return out


def standard_diag(p, w, lam):
    """Cost of every bitstring of the one-hot form; bit 2j = repair j on crew A, bit 2j+1 = crew B."""
    n = len(p)
    J = pair_matrix(p, w)
    const = float(np.sum(w * p))
    size = 4 ** n
    idx = np.arange(size)
    x = np.array([[(idx >> (2 * j + c)) & 1 for c in (0, 1)] for j in range(n)])  # (n, 2, size)
    cost = np.full(size, const)
    for c in (0, 1):
        for i in range(n):
            for j in range(i + 1, n):
                cost += J[i, j] * x[i, c] * x[j, c]
    for j in range(n):
        cost += lam * (x[j, 0] + x[j, 1] - 1) ** 2
    return cost


def standard_valid_and_plan(n):
    size = 4 ** n
    idx = np.arange(size)
    valid = np.ones(size, dtype=bool)
    plan = np.zeros(size, dtype=int)
    for j in range(n):
        a = (idx >> (2 * j)) & 1
        b = (idx >> (2 * j + 1)) & 1
        valid &= (a + b) == 1
        plan |= b << j
    return valid, plan


# ---------------------------------------------------------------- QAOA
def evolve(diag_norm, nq, gammas, betas):
    psi = np.full(2 ** nq, 1.0 / np.sqrt(2 ** nq), dtype=complex).reshape((2,) * nq)
    flat_shape = (2 ** nq,)
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * diag_norm).reshape((2,) * nq)
        c, s = np.cos(b), np.sin(b)
        for k in range(nq):
            psi = c * psi - 1j * s * np.flip(psi, axis=k)
    return psi.reshape(flat_shape)


def train(diag, nq, depth, rng, starts=6):
    lo, hi = diag.min(), diag.max()
    dn = (diag - lo) / (hi - lo)

    def energy(theta):
        psi = evolve(dn, nq, theta[:depth], theta[depth:])
        return float(np.sum(np.abs(psi) ** 2 * dn))

    best = None
    for _ in range(starts):
        x0 = np.concatenate([rng.uniform(0, 2 * np.pi, depth), rng.uniform(0, np.pi / 2, depth)])
        res = minimize(energy, x0, method="COBYLA", options={"maxiter": 250, "rhobeg": 0.4})
        if best is None or res.fun < best.fun:
            best = res
    psi = evolve(dn, nq, best.x[:depth], best.x[depth:])
    return np.abs(psi) ** 2, best.x


# ---------------------------------------------------------------- metrics
def metrics_crewcut(prob, costs):
    opt = costs.min()
    best = np.isclose(costs, opt)
    rand_mean = costs.mean()
    shot_mean = float(np.sum(prob * costs))
    return {
        "valid": 1.0,
        "best": float(prob[best].sum()),
        "gain": (rand_mean - shot_mean) / (rand_mean - opt),
    }


def metrics_standard(prob, costs, valid, plan):
    opt = costs.min()
    rand_mean = costs.mean()
    pv = prob[valid]
    tot = pv.sum()
    pc = costs[plan[valid]]
    best = np.isclose(pc, opt)
    shot_mean = float(np.sum(pv * pc) / tot) if tot > 0 else rand_mean
    return {
        "valid": float(tot),
        "best": float(pv[best].sum()),
        "gain": (rand_mean - shot_mean) / (rand_mean - opt),
    }


def run_instance(seed, depths=(1, 2), keep_probs=False):
    p, w = make_instance(seed)
    J = pair_matrix(p, w)
    costs = crewcut_costs(p, w)
    # independent check of the pair-cost identity against a direct schedule evaluation
    for s in range(2 ** N):
        plan = [(s >> j) & 1 for j in range(N)]
        assert abs(schedule_cost_direct(p, w, plan) - costs[s]) < 1e-9, "pair cost != schedule cost"
    bound = float(J.sum(axis=1).max())
    valid, plan = standard_valid_and_plan(N)
    rng = np.random.default_rng(1000 + seed)
    out = {"seed": seed, "p": p.tolist(), "w": w.tolist(), "bound": bound,
           "costs": costs.tolist(), "crewcut": {}, "standard": {}}
    for d in depths:
        prob, theta = train(costs, N, d, rng)
        out["crewcut"][str(d)] = {"m": metrics_crewcut(prob, costs)}
        if keep_probs:
            out["crewcut"][str(d)]["prob"] = prob.tolist()
        for m_ in MULTS:
            diag = standard_diag(p, w, m_ * bound)
            prob, theta = train(diag, 2 * N, d, rng)
            entry = {"m": metrics_standard(prob, costs, valid, plan)}
            if keep_probs:
                entry["prob"] = prob.tolist()
            out["standard"].setdefault(str(d), {})[str(m_)] = entry
    return out


def summarise(results):
    def med(vals):
        return float(np.median(vals))
    rows = {}
    for d in ("1", "2"):
        cc = [r["crewcut"][d]["m"]["best"] for r in results]
        at_bound = [r["standard"][d]["1.0"]["m"]["best"] for r in results]
        oracle = [max(v["m"]["best"] for v in r["standard"][d].values()) for r in results]
        rows[d] = {
            "random_valid": 2 / 2 ** N,
            "crewcut_best": med(cc),
            "standard_at_bound_best": med(at_bound),
            "standard_oracle_best": med(oracle),
            "crewcut_gain": med([r["crewcut"][d]["m"]["gain"] for r in results]),
            "standard_oracle_gain": med([
                max(r["standard"][d].values(), key=lambda v: v["m"]["best"])["m"]["gain"] for r in results]),
            "standard_at_bound_valid": med([r["standard"][d]["1.0"]["m"]["valid"] for r in results]),
            "crewcut_wins_per_instance": int(sum(a > b for a, b in zip(cc, oracle))),
        }
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--show", type=int, default=None, help="seed to export with full distributions")
    ap.add_argument("--out", default="demo_data.json")
    ap.add_argument("--summary", default="replication.json")
    a = ap.parse_args()
    if a.show is None:
        t0 = time.time()
        res = [run_instance(s) for s in range(a.seeds)]
        for r in res:
            print(r["seed"], "crewcut d2 best %.4f" % r["crewcut"]["2"]["m"]["best"],
                  "std oracle d2 best %.4f" % max(v["m"]["best"] for v in r["standard"]["2"].values()),
                  "%.0fs" % (time.time() - t0), flush=True)
        summ = summarise(res)
        json.dump({"summary": summ, "per_instance": [
            {"seed": r["seed"], "cc1": r["crewcut"]["1"]["m"]["best"], "cc2": r["crewcut"]["2"]["m"]["best"],
             "so1": max(v["m"]["best"] for v in r["standard"]["1"].values()),
             "so2": max(v["m"]["best"] for v in r["standard"]["2"].values())} for r in res]},
            open(a.summary, "w"), indent=1)
        print(json.dumps(summ, indent=1))
    else:
        r = run_instance(a.show, keep_probs=True)
        json.dump(r, open(a.out, "w"))
        print("wrote", a.out, "seed", a.show)
