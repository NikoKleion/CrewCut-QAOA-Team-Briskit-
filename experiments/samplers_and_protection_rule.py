import itertools

import numpy as np

from briskit.baselines import matched_thermal, thermal_distributions
from briskit.feeder import all_plans, sample_instance, upstream_parents
from briskit.pair_cost import fit_pair_cost, pair_features, protection_terms
from briskit.qaoa import expected_best_of, train_qaoa
from briskit.scheduling import energization_cost, single_crew_order

SEED = 20261010


def samplers_on_fitted_cost(rng, instances=20, samples=150, depth=3):
    parent = upstream_parents()
    n = len(parent)
    plans = all_plans(n)
    features = pair_features(plans)
    betas = np.geomspace(0.05, 20, 40)
    table = {}
    add = lambda name, value: table.setdefault(name, []).append(value)
    for _ in range(instances):
        w, p = sample_instance(rng, n)
        true = energization_cost(plans, single_crew_order(w, p, parent), w, p, parent)
        best = true.min()
        rows = rng.choice(true.size, size=samples, replace=False)
        fit = fit_pair_cost(features, true, rows)
        _, prob = train_qaoa(fit, depth, rng, starts=2)
        thermal = thermal_distributions(fit, betas)
        matched = matched_thermal(fit, betas, float(prob @ fit))
        for shots in (10, 100):
            add(f"QAOA depth {depth}, best of {shots} shots", expected_best_of(prob, true, shots) / best - 1)
            add(f"thermal sampler at the same mean fitted cost, best of {shots}",
                expected_best_of(matched, true, shots) / best - 1)
            add(f"thermal sampler at its best temperature, best of {shots}",
                min(expected_best_of(x, true, shots) for x in thermal) / best - 1)
            add(f"fitted cost, its {shots} best plans", true[np.argsort(fit)[:shots]].min() / best - 1)
    return table


def single_line_rule(rng, low, high, protect, n=12, instances=200):
    subsets = list(itertools.combinations(range(n), protect))
    unprotected = np.ones((len(subsets), n))
    for r, subset in enumerate(subsets):
        unprotected[r, list(subset)] = 0.0
    regret, differ = [], 0
    for _ in range(instances):
        w, p = sample_instance(rng, n)
        failure = rng.uniform(low, high, n)
        single, pair = protection_terms(w, p, failure)
        cost = unprotected @ single + 0.5 * np.einsum("ri,ij,rj->r", unprotected, pair, unprotected)
        rule = tuple(sorted(np.argsort(-single)[:protect]))
        regret.append(cost[subsets.index(rule)] / cost.min() - 1)
        differ += subsets[int(np.argmin(cost))] != rule
    regret = np.array(regret)
    return regret.mean(), regret.max(), differ, instances


def main():
    rng = np.random.default_rng(SEED)
    print("IEEE 13 node feeder, 12 repairs, 20 instances, pair cost fitted on 150 plans, mean percent above the best plan")
    for name, values in samplers_on_fitted_cost(rng).items():
        print(f"   {name:62s} {100 * np.mean(values):7.3f}%")
    print("protection, one crew, independent lines: single-line rule against the optimum of the expected cost")
    for low, high, protect in ((0.05, 0.6, 4), (0.5, 1.0, 4), (0.5, 1.0, 6), (0.9, 1.0, 6)):
        mean, worst, differ, instances = single_line_rule(rng, low, high, protect)
        print(f"   failure chance {low:.2f} to {high:.2f}, protect {protect} of 12: mean +{100 * mean:.2f}%,"
              f" worst +{100 * worst:.2f}%, optimal set differs in {differ} of {instances}")


if __name__ == "__main__":
    main()
