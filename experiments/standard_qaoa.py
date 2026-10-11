import os
from multiprocessing import Pool

import numpy as np

from briskit.feeder import sample_instance
from briskit.instances import crew_assignment
from briskit.one_hot import assignments, cost_table, one_hot_circuit, one_hot_terms, penalty_bound, penalty_bounds
from briskit.qaoa import qaoa_circuit, train_qaoa
from briskit.scheduling import smith_order

SEED = 20261010
DEPTHS = (1, 2)
STARTS = 6
# kind of penalty and its share of the bound: stream tag of the trainer, label
VARIANTS = {
    ("uniform", 1.0): (1, "one penalty at the bound"),
    ("uniform", 0.5): (2, "one penalty at 1/2 of the bound"),
    ("uniform", 0.25): (3, "one penalty at 1/4 of the bound"),
    ("uniform", 0.125): (4, "one penalty at 1/8 of the bound"),
    ("own", 1.0): (5, "each repair's penalty at its own bound"),
    ("own", 0.5): (6, "each repair's penalty at half its own bound"),
    ("uniform", 2.0): (7, "one penalty at 2 times the bound"),
    ("uniform", 4.0): (8, "one penalty at 4 times the bound"),
}
REGISTERED = [("uniform", 1.0), ("uniform", 0.5), ("uniform", 0.25), ("uniform", 0.125)]
ADDED = [("uniform", 2.0), ("uniform", 4.0), ("own", 1.0), ("own", 0.5)]
RUN = {6: (REGISTERED, ADDED), 8: (REGISTERED[:1], ADDED[:3])}


def pair_gates(circuit):
    return sum(1 for inst in circuit.data if inst.operation.num_qubits == 2)


def gate_counts(weights, hours):
    n = len(weights)
    order = smith_order(weights, hours)
    pairs = {(order[a], order[b]): weights[order[b]] * hours[order[a]] for a in range(n) for b in range(a + 1, n)}
    _, single, pair = one_hot_terms(weights, hours, order, 2, penalty_bound(weights, hours, order))
    return (n, pair_gates(qaoa_circuit(n, pairs, 1.0, [0.1, 0.1])),
            2 * n, pair_gates(one_hot_circuit(single, pair, [0.1, 0.1])))


def shares(probability, cost, valid, best, random_mean):
    inside = probability[valid]
    share = float(inside.sum())
    hit = float(inside[cost[valid] <= best * (1 + 1e-12)].sum())
    kept = float((random_mean - inside @ cost[valid] / share) / (random_mean - best))
    return share, hit, kept


def one_instance(job):
    n, k = job
    valid, _ = assignments(n, 2)
    weights, hours = sample_instance(np.random.default_rng([SEED, n, k]), n)
    order = smith_order(weights, hours)
    project = crew_assignment(f"{n} repairs, instance {k}", weights, hours).table()
    best, random_mean = project.min(), project.mean()
    bounds = penalty_bounds(weights, hours, order)
    rows = {}
    for depth in DEPTHS:
        _, probability = train_qaoa(project, depth, np.random.default_rng([SEED, n, k, depth, 0]), starts=STARTS)
        rows[depth, "project"] = shares(probability, project, np.ones(project.size, dtype=bool), best, random_mean)
        for variant in RUN[n][0] + RUN[n][1]:
            kind, factor = variant
            penalty = factor * (bounds.max() if kind == "uniform" else bounds)
            table = cost_table(*one_hot_terms(weights, hours, order, 2, penalty))
            stream = np.random.default_rng([SEED, n, k, depth, VARIANTS[variant][0]])
            _, probability = train_qaoa(table, depth, stream, starts=STARTS)
            rows[depth, variant] = shares(probability, table, valid, best, random_mean)
    return rows


def line(label, qubits, gates, rows):
    share, hit, kept = (np.percentile([row[i] for row in rows], [25, 50, 75]) for i in range(3))
    needed = np.log(0.05) / np.log1p(-hit[1])
    return (f"   {label:50s} {qubits:6d} {gates:11d}   {100 * share[1]:6.2f}%"
            f"   {100 * hit[1]:6.2f}% [{100 * hit[0]:5.2f}, {100 * hit[2]:5.2f}]   {needed:9.0f}   {100 * kept[1]:6.1f}%")


def best_of(results, depth, variants):
    picked = [max(variants, key=lambda variant: r[depth, variant][1]) for r in results]
    return picked, [r[depth, variant] for r, variant in zip(results, picked)]


def ahead(project, other):
    return sum(p[1] > s[1] for p, s in zip(project, other))


def main(sizes=(6, 8), instances=30):
    # project choice: at most 6 workers, each loads NumPy, SciPy and Qiskit
    processes = min(int(os.environ.get("FLOW_PROCESSES", 1)), 6)
    for n in sizes:
        jobs = [(n, k) for k in range(instances)]
        if processes == 1:
            results = [one_instance(job) for job in jobs]
        else:
            with Pool(processes) as pool:
                results = pool.map(one_instance, jobs)
        weights, hours = sample_instance(np.random.default_rng([SEED, n, 0]), n)
        qubits, gates, standard_qubits, standard_gates = gate_counts(weights, hours)
        registered, added = RUN[n]
        for depth in DEPTHS:
            project = [r[depth, "project"] for r in results]
            at_bound = [r[depth, ("uniform", 1.0)] for r in results]
            picked, tuned = best_of(results, depth, registered + added)
            print(f"{n} repairs, two crews, {instances} instances, depth {depth}, {STARTS} starts, median over instances"
                  f" [quartiles]")
            print("   form                                               qubits  pair gates    valid   best plan shots"
                  "            shots for   gain kept,")
            print("                                                               per layer    shots                  "
                  "            95%         valid shots")
            print(line("one qubit per repair", qubits, gates, project))
            print("   one qubit per repair and crew:")
            print(line("   " + VARIANTS["uniform", 1.0][1], standard_qubits, standard_gates, at_bound))
            if len(registered) > 1:
                print(line(f"   best of the {len(registered)} penalties at or below the bound", standard_qubits, standard_gates,
                           best_of(results, depth, registered)[1]))
            for variant in added:
                print(line("   " + VARIANTS[variant][1], standard_qubits, standard_gates,
                           [r[depth, variant] for r in results]))
            print(line("   best of all penalties for each instance", standard_qubits, standard_gates, tuned))
            print(f"   more best plan shots with one qubit per repair: {ahead(project, at_bound)} of {instances} instances"
                  f" against one penalty at the bound, {ahead(project, tuned)} of {instances} against the best of all")
            picks = "; ".join(f"{VARIANTS[variant][1]}: {picked.count(variant)}" for variant in registered + added)
            print(f"   picked for the best of all: {picks}")
        if len(registered) == 1:
            print(f"   at {n} repairs the penalties below the bound and at half of each repair's own bound are not run")
    weights, hours = sample_instance(np.random.default_rng([SEED, 12, 0]), 12)
    qubits, gates, standard_qubits, standard_gates = gate_counts(weights, hours)
    print(f"12 repairs, counted from the circuits, not simulated: {qubits} qubits and {gates} pair gates per layer"
          f" against {standard_qubits} and {standard_gates}")
    print("the best of several penalties is picked by the share of best plan shots, which uses the answer")
    print("penalties above the bound and penalties for each repair were added after the first result at 6 repairs")


if __name__ == "__main__":
    main()
