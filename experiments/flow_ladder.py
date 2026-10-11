import os
from multiprocessing import Pool

import numpy as np

from briskit import benchmark
from briskit.cost_circuit import ising_terms
from briskit.flow import RESULTS_DIR, instance_names, restoration, stored_instance, stored_value_table
from briskit.solvers import best_first, cost_at_bound, fit_on_sample, fitted_merit, qaoa, random_plans
from experiments.flow_budget_experiment import FINAL, SAMPLES, report

SEEDS = 30
DAMAGED = 12
DEPTHS = (1, 2)
KEEP = (100, 50, 25)  # shares of pair terms kept, as in the keep setting of qollab/main.py
SCRIPT = "experiments/flow_ladder.py"


def kept_terms(coefficients, n, keep):
    pair = np.array(coefficients[n + 1:], dtype=float)
    pair[np.argsort(-np.abs(pair), kind="stable")[round(keep / 100 * len(pair)):]] = 0.0
    return np.concatenate([coefficients[:n + 1], pair])


def sparse_fitted(samples, keep, select, pairs=True):
    def solve(instance, budget, rng):
        coefficients, rows, plans = fit_on_sample(instance, samples, rng, pairs)
        merit = fitted_merit(kept_terms(coefficients, instance.size, keep), plans, instance.sense)
        return plans[select(merit, rows, instance, budget, rng)]
    return solve


def ladder(samples):
    methods = {"single-link fit, best first": sparse_fitted(samples, 100, best_first, pairs=False),
               "pair fit, best first": sparse_fitted(samples, 100, best_first)}
    shapes = {}
    for depth in DEPTHS:
        methods[f"single-link fit, QAOA depth {depth}"] = sparse_fitted(samples, 100, qaoa(depth, at_bound=True),
                                                                       pairs=False)
        shapes[f"single-link fit, QAOA depth {depth}"] = (100, False)
        for keep in KEEP:
            label = "pair fit" if keep == 100 else f"pair fit, {keep}% of pair terms"
            methods[f"{label}, QAOA depth {depth}"] = sparse_fitted(samples, keep, qaoa(depth, at_bound=True))
            shapes[f"{label}, QAOA depth {depth}"] = (keep, True)
    return methods, shapes


def collect(job):
    stored, table, count, seed = job
    instance = restoration(stored, table, count)
    methods, _ = ladder(SAMPLES[instance.size])
    rows = benchmark.run("B9", [instance], methods, range(FINAL + 1), [seed], SCRIPT, prefix_safe=set(methods),
                         trace=True)
    rows += benchmark.run("B9", [instance], {"random plans": random_plans}, range(1, SAMPLES[instance.size] + FINAL + 1),
                          [seed], SCRIPT, prefix_safe={"random plans"}, trace=True)
    return benchmark.changes_only(rows)


def pair_gates(instance, keep, pairs):
    rng = benchmark.stream(0, instance.name, "gate count")
    coefficients, rows, plans = fit_on_sample(instance, SAMPLES[instance.size], rng, pairs)
    coefficients = kept_terms(coefficients, instance.size, keep)
    _, penalty = cost_at_bound(fitted_merit(coefficients, plans, instance.sense), rows, instance)
    with_penalty = ising_terms(coefficients, instance.size, instance.sense, penalty, instance.count)[1]
    without = ising_terms(coefficients, instance.size, instance.sense)[1]
    return int(np.count_nonzero(with_penalty)), int(np.count_nonzero(without))


def main():
    jobs, instances = [], []
    for name in instance_names():
        stored = stored_instance(name)
        if len(stored["elements"]) != DAMAGED:
            continue
        table, _ = stored_value_table(stored)
        for count in stored["budget"]:
            instances.append(restoration(stored, table, count))
            jobs += [(stored, table, count, seed) for seed in range(SEEDS)]
    # project choice: at most 6 workers, each loads NumPy, SciPy and Qiskit, as in experiments/standard_qaoa.py
    processes = min(int(os.environ.get("FLOW_PROCESSES", 1)), 6)
    if processes == 1:
        collected = [collect(job) for job in jobs]
    else:
        with Pool(processes) as pool:
            collected = pool.map(collect, jobs, chunksize=1)
    rows = [row for part in collected for row in part]
    benchmark.write_rows(rows, RESULTS_DIR / "ladder.csv")
    samples = SAMPLES[DAMAGED]
    methods, shapes = ladder(samples)
    totals = (samples // 3, 2 * samples // 3, samples, samples + 1, samples + 10, samples + 25, samples + FINAL)
    for count in sorted({instance.count for instance in instances}):
        group = [instance for instance in instances if instance.count == count]
        picked = [row for row in rows if row["instance"] in {instance.name for instance in group}]
        report(f"flow networks, rungs 2 and 4, restore {count} of {DAMAGED} damaged links, {len(group)} instances,"
               f" {SEEDS} seeds, fit on {samples} plans", picked, ["random plans", *methods], totals, width=45)
        print(f"   pair gates per cost layer, {DAMAGED} qubits, median over the instances")
        print(f"   {'method':45s} {'with the count penalty':>24s} {'without it':>12s}")
        for method, (keep, pairs) in shapes.items():
            if not method.endswith("depth 1"):
                continue
            counted = np.array([pair_gates(instance, keep, pairs) for instance in group])
            print(f"   {method.removesuffix(', QAOA depth 1'):45s} {np.median(counted[:, 0]):24.0f}"
                  f" {np.median(counted[:, 1]):12.0f}")
        print()
    print("the count penalty puts a pair term on every pair of links, so dropping weak fitted pair terms does not"
          " remove pair gates here; the column without it is what a mixer that keeps the count would need for the"
          " cost layer, before the gates of that mixer")


if __name__ == "__main__":
    main()
