import os
from multiprocessing import Pool

import numpy as np

from briskit import benchmark, summary
from briskit.flow import RESULTS_DIR, instance_names, restoration, stored_instance, stored_value_table
from briskit.solvers import best_first, fitted, greedy, qaoa, random_plans, thermal

SEEDS = 30
SAMPLES = {12: 150, 16: 400}  # project choice: 150 as in experiments/budget_experiment.py, 400 from Jithesh's fit sizes
FINAL = 50
TARGETS = (0.05, 0.01, 0.0)
ANNEALING_START, ANNEALING_END = 0.1, 0.005  # project choice: Jithesh's schedule, as a share of the current value
SCRIPT = "experiments/flow_budget_experiment.py"


def fitted_methods(samples):
    return {
        "fitted cost, best first": fitted(samples, best_first),
        "fitted cost, thermal beta 2": fitted(samples, thermal(2.0)),
        "fitted cost, thermal beta 5": fitted(samples, thermal(5.0)),
        "fitted cost, QAOA depth 1": fitted(samples, qaoa(1, at_bound=True)),
        "fitted cost, QAOA depth 2": fitted(samples, qaoa(2, at_bound=True)),
    }


def swap_annealing(instance, budget, rng):
    plan = np.zeros(instance.size, dtype=np.int8)
    plan[rng.choice(instance.size, instance.count, replace=False)] = 1
    current = float(instance.score(plan)[0])
    visited = [plan]
    for step in range(budget - 1):
        temperature = ANNEALING_START * (ANNEALING_END / ANNEALING_START) ** (step / max(budget - 2, 1))
        proposal = plan.copy()
        proposal[rng.choice(np.flatnonzero(plan == 1))] = 0
        proposal[rng.choice(np.flatnonzero(plan == 0))] = 1
        value = float(instance.score(proposal)[0])
        visited.append(proposal)
        if value >= current or rng.random() < np.exp((value - current) / (temperature * max(current, 1e-12))):
            plan, current = proposal, value
    return np.array(visited)


def collect(job):
    stored, table, count, seed = job
    instance = restoration(stored, table, count)
    samples = SAMPLES[instance.size]
    methods = fitted_methods(samples)
    total = range(1, samples + FINAL + 1)
    seeds = [seed]
    rows = benchmark.run("E8", [instance], methods, range(FINAL + 1), seeds, SCRIPT, prefix_safe=set(methods),
                         trace=True)
    rows += benchmark.run("E8", [instance], {"random plans": random_plans, "swap annealing": swap_annealing}, total,
                          seeds, SCRIPT, prefix_safe={"random plans", "swap annealing"}, trace=True)
    rows += benchmark.run("E8", [instance], {"greedy": greedy}, [1], seeds, SCRIPT)
    return benchmark.changes_only(rows)


def report(title, rows, methods, totals, width=30):
    runs = summary.by_run(rows)
    print(title)
    print("regret at a total number of evaluations: median (lower to upper quartile), mean, in percent")
    print(f"   {'method':{width}s}" + "".join(f"{total:>34d}" for total in totals))
    for method in methods:
        cells = []
        for total in totals:
            s = summary.regret_summary(runs, method, total)
            cells.append(f"{'none':>34s}" if s is None else
                         f"{100 * s['median']:9.2f} ({100 * s['lower']:5.2f} to {100 * s['upper']:6.2f}) {100 * s['mean']:6.2f}")
        print(f"   {method:{width}s}" + "".join(cells))
    print("runs that reach a regret target, and average runtime in evaluations")
    print(f"   {'method':{width}s}" + "".join(f"{f'regret <= {100 * target:g}%':>24s}" for target in TARGETS))
    for method in methods:
        cells = []
        for target in TARGETS:
            r = summary.runtime_to_target(runs, method, target)
            average = "none" if r["average"] is None else f"{r['average']:.0f}"
            cells.append(f"{r['reached']:>11d} of {r['runs']:3d}, {average:>6s}")
        print(f"   {method:{width}s}" + "".join(cells))


def main():
    jobs = []
    for name in instance_names():
        stored = stored_instance(name)
        table, _ = stored_value_table(stored)
        for count in stored["budget"]:
            if restoration(stored, table, count).optimum() <= 0:
                print(f"skipped {name}, restore {count}: no plan of this size serves any demand")
            else:
                jobs += [(stored, table, count, seed) for seed in range(SEEDS)]
    # project choice: at most 6 workers, each loads NumPy, SciPy and Qiskit, as in experiments/standard_qaoa.py
    processes = min(int(os.environ.get("FLOW_PROCESSES", 1)), 6)
    if processes == 1:
        collected = [collect(job) for job in jobs]
    else:
        with Pool(processes) as pool:
            collected = pool.map(collect, jobs, chunksize=1)
    rows = [row for part in collected for row in part]
    benchmark.write_rows(rows, RESULTS_DIR / "budget_experiment.csv")
    groups = sorted({(len(stored["elements"]), count) for stored, _, count, _ in jobs})
    for damaged, count in groups:
        picked = [row for (stored, _, c, _), part in zip(jobs, collected) if (len(stored["elements"]), c) == (damaged, count)
                  for row in part]
        samples = SAMPLES[damaged]
        totals = (samples // 3, 2 * samples // 3, samples, samples + 1, samples + 10, samples + 25, samples + FINAL)
        instances = len({row["instance"] for row in picked})
        report(f"flow networks, restore {count} of {damaged} damaged links, {instances} instances, {SEEDS} seeds, "
               f"fit on {samples} plans", picked,
               ["random plans", "swap annealing", *fitted_methods(samples), "greedy"], totals)
        print()


if __name__ == "__main__":
    main()
