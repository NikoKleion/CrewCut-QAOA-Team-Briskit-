from pathlib import Path

import numpy as np

from briskit import benchmark, summary
from briskit.feeder import sample_instance, upstream_parents
from briskit.instances import crew_assignment, restoration
from briskit.solvers import (best_first, conversion, fitted, greedy, largest_weights, qaoa, random_plans, thermal)

SEED = 20261010
SAMPLES = 150
FINAL = 50
TOTALS = (50, 100, 150, 151, 160, 175, 200)
TARGETS = (0.05, 0.01, 0.0)
RESULTS = Path(__file__).resolve().parent.parent / "results"
SCRIPT = "experiments/budget_experiment.py"

FITTED = {
    "fitted cost, best first": fitted(SAMPLES, best_first),
    "fitted cost, thermal beta 2": fitted(SAMPLES, thermal(2.0)),
    "fitted cost, thermal beta 5": fitted(SAMPLES, thermal(5.0)),
    "fitted cost, QAOA depth 1": fitted(SAMPLES, qaoa(1)),
    "fitted cost, QAOA depth 2": fitted(SAMPLES, qaoa(2)),
    "fitted cost, QAOA depth 3": fitted(SAMPLES, qaoa(3)),
}


def feeder_draw(k):
    return sample_instance(np.random.default_rng([SEED, k]), 12)


def restoration_instance(k):
    weights, _ = feeder_draw(k)
    return restoration(f"restore 6 of 12, instance {k}", weights, upstream_parents(), 6)


def crew_instance(k):
    weights, hours = feeder_draw(k)
    return crew_assignment(f"two crews, 12 repairs, instance {k}", weights, hours, upstream_parents())


def collect(experiment, make_instance, rules, instances):
    rows = []
    for k in range(instances):
        instance = make_instance(k)
        rows += benchmark.run(experiment, [instance], FITTED, range(FINAL + 1), [k], SCRIPT,
                              prefix_safe=set(FITTED), trace=True)
        rows += benchmark.run(experiment, [instance], {"random plans": random_plans},
                              range(1, SAMPLES + FINAL + 1), [k], SCRIPT, prefix_safe={"random plans"})
        rows += benchmark.run(experiment, [instance], rules, [1], [k], SCRIPT)
    return benchmark.changes_only(rows)


def report(title, rows, methods):
    runs = summary.by_run(rows)
    print(title)
    print("regret at a total number of evaluations: median (lower to upper quartile), mean, in percent")
    print(f"   {'method':30s}" + "".join(f"{total:>26d}" for total in TOTALS))
    for method in methods:
        cells = []
        for total in TOTALS:
            s = summary.regret_summary(runs, method, total)
            cells.append(f"{'none':>26s}" if s is None else
                         f"{100 * s['median']:6.2f} ({100 * s['lower']:5.2f} to {100 * s['upper']:5.2f}) {100 * s['mean']:5.2f}")
        print(f"   {method:30s}" + "".join(cells))
    print("instances that reach a regret target within the run, and average runtime in evaluations")
    print(f"   {'method':30s}" + "".join(f"{f'regret <= {100 * target:g}%':>24s}" for target in TARGETS))
    for method in methods:
        cells = []
        for target in TARGETS:
            r = summary.runtime_to_target(runs, method, target)
            average = "none" if r["average"] is None else f"{r['average']:.0f}"
            cells.append(f"{r['reached']:>11d} of {r['runs']:2d}, {average:>6s}")
        print(f"   {method:30s}" + "".join(cells))


def main(instances=30):
    restoring = collect("budget experiment, restoration", restoration_instance,
                        {"greedy": greedy, "largest weights": largest_weights}, instances)
    benchmark.write_rows(restoring, RESULTS / "budget_experiment_restoration.csv")
    report(f"IEEE 13 node feeder, restore 6 of 12 lines, {instances} instances, fit on {SAMPLES} plans",
           restoring, ["random plans", *FITTED, "greedy", "largest weights"])
    print()
    crews = collect("budget experiment, crews", crew_instance, {"conversion algorithm": conversion}, instances)
    benchmark.write_rows(crews, RESULTS / "budget_experiment_crews.csv")
    report(f"IEEE 13 node feeder, 12 repairs on two crews, {instances} instances, fit on {SAMPLES} plans",
           crews, ["random plans", *FITTED, "conversion algorithm"])


if __name__ == "__main__":
    main()
