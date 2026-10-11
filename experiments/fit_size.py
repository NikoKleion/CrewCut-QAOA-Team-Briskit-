from math import comb
from pathlib import Path

import numpy as np

from briskit import benchmark, summary
from briskit.feeder import sample_instance, upstream_parents
from briskit.instances import restoration
from briskit.solvers import best_first, fitted, random_plans

SEED = 20261010
SIZES = (20, 40, 60, 80, 100, 150, 200, 300)
EXTRA = 10
RESULTS = Path(__file__).resolve().parent.parent / "results" / "feeder"
SCRIPT = "experiments/fit_size.py"


def feeder_instance(k, count):
    weights, _ = sample_instance(np.random.default_rng([SEED, k]), 12)
    return restoration(f"restore {count} of 12, instance {k}", weights, upstream_parents(), count)


def labels(count):
    every = comb(12, count)
    return [(size, str(size)) for size in SIZES] + [(every, "every plan")]


def methods(count):
    solvers = {}
    for size, label in labels(count):
        solvers[f"pair terms, fit on {label}"] = fitted(size, best_first)
        solvers[f"single terms, fit on {label}"] = fitted(size, best_first, pairs=False)
    return solvers


def collect(count, instances):
    experiment = f"fit size, restore {count} of 12"
    solvers = methods(count)
    rows, optimum = [], {}
    for k in range(instances):
        instance = feeder_instance(k, count)
        optimum[instance.name] = instance.optimum()
        rows += benchmark.run(experiment, [instance], solvers, [0, 1, EXTRA], [k], SCRIPT, prefix_safe=set(solvers))
        rows += benchmark.run(experiment, [instance], {"random plans": random_plans},
                              [size + EXTRA for size in SIZES], [k], SCRIPT, prefix_safe={"random plans"})
    return benchmark.changes_only(rows), optimum


def spread(values):
    lower, median, upper = np.percentile(values, [25, 50, 75])
    return f"{100 * median:7.2f} ({100 * lower:5.2f} to {100 * upper:6.2f}) {100 * np.mean(values):6.2f}"


def report(count, rows, optimum, instances):
    runs = summary.by_run(rows)
    model = {}
    for row in rows:
        if row["budget"] == 1 and row["method"] != "random plans":
            model.setdefault(row["method"], []).append(1 - row["returned"] / optimum[row["instance"]])
    print(f"IEEE 13 node feeder, restore {count} of 12 lines, {comb(12, count)} plans of that size, {instances} instances")
    print("regret in percent: median (lower to upper quartile) mean")
    print("regret of the fitted cost's own best plan")
    print(f"   {'plans in the fit':18s}{'single and pair terms':>32s}{'single terms only':>32s}")
    for size, label in labels(count):
        print(f"   {label:18s}{spread(model[f'pair terms, fit on {label}']):>32s}{spread(model[f'single terms, fit on {label}']):>32s}")
    print(f"regret of the run after the fit and {EXTRA} more plans, and of random plans at the same total")
    print(f"   {'plans in the fit':18s}{'single and pair terms':>32s}{'single terms only':>32s}{'random plans':>32s}")
    for size, label in labels(count)[:-1]:
        cells = []
        for method in (f"pair terms, fit on {label}", f"single terms, fit on {label}", "random plans"):
            s = summary.regret_summary(runs, method, size + EXTRA)
            cells.append(f"{100 * s['median']:7.2f} ({100 * s['lower']:5.2f} to {100 * s['upper']:6.2f}) {100 * s['mean']:6.2f}")
        print(f"   {label:18s}" + "".join(f"{cell:>32s}" for cell in cells))


def main(instances=30):
    for count in (4, 6):
        rows, optimum = collect(count, instances)
        benchmark.write_rows(rows, RESULTS / f"fit_size_restore_{count}.csv")
        report(count, rows, optimum, instances)
        print()


if __name__ == "__main__":
    main()
