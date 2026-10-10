import csv
import time

import numpy as np

from briskit.feeder import all_plans
from briskit.flow import RESULTS_DIR, instance_names, stored_instance, stored_value_table
from briskit.pair_cost import ridge_coefficients, selection_features

SEED = 20261010
FIT_SIZES = (10, 20, 40, 80, 160, 320)  # project choice: doubling steps, as plan E4 asks for a regret curve
DRAWS = 10  # project choice, no source
SCRIPT = "experiments/flow_fitted_pair_cost.py"
FIELDS = ["experiment", "instance", "method", "budget", "seed", "value", "regret", "returned", "evaluations",
          "seconds", "script"]


def fitted_runs(experiment, problem, value, models, rows, draw, fit_label, group):
    optimum = value.max()
    sample_best = value[rows].max()
    runs = []
    for name, features in models.items():
        start = time.perf_counter()
        pick = int(np.argmax(features @ ridge_coefficients(features[rows], value[rows])))
        seconds = time.perf_counter() - start
        best = max(sample_best, value[pick])
        method = f"{name}, fit on {fit_label}"
        common = {"experiment": experiment, "instance": problem, "method": method, "seed": draw, "script": SCRIPT,
                  "optimum": optimum, **group}
        runs.append({**common, "budget": 0, "value": sample_best, "regret": 1 - sample_best / optimum,
                     "returned": "", "evaluations": len(rows), "seconds": 0.0})
        runs.append({**common, "budget": 1, "value": best, "regret": 1 - best / optimum, "returned": value[pick],
                     "evaluations": len(set(rows.tolist()) | {pick}), "seconds": seconds})
    return runs


def main():
    runs = []
    for name in instance_names():
        instance = stored_instance(name)
        m = len(instance["elements"])
        table, _ = stored_value_table(instance)
        plans = all_plans(m)
        size = plans.sum(axis=1)
        for count in instance["budget"]:
            feasible = np.where(size == count)[0]
            value = table[feasible]
            problem = f"{name}_restore{count}"
            if value.max() <= 0:
                print(f"   skipped {problem}: no plan of this size serves any demand, so regret is undefined")
                continue
            rng = np.random.default_rng([SEED, *problem.encode()])
            group = {"family": instance["family"], "damaged": m, "count": count}
            models = {"single-link terms": selection_features(plans[feasible], pairs=False),
                      "single and pair terms": selection_features(plans[feasible])}
            runs += fitted_runs("E3", problem, value, models, np.arange(len(feasible)), 0, "all plans", group)
            for fit_size in FIT_SIZES:
                for draw in range(DRAWS):
                    rows = rng.choice(len(feasible), size=fit_size, replace=False)
                    runs += fitted_runs("E4", problem, value, models, rows, draw, f"{fit_size} plans", group)
    with open(RESULTS_DIR / "fitted_pair_cost.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)
    summarize(runs)


def summarize(runs):
    print("flow networks, restore a fixed number of damaged links; value lost against the optimum, mean over"
          f" 3 seeds and {DRAWS} draws")
    print("each cell: plan the fit returns / best plan scored in the run (budget 1) / fitting sample alone (budget 0)")
    families = sorted({r["family"] for r in runs})
    methods = list(dict.fromkeys(r["method"] for r in runs))
    for damaged, count in sorted({(r["damaged"], r["count"]) for r in runs}):
        print(f"   {damaged} damaged links, restore {count}")
        print(f"      {'method':40s} " + " ".join(f"{f:>24s}" for f in families))
        for method in methods:
            cells = []
            for family in families:
                picked = [r for r in runs if (r["family"], r["damaged"], r["count"], r["method"])
                          == (family, damaged, count, method)]
                one = [r for r in picked if r["budget"] == 1]
                zero = [r for r in picked if r["budget"] == 0]
                returned = np.mean([1 - r["returned"] / r["optimum"] for r in one])
                cells.append(f"{100 * returned:6.1f} /{100 * np.mean([r['regret'] for r in one]):5.1f} /"
                             f"{100 * np.mean([r['regret'] for r in zero]):5.1f}%")
            print(f"      {method:40s} " + " ".join(f"{c:>24s}" for c in cells))


if __name__ == "__main__":
    main()
