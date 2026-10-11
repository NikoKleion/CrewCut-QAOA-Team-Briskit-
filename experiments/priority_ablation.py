import copy
import csv

import numpy as np

from briskit import benchmark
from briskit.feeder import all_plans
from briskit.flow import RESULTS_DIR, instance_names, restoration, stored_instance, stored_value_table

TOLERANCE = 1e-7
SCRIPT = "experiments/priority_ablation.py"
TIE_COLUMNS = ["instance", "family", "damaged", "count", "optimal_plans_weighted", "optimal_plans_unweighted",
               "shared_optimal_plan", "weighted_loss_best_tie", "weighted_loss_worst_tie",
               "unweighted_loss_best_tie", "unweighted_loss_worst_tie"]


def unweighted_copy(instance):
    flattened = copy.deepcopy(instance)
    flattened["name"] = instance["name"] + "_unweighted"
    for demand in flattened["demands"]:
        demand["weight"] = 1.0
    return flattened


def relative_loss(best, achieved):
    return float((best - achieved) / best) if best > TOLERANCE else float("nan")


def analyse(weighted, unweighted, feasible):
    best_weighted, best_unweighted = weighted[feasible].max(), unweighted[feasible].max()
    optimal_weighted = feasible[weighted[feasible] >= best_weighted - TOLERANCE]
    optimal_unweighted = feasible[unweighted[feasible] >= best_unweighted - TOLERANCE]
    at_unweighted_optima = weighted[optimal_unweighted]
    at_weighted_optima = unweighted[optimal_weighted]
    plans = {"weights on": int(optimal_weighted[0]),
             "weights off, best tie": int(optimal_unweighted[np.argmax(at_unweighted_optima)]),
             "weights off, worst tie": int(optimal_unweighted[np.argmin(at_unweighted_optima)])}
    ties = {"optimal_plans_weighted": len(optimal_weighted),
            "optimal_plans_unweighted": len(optimal_unweighted),
            "shared_optimal_plan": bool(np.intersect1d(optimal_weighted, optimal_unweighted).size),
            "weighted_loss_best_tie": relative_loss(best_weighted, at_unweighted_optima.max()),
            "weighted_loss_worst_tie": relative_loss(best_weighted, at_unweighted_optima.min()),
            "unweighted_loss_best_tie": relative_loss(best_unweighted, at_weighted_optima.max()),
            "unweighted_loss_worst_tie": relative_loss(best_unweighted, at_weighted_optima.min())}
    return plans, ties


def returns(plan):
    return lambda instance, budget, rng: all_plans(instance.size)[[plan]]


def main():
    rows, ties = [], []
    for name in instance_names():
        stored = stored_instance(name)
        damaged = len(stored["elements"])
        weighted, _ = stored_value_table(stored)
        unweighted, _ = stored_value_table(unweighted_copy(stored))
        sizes = all_plans(damaged).sum(axis=1)
        for count in range(1, damaged):
            feasible = np.flatnonzero(sizes == count)
            if weighted[feasible].max() <= TOLERANCE:
                continue
            plans, tie = analyse(weighted, unweighted, feasible)
            instance = restoration(stored, weighted, count)
            solvers = {method: returns(plan) for method, plan in plans.items()}
            rows += benchmark.run("E10", [instance], solvers, [1], [stored["seed"]], SCRIPT)
            ties.append({"instance": name, "family": stored["family"], "damaged": damaged, "count": count, **tie})
    benchmark.write_rows(rows, RESULTS_DIR / "priority_ablation.csv")
    with open(RESULTS_DIR / "priority_ablation_ties.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=TIE_COLUMNS)
        writer.writeheader()
        writer.writerows(ties)
    print("flow networks, priority weights on and off (E10): for each instance and number of links restored, the")
    print("plans that are optimal with every commodity weight set to 1, scored with the real weights")
    print(f"   {'damaged':>7s} {'restore':>7s} {'instances':>9s} {'share a plan':>12s} {'weighted loss, best tie':>24s}"
          f" {'weighted loss, worst tie':>25s}")
    for damaged, count in sorted({(t["damaged"], t["count"]) for t in ties}):
        group = [t for t in ties if (t["damaged"], t["count"]) == (damaged, count)]
        shared = sum(t["shared_optimal_plan"] for t in group)
        best = 100 * np.mean([t["weighted_loss_best_tie"] for t in group])
        worst = 100 * np.mean([t["weighted_loss_worst_tie"] for t in group])
        print(f"   {damaged:7d} {count:7d} {len(group):9d} {shared:12d} {best:23.2f}% {worst:24.2f}%")
    print("rows in results/flow/priority_ablation.csv, tie counts in results/flow/priority_ablation_ties.csv")


if __name__ == "__main__":
    main()
