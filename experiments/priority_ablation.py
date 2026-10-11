import argparse
import copy
import csv
import hashlib
import json
from multiprocessing import Pool

import numpy as np

from briskit.feeder import all_plans
from briskit.flow import RESULTS_DIR, FlowProgram, instance_names, stored_instance

EXPERIMENT_ID = "E10"
SCRIPT_NAME = "priority_ablation.py"
TOLERANCE = 1e-7
OUTPUT_DIR = RESULTS_DIR / "priority_ablation"
ROW_COLUMNS = ["experiment", "instance", "method", "budget", "seed", "value", "regret", "script"]
SUMMARY_COLUMNS = ["instance", "family", "damaged", "budget", "optimal_plans_weighted", "optimal_plans_unweighted",
                   "shared_optimal_plan", "weighted_loss_best_tie", "weighted_loss_worst_tie",
                   "unweighted_loss_best_tie", "unweighted_loss_worst_tie"]


def unweighted_copy(instance):
    flattened = copy.deepcopy(instance)
    flattened["name"] = instance["name"] + "_unweighted"
    for demand in flattened["demands"]:
        demand["weight"] = 1.0
    return flattened


def plan_values(arguments):
    instance, plans = arguments
    program = FlowProgram(instance)
    return [program.value(plan) for plan in plans]


def values_of_plans(instance, plans, processes):
    if processes == 1 or len(plans) < 64:
        return np.array(plan_values((instance, plans)))
    pieces = np.array_split(plans, min(256, len(plans)))
    with Pool(processes) as pool:
        return np.concatenate(pool.map(plan_values, [(instance, piece) for piece in pieces]))


def cached_values(instance, plans, subset, processes):
    fingerprint = hashlib.sha1(json.dumps(instance, sort_keys=True).encode()).hexdigest()[:10]
    subset_fingerprint = hashlib.sha1(subset.tobytes()).hexdigest()[:10]
    path = OUTPUT_DIR / "cache" / f"{instance['name']}_{fingerprint}_{subset_fingerprint}.npy"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, values_of_plans(instance, plans[subset], processes))
    return np.load(path)


def filled_table(plan_count, subset, values):
    table = np.full(plan_count, np.nan)
    table[subset] = values
    return table


def relative_loss(best, achieved):
    if best <= TOLERANCE:
        return float("nan")
    return float((best - achieved) / best)


def analyse_budget(weighted, unweighted, plans, budget):
    feasible = np.flatnonzero(plans.sum(axis=1) == budget)
    best_weighted = weighted[feasible].max()
    best_unweighted = unweighted[feasible].max()
    optimal_weighted = feasible[weighted[feasible] >= best_weighted - TOLERANCE]
    optimal_unweighted = feasible[unweighted[feasible] >= best_unweighted - TOLERANCE]
    weighted_at_unweighted_optima = weighted[optimal_unweighted]
    unweighted_at_weighted_optima = unweighted[optimal_weighted]
    return {
        "best_weighted": best_weighted,
        "best_unweighted": best_unweighted,
        "weighted_plan": int(optimal_weighted[0]),
        "unweighted_best_tie_plan": int(optimal_unweighted[np.argmax(weighted_at_unweighted_optima)]),
        "unweighted_worst_tie_plan": int(optimal_unweighted[np.argmin(weighted_at_unweighted_optima)]),
        "optimal_plans_weighted": len(optimal_weighted),
        "optimal_plans_unweighted": len(optimal_unweighted),
        "shared_optimal_plan": bool(np.intersect1d(optimal_weighted, optimal_unweighted).size),
        "weighted_loss_best_tie": relative_loss(best_weighted, weighted_at_unweighted_optima.max()),
        "weighted_loss_worst_tie": relative_loss(best_weighted, weighted_at_unweighted_optima.min()),
        "unweighted_loss_best_tie": relative_loss(best_unweighted, unweighted_at_weighted_optima.max()),
        "unweighted_loss_worst_tie": relative_loss(best_unweighted, unweighted_at_weighted_optima.min()),
    }


def result_rows(instance, budget, weighted, analysis):
    best = analysis["best_weighted"]
    methods = (("plan_weights_on", analysis["weighted_plan"]),
               ("plan_weights_off_best_tie", analysis["unweighted_best_tie_plan"]),
               ("plan_weights_off_worst_tie", analysis["unweighted_worst_tie_plan"]))
    return [[EXPERIMENT_ID, instance["name"], method, budget, instance["seed"], float(weighted[plan]),
             relative_loss(best, weighted[plan]), SCRIPT_NAME] for method, plan in methods]


def run_instance(name, budgets, processes):
    instance = stored_instance(name)
    plans = all_plans(len(instance["elements"]))
    chosen_budgets = budgets or [instance["budget"]]
    subset = np.flatnonzero(np.isin(plans.sum(axis=1), chosen_budgets))
    weighted = filled_table(len(plans), subset, cached_values(instance, plans, subset, processes))
    unweighted = filled_table(len(plans), subset, cached_values(unweighted_copy(instance), plans, subset, processes))
    rows, summaries = [], []
    for budget in chosen_budgets:
        analysis = analyse_budget(weighted, unweighted, plans, budget)
        rows += result_rows(instance, budget, weighted, analysis)
        summaries.append([name, instance["family"], len(instance["elements"]), budget]
                         + [analysis[column] for column in SUMMARY_COLUMNS[4:]])
    return rows, summaries


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="*", default=instance_names())
    parser.add_argument("--budgets", nargs="*", type=int, default=None)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--tag", default="run")
    arguments = parser.parse_args()
    all_rows, all_summaries = [], []
    for name in arguments.names:
        rows, summaries = run_instance(name, arguments.budgets, arguments.processes)
        all_rows += rows
        all_summaries += summaries
        print(name, "done", flush=True)
    write_csv(OUTPUT_DIR / f"e10_result_rows_{arguments.tag}.csv", ROW_COLUMNS, all_rows)
    write_csv(OUTPUT_DIR / f"e10_summary_{arguments.tag}.csv", SUMMARY_COLUMNS, all_summaries)


if __name__ == "__main__":
    main()
