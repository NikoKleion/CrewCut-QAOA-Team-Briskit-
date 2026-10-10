import sys
from pathlib import Path

import numpy as np

from briskit import benchmark, summary
from briskit.feeder import sample_instance, upstream_parents
from briskit.instances import restoration
from briskit.solvers import random_plans
from checks.record import run_checks


def row(instance, evaluations, regret, budget=0, method="m"):
    return {"instance": instance, "method": method, "seed": 0, "budget": budget, "evaluations": evaluations,
            "regret": regret}


def regret_at_reads_the_step_curve(regret_at):
    run = [row("a", 1, 0.5), row("a", 3, 0.2), row("a", 4, 0.2), row("a", 5, 0.0)]
    got = [regret_at(run, total) for total in (0, 2, 3, 10)]
    return [None, 0.5, 0.2, 0.0], got, got == [None, 0.5, 0.2, 0.0]


def average_runtime_matches_hand_arithmetic(runtime_to_target):
    rows = [row("a", 2, 0.5), row("a", 4, 0.0), row("b", 6, 0.3), row("b", 10, 0.01), row("c", 5, 0.4), row("c", 20, 0.2)]
    result = runtime_to_target(summary.by_run(rows), "m", 0.01)
    got = (result["reached"], result["runs"], result["average"])
    return (2, 3, 17.0), got, got == (2, 3, 17.0)


def quartiles_match_hand_values(regret_summary):
    rows = [row(name, 1, value) for name, value in zip("abcde", (0.0, 0.1, 0.2, 0.3, 1.0))]
    result = regret_summary(summary.by_run(rows), "m", 1)
    got = tuple(round(result[key], 12) for key in ("lower", "median", "upper", "mean"))
    return (0.1, 0.2, 0.3, 0.32), got, got == (0.1, 0.2, 0.3, 0.32)


def compact_rows_give_the_same_summary(changes_only):
    weights, _ = sample_instance(np.random.default_rng(1), 12)
    instance = restoration("a", weights, upstream_parents(), 6)
    rows = benchmark.run("check", [instance], {"m": random_plans}, range(1, 61), range(8), "checks", prefix_safe={"m"})
    compact = changes_only(rows)

    def figures(some):
        runs = summary.by_run(some)
        target = summary.runtime_to_target(runs, "m", 0.2)
        return ([summary.regret_summary(runs, "m", total)["mean"] for total in (5, 20, 60)], target["reached"], target["average"])

    return (f"{figures(rows)} from fewer than {len(rows)} rows", f"{figures(compact)} from {len(compact)} rows",
            figures(rows) == figures(compact) and len(compact) < len(rows))


def first_row_past_the_total(run, total):
    later = [r for r in run if r["evaluations"] >= total]
    return later[0]["regret"] if later else None


def mean_of_successful_runtimes(runs, method, target, tolerance=1e-12):
    result = summary.runtime_to_target(runs, method, target, tolerance)
    hits = [next(r["evaluations"] for r in run if r["regret"] <= target + tolerance)
            for run in runs.values() if any(r["regret"] <= target + tolerance for r in run)]
    return {**result, "average": float(np.mean(hits))}


def changes_without_the_last_row(rows):
    kept = benchmark.changes_only(rows)
    last = {}
    for row in kept:
        last[(row["instance"], row["method"], row["seed"])] = row
    return [row for row in kept if row is not last[(row["instance"], row["method"], row["seed"])] or row["regret"] == 0.0]


def summary_with_the_mean_as_median(runs, method, total):
    result = summary.regret_summary(runs, method, total)
    return {**result, "median": result["mean"]}


CASES = [
    (regret_at_reads_the_step_curve, summary.regret_at, first_row_past_the_total, "the first row at or past the total"),
    (average_runtime_matches_hand_arithmetic, summary.runtime_to_target, mean_of_successful_runtimes, "unsuccessful runs left out"),
    (quartiles_match_hand_values, summary.regret_summary, summary_with_the_mean_as_median, "the mean reported as the median"),
    (compact_rows_give_the_same_summary, benchmark.changes_only, changes_without_the_last_row, "the last row of an unsuccessful run dropped"),
]


if __name__ == "__main__":
    sys.exit(0 if run_checks(CASES, Path(__file__).with_suffix(".csv")) else 1)
