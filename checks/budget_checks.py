import sys
from math import comb
from pathlib import Path

import numpy as np

from briskit import benchmark, summary
from checks.record import run_checks
from experiments.budget_experiment import FITTED, crew_instance, restoration_instance

ROOT = Path(__file__).resolve().parent.parent


def best_of_distinct_draws(regrets, draws):
    ordered = np.sort(regrets)
    total = comb(len(ordered), draws)
    chance = [(comb(len(ordered) - m, draws) - comb(len(ordered) - m - 1, draws)) / total for m in range(len(ordered))]
    return float(np.dot(ordered, chance))


def best_of_half_the_draws(regrets, draws):
    return best_of_distinct_draws(regrets, draws // 2)


def sample_phase_agrees(expectation, file, make_instance, methods, total=100):
    runs = summary.by_run(benchmark.read_rows(ROOT / "results" / file))
    measured = [summary.regret_at(run, total) for (instance, method, seed), run in runs.items() if method in methods]
    exact = []
    for k in sorted({seed for (instance, method, seed) in runs}):
        instance = make_instance(k)
        values = instance.table()[instance.feasible()]
        exact.append(expectation(np.abs(values - instance.optimum()) / instance.optimum(), total))
    margin = 4 * np.std(measured, ddof=1) / np.sqrt(len(measured))
    return (f"{np.mean(exact):.5f} within {margin:.5f}", f"{np.mean(measured):.5f} over {len(measured)} runs",
            abs(np.mean(measured) - np.mean(exact)) <= margin)


def restoration_random_plans_match_distinct_draws(expectation):
    return sample_phase_agrees(expectation, "budget_experiment_restoration.csv", restoration_instance, {"random plans"})


def restoration_fitting_samples_match_distinct_draws(expectation):
    return sample_phase_agrees(expectation, "budget_experiment_restoration.csv", restoration_instance, set(FITTED))


def crews_random_plans_match_distinct_draws(expectation):
    return sample_phase_agrees(expectation, "budget_experiment_crews.csv", crew_instance, {"random plans"})


def crews_fitting_samples_match_distinct_draws(expectation):
    return sample_phase_agrees(expectation, "budget_experiment_crews.csv", crew_instance, set(FITTED))


CASES = [
    (restoration_random_plans_match_distinct_draws, best_of_distinct_draws, best_of_half_the_draws, "half the evaluations counted"),
    (restoration_fitting_samples_match_distinct_draws, best_of_distinct_draws, best_of_half_the_draws, "half the evaluations counted"),
    (crews_random_plans_match_distinct_draws, best_of_distinct_draws, best_of_half_the_draws, "half the evaluations counted"),
    (crews_fitting_samples_match_distinct_draws, best_of_distinct_draws, best_of_half_the_draws, "half the evaluations counted"),
]


if __name__ == "__main__":
    sys.exit(0 if run_checks(CASES, Path(__file__).with_suffix(".csv")) else 1)
