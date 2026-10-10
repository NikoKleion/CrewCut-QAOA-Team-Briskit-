import re
import sys
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

from briskit import benchmark
from briskit.feeder import all_plans, sample_instance, upstream_parents
from briskit.instances import restoration
from briskit.solvers import best_first, fitted
from checks.record import run_checks

ROOT = Path(__file__).resolve().parent.parent
PARENT = upstream_parents()


def second_route_best_value(instance, strength=1e-3):
    plans = all_plans(instance.size)
    plans = plans[plans.sum(axis=1) == instance.count].astype(float)
    columns = [np.ones(len(plans))] + [plans[:, i] for i in range(instance.size)]
    columns += [plans[:, i] * plans[:, j] for i, j in combinations(range(instance.size), 2)]
    design = np.stack(columns, axis=1)
    values = instance.score(plans)
    stacked = np.vstack([design, np.sqrt(strength) * np.eye(design.shape[1])])
    target = np.concatenate([values, np.zeros(design.shape[1])])
    coefficients = np.linalg.lstsq(stacked, target, rcond=None)[0]
    return float(values[np.argmax(design @ coefficients)])


def fit_on_every_plan_matches_second_route(factory, cases=10):
    same = 0
    for seed in range(cases):
        weights, _ = sample_instance(np.random.default_rng(seed), 12)
        instance = restoration(f"feeder {seed}", weights, PARENT, 6)
        row = benchmark.run("check", [instance], {"fit": factory(comb(12, 6))}, [1], [0], "checks", prefix_safe={"fit"})[0]
        same += bool(np.isclose(row["returned"], second_route_best_value(instance), rtol=1e-9, atol=0.0))
    return f"{cases} of {cases} best plans of equal value", f"{same} of {cases}", same == cases


def model_regret_agrees_with_saved_experiment(factory):
    text = (ROOT / "results" / "feeder_fitted_models.txt").read_text(encoding="utf-8")
    line = next(row for row in text.splitlines() if row.strip().startswith("restore 6:"))
    saved = float(re.search(r"single and pair terms (\d+\.\d)%", line).group(1)) / 100
    rng = np.random.default_rng(20261010)
    for _ in range(200):
        sample_instance(rng, 12)
        rng.choice(comb(12, 4), size=300, replace=False)
    regrets = []
    for k in range(200):
        weights, _ = sample_instance(rng, 12)
        rng.choice(comb(12, 6), size=300, replace=False)
        instance = restoration(f"restore 6 of 12, {k}", weights, PARENT, 6)
        row = benchmark.run("check", [instance], {"fit": factory(300)}, [1], [0], "checks", prefix_safe={"fit"})[0]
        regrets.append(1 - row["returned"] / instance.optimum())
    margin = 4 * np.sqrt(2) * np.std(regrets, ddof=1) / np.sqrt(len(regrets))
    return f"{saved:.3f} within {margin:.3f}", f"{np.mean(regrets):.3f}", abs(np.mean(regrets) - saved) <= margin


def pair_fit(samples):
    return fitted(samples, best_first)


def single_term_fit(samples):
    return fitted(samples, best_first, pairs=False)


CASES = [
    (fit_on_every_plan_matches_second_route, pair_fit, single_term_fit, "the fit with single terms only"),
    (model_regret_agrees_with_saved_experiment, pair_fit, single_term_fit, "the fit with single terms only"),
]


if __name__ == "__main__":
    sys.exit(0 if run_checks(CASES, Path(__file__).with_suffix(".csv")) else 1)
