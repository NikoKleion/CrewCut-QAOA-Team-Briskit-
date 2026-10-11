import re
import sys
from math import comb
from pathlib import Path

import numpy as np

from briskit import benchmark
from briskit.baselines import greedy_marginal
from briskit.feeder import plan_index, sample_instance, upstream_parents
from briskit.instances import Instance, crew_assignment, restoration
from briskit.qaoa import expected_best_of
from briskit.scheduling import list_schedule, single_crew_order
from briskit.solvers import (allowed, best_first, conversion, fitted, greedy, largest_weights, qaoa, random_plans,
                             thermal)
from checks.record import run_checks

ROOT = Path(__file__).resolve().parent.parent
PARENT = upstream_parents()


def feeder_instance(name, seed, count=6):
    weights, _ = sample_instance(np.random.default_rng(seed), 12)
    return restoration(name, weights, PARENT, count)


def random_best_matches_order_statistics(solver, budget=3, seeds=2000):
    instance = feeder_instance("a", 1)
    rows = benchmark.run("check", [instance], {"random plans": solver}, [budget], range(seeds), "checks")
    values = np.array([row["value"] for row in rows])
    allowed = instance.table()[instance.feasible()]
    exact = -expected_best_of(np.full(allowed.size, 1 / allowed.size), -allowed, budget)
    margin = 4 * values.std(ddof=1) / np.sqrt(seeds)
    return f"{exact:.4f} within {margin:.4f}", f"{values.mean():.4f}", abs(values.mean() - exact) <= margin


def greedy_matches_table_route(solver, cases=20):
    same = 0
    for seed in range(cases):
        instance = feeder_instance(f"feeder {seed}", seed)
        counted = benchmark.Counted(instance)
        plan = solver(counted, 1, None)[0]
        same += plan_index(plan) == greedy_marginal(instance.table(), instance.size, instance.count)
    return f"{cases} of {cases} plans equal", f"{same} of {cases} plans equal", same == cases


def greedy_evaluations_match_hand_count(solver):
    counted = benchmark.Counted(feeder_instance("a", 1))
    solver(counted, 1, None)
    expected = sum(12 - step for step in range(6))
    return expected, counted.used(), counted.used() == expected


def largest_weights_matches_argsort(solver, cases=20):
    same = 0
    for seed in range(cases):
        instance = feeder_instance(f"feeder {seed}", seed)
        plan = solver(instance, 1, None)[0]
        direct = np.zeros(12, dtype=np.int8)
        direct[np.argsort(-instance.data["weights"])[:6]] = 1
        same += np.array_equal(plan, direct)
    return f"{cases} of {cases} plans equal", f"{same} of {cases} plans equal", same == cases


def fit_evaluations_are_counted(factory, samples=50, budget=3):
    instance = feeder_instance("a", 1, count=4)
    rows = benchmark.run("check", [instance], {"fitted": factory(samples)}, [budget], range(10), "checks")
    used = [row["evaluations"] for row in rows]
    return f"{samples} to {samples + budget} in every run", f"{min(used)} to {max(used)}", min(used) >= samples and max(used) <= samples + budget


def budget_zero_gives_the_sample(factory, samples=50):
    base = feeder_instance("a", 1, count=4)
    seen = {}

    def recording(plans):
        values = base.score(plans)
        for plan, value in zip(np.atleast_2d(plans), values):
            seen[plan_index(plan)] = float(value)
        return values

    class Recorded(Instance):
        def table(self):
            return base.table()

    instance = Recorded("a", base.size, base.sense, recording, base.count, base.data)
    row = benchmark.run("check", [instance], {"fitted": factory(samples)}, [0], [0], "checks", prefix_safe={"fitted"})[0]
    expected = (max(seen.values()) if seen else None, len(seen))
    got = (row["value"], row["evaluations"])
    return f"{expected} with {samples} plans scored", got, len(seen) == samples and got == expected


def feeder_figures_are_reproduced(solvers):
    text = (ROOT / "results" / "feeder_fitted_models.txt").read_text(encoding="utf-8")
    saved, measured = [], []
    rng = np.random.default_rng(20261010)
    for count in (4, 6):
        line = next(row for row in text.split("\n") if row.strip().startswith(f"restore {count}:"))
        saved += [float(re.search(rf"{label} (\d+\.\d)%", line).group(1)) for label in ("largest weights first", "greedy by marginal gain")]
        instances = []
        for k in range(200):
            weights, _ = sample_instance(rng, 12)
            rng.choice(comb(12, count), size=min(300, comb(12, count)), replace=False)
            instances.append(restoration(f"restore {count} of 12, {k}", weights, PARENT, count))
        rows = benchmark.run("check", instances, {"largest": solvers[0], "greedy": solvers[1]}, [1], [0], "checks")
        measured += [float(round(100 * np.mean([row["regret"] for row in rows if row["method"] == method]), 1)) for method in ("largest", "greedy")]
    return saved, measured, saved == measured


def conversion_figures_are_reproduced(solver):
    text = (ROOT / "results" / "dispatch_rules.txt").read_text(encoding="utf-8")
    line = next(row for row in text.splitlines() if row.strip().startswith("conversion algorithm"))
    saved = [float(number) for number in re.findall(r"\+\s*(\d+\.\d+)%", line)]
    rng = np.random.default_rng(20261010)
    instances = []
    for k in range(300):
        weights, hours = sample_instance(rng, 12)
        instances.append(crew_assignment(f"crews {k}", weights, hours, PARENT))
    rows = benchmark.run("check", instances, {"conversion": solver}, [1], [0], "checks")
    regrets = np.array([row["regret"] for row in rows])
    measured = [float(round(100 * number, 2)) for number in (regrets.mean(), np.median(regrets), regrets.max())]
    return saved, measured, saved == measured


def plan_follows_one_weight(solver):
    weights = np.array([1.0, 0.1, 0.2, 0.3, 0.2, 0.9, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    raised = weights.copy()
    raised[1] = 5.0
    chosen = []
    for name, w in (("before", weights), ("after", raised)):
        plan = solver(benchmark.Counted(restoration(name, w, PARENT, 2)), 1, np.random.default_rng(0))[0]
        chosen.append(sorted(int(i) for i in np.flatnonzero(plan)))
    return [[0, 5], [0, 1]], chosen, chosen == [[0, 5], [0, 1]]


def greedy_plan_follows_one_weight(solver):
    return plan_follows_one_weight(solver)


def fitted_plan_follows_one_weight(solver):
    return plan_follows_one_weight(solver)


def thermal_mean_matches_direct_formula(select, beta=2.0, draws=20000):
    instance = feeder_instance("a", 1)
    rows, plans = allowed(instance)
    merit = instance.table()
    chosen = select(beta)(merit, rows, instance, draws, np.random.default_rng(0))
    weight = np.exp(beta * (merit[rows] - merit[rows].max()) / merit[rows].std())
    exact = float(weight @ merit[rows] / weight.sum())
    margin = 4 * merit[chosen].std(ddof=1) / np.sqrt(draws)
    return f"{exact:.4f} within {margin:.4f}", f"{merit[chosen].mean():.4f}", abs(merit[chosen].mean() - exact) <= margin


def qaoa_draws_beat_random_plans(select, draws=4000):
    instance = feeder_instance("a", 1, count=4)
    rows, plans = allowed(instance)
    merit = instance.table()
    chosen = select(1)(merit, rows, instance, draws, np.random.default_rng(0))
    right_count = bool((plans[chosen].sum(axis=1) == instance.count).all())
    margin = 4 * merit[chosen].std(ddof=1) / np.sqrt(draws)
    return (f"above {merit[rows].mean():.4f} by more than {margin:.4f}, every draw with {instance.count} lines",
            f"{merit[chosen].mean():.4f}, right count {right_count}",
            right_count and merit[chosen].mean() > merit[rows].mean() + margin)


def one_draw_short(instance, budget, rng):
    return random_plans(instance, budget - 1, rng)


def greedy_by_worst_gain(instance, budget, rng):
    plan = np.zeros(instance.size, dtype=np.int8)
    for _ in range(instance.count):
        free = np.flatnonzero(plan == 0)
        trials = np.repeat(plan[None, :], len(free), axis=0)
        trials[np.arange(len(free)), free] = 1
        plan[free[np.argmin(instance.score(trials))]] = 1
    return plan[None, :]


def greedy_that_scores_twice(instance, budget, rng):
    greedy(instance, budget, rng)
    extra = np.ones((1, instance.size), dtype=np.int8)
    instance.score(extra)
    return greedy(instance, budget, rng)


def smallest_weights(instance, budget, rng):
    plan = np.zeros(instance.size, dtype=np.int8)
    plan[np.argsort(instance.data["weights"])[: instance.count]] = 1
    return plan[None, :]


def fitted_past_the_counter(samples):
    inner = fitted(samples, best_first)

    class Open:
        def __init__(self, counted):
            self.__dict__.update(counted.__dict__)
            self.score = counted._instance.score

    return lambda instance, budget, rng: inner(Open(instance), budget, rng)


def thermal_with_the_sign_flipped(beta):
    return thermal(-beta)


def qaoa_on_the_negated_cost(depth):
    inner = qaoa(depth)
    return lambda merit, rows, instance, budget, rng: inner(-merit, rows, instance, budget, rng)


def conversion_in_reverse_order(instance, budget, rng):
    weights, hours, parent = instance.data["weights"], instance.data["hours"], instance.data["parent"]
    return list_schedule(single_crew_order(weights, hours, parent)[::-1], hours)[None, :]


def fixed_plan(instance, budget, rng):
    plan = np.zeros(instance.size, dtype=np.int8)
    plan[[0, 5]] = 1
    return plan[None, :]


CASES = [
    (random_best_matches_order_statistics, random_plans, one_draw_short, "a sampler that returns one draw fewer"),
    (greedy_matches_table_route, greedy, greedy_by_worst_gain, "greedy that takes the worst gain"),
    (greedy_evaluations_match_hand_count, greedy, greedy_that_scores_twice, "greedy that scores one plan more"),
    (largest_weights_matches_argsort, largest_weights, smallest_weights, "the smallest weights"),
    (fit_evaluations_are_counted, lambda samples: fitted(samples, best_first), fitted_past_the_counter, "a fit that scores past the counter"),
    (budget_zero_gives_the_sample, lambda samples: fitted(samples, best_first), fitted_past_the_counter, "a fit that scores past the counter"),
    (feeder_figures_are_reproduced, (largest_weights, greedy), (smallest_weights, greedy_by_worst_gain), "smallest weights and worst gain"),
    (conversion_figures_are_reproduced, conversion, conversion_in_reverse_order, "the single crew order reversed"),
    (greedy_plan_follows_one_weight, greedy, fixed_plan, "a solver that ignores the scores"),
    (fitted_plan_follows_one_weight, fitted(66, best_first), fixed_plan, "a solver that ignores the scores"),
    (thermal_mean_matches_direct_formula, thermal, thermal_with_the_sign_flipped, "the temperature sign flipped"),
    (qaoa_draws_beat_random_plans, qaoa, qaoa_on_the_negated_cost, "QAOA trained on the negated cost"),
]


if __name__ == "__main__":
    sys.exit(0 if run_checks(CASES, Path(__file__).with_suffix(".csv")) else 1)
