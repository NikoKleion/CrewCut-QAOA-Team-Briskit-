import sys
from pathlib import Path

import numpy as np

from briskit import benchmark
from briskit.feeder import sample_instance, upstream_parents
from briskit.instances import Instance, restoration
from briskit.solvers import random_plans
from checks.record import run_checks


def tiny_instance(count=None):
    return Instance("tiny", 4, "max", lambda plans: plans @ np.array([1.0, 2.0, 4.0, 8.0]), count)


def feeder_instance(name, seed, count=6):
    weights, _ = sample_instance(np.random.default_rng(seed), 12)
    return restoration(name, weights, upstream_parents(), count)


def outcome(accept, plans, instance, budget):
    try:
        accept(np.array(plans), instance, budget)
    except ValueError:
        return "refused"
    return "accepted"


def distinct_plans_are_counted(counter):
    counted = counter(tiny_instance())
    counted.score([[1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]])
    counted.score([[1, 1, 0, 0], [0, 0, 1, 0]])
    return 3, counted.used(), counted.used() == 3


def budget_is_enforced(accept):
    got = outcome(accept, [[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0]], tiny_instance(count=2), 2)
    return "refused", got, got == "refused"


def wrong_count_is_refused(accept):
    got = outcome(accept, [[1, 1, 1, 0]], tiny_instance(count=2), 5)
    return "refused", got, got == "refused"


def entry_other_than_0_or_1_is_refused(accept):
    got = outcome(accept, [[2, 0, 0, 0]], tiny_instance(), 5)
    return "refused", got, got == "refused"


def regret_matches_hand_arithmetic(regret):
    got = (regret(6.0, 8.0, "max"), regret(5.0, 4.0, "min"))
    return (0.25, 0.25), got, got == (0.25, 0.25)


def rows_regenerate_alone(run):
    instances = [feeder_instance("a", 1), feeder_instance("b", 2)]
    solvers = {"first sampler": random_plans, "second sampler": random_plans}
    together = run("check", instances, solvers, [1, 3], [0, 1, 2], "checks")
    alone = run("check", instances[1:], {"second sampler": random_plans}, [1, 3], [0, 1, 2], "checks")
    key = lambda row: (row["budget"], row["seed"], row["value"], row["evaluations"])
    wanted = sorted(key(row) for row in together if row["instance"] == "b" and row["method"] == "second sampler")
    same = sum(a == b for a, b in zip(wanted, sorted(key(row) for row in alone)))
    return f"{len(wanted)} of {len(wanted)} rows equal", f"{same} of {len(wanted)} rows equal", same == len(wanted)


def prefix_route_equals_budget_route(solver):
    instance = feeder_instance("a", 1)
    budgets, seeds = [1, 2, 5, 10], range(5)
    key = lambda row: (row["budget"], row["seed"], row["value"], row["evaluations"])
    prefix = sorted(key(row) for row in benchmark.run("check", [instance], {"m": solver}, budgets, seeds, "checks", prefix_safe={"m"}))
    each = sorted(key(row) for row in benchmark.run("check", [instance], {"m": solver}, budgets, seeds, "checks"))
    same = sum(a[:2] == b[:2] and a[3] == b[3] and bool(np.isclose(a[2], b[2], rtol=1e-12, atol=0.0)) for a, b in zip(prefix, each))
    return f"{len(each)} of {len(each)} rows equal", f"{same} of {len(each)} rows equal", same == len(each)


class CountsRows(benchmark.Counted):
    def __init__(self, instance):
        super().__init__(instance)
        self.rows = 0

    def score(self, plans):
        self.rows += len(np.atleast_2d(plans))
        return super().score(plans)

    def used(self):
        return self.rows


def accepts_anything(plans, instance, budget):
    return np.atleast_2d(plans)


def regret_with_one_sign(value, optimum, sense):
    return (optimum - value) / optimum


def run_on_one_shared_stream(*arguments, **options):
    shared = np.random.default_rng(0)
    original = benchmark.stream
    benchmark.stream = lambda seed, instance, method: shared
    try:
        return benchmark.run(*arguments, **options)
    finally:
        benchmark.stream = original


def reversed_sampler(instance, budget, rng):
    return random_plans(instance, budget, rng)[::-1]


CASES = [
    (distinct_plans_are_counted, benchmark.Counted, CountsRows, "a counter that counts rows scored"),
    (budget_is_enforced, benchmark.accepted, accepts_anything, "no test on returned plans"),
    (wrong_count_is_refused, benchmark.accepted, accepts_anything, "no test on returned plans"),
    (entry_other_than_0_or_1_is_refused, benchmark.accepted, accepts_anything, "no test on returned plans"),
    (regret_matches_hand_arithmetic, benchmark.regret, regret_with_one_sign, "the same sign for both senses"),
    (rows_regenerate_alone, benchmark.run, run_on_one_shared_stream, "one random stream shared by all runs"),
    (prefix_route_equals_budget_route, random_plans, reversed_sampler, "a sampler whose order depends on the budget"),
]


if __name__ == "__main__":
    sys.exit(0 if run_checks(CASES, Path(__file__).with_suffix(".csv")) else 1)
