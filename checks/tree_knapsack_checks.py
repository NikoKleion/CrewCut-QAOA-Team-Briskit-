import itertools
import sys
from pathlib import Path

import numpy as np

from briskit.feeder import random_tree_parents, sample_instance, upstream_parents
from briskit.instances import restoration
from briskit.tree_knapsack import best_closed_set, depth_first_order, tree_optimum
from checks.record import run_checks

SEED = 20261010


def table_without_skipping(weights, parent, capacity, costs=None):
    n = len(parent)
    costs = np.ones(n, dtype=int) if costs is None else np.asarray(costs, dtype=int)
    table = np.zeros((n + 1, capacity + 1))
    for i in range(n - 1, -1, -1):
        cost = int(costs[i])
        table[i] = table[i + 1]
        if cost <= capacity:
            table[i, cost:] = np.maximum(table[i, cost:], weights[i] + table[i + 1, : capacity + 1 - cost])
    return float(table[0, capacity]), np.zeros(n, dtype=np.int8)


def table_with_unit_costs(weights, parent, capacity, costs=None):
    return best_closed_set(weights, parent, capacity)


def table_that_ignores_weights(weights, parent, capacity, costs=None):
    return best_closed_set(np.ones(len(parent)), parent, capacity, costs)


def zero_weight_lines_left_out(instance, budget, rng):
    return tree_optimum(instance, budget, rng) * (instance.data["weights"] > 0)


def equals_enumeration_on_the_feeder(solve):
    equal, cases = 0, 0
    for count in (4, 6):
        for k in range(30):
            weights, _ = sample_instance(np.random.default_rng([SEED, k]), 12)
            instance = restoration("feeder", weights, upstream_parents(), count)
            cases += 1
            equal += abs(solve(weights, upstream_parents(), count)[0] - instance.optimum()) < 1e-9
    return f"{cases} of {cases}", f"{equal} of {cases}", equal == cases


def equals_enumeration_on_random_trees(solve):
    rng = np.random.default_rng(SEED)
    equal = 0
    for _ in range(10):
        parent = random_tree_parents(rng, 16)
        weights, _ = sample_instance(rng, 16)
        instance = restoration("tree", weights, parent, 7)
        equal += abs(solve(weights, parent, 7)[0] - instance.optimum()) < 1e-9
    return "10 of 10", f"{equal} of 10", equal == 10


def plan_has_the_count_and_the_value(solver):
    good = 0
    for k in range(30):
        rng = np.random.default_rng([SEED, k])
        weights, _ = sample_instance(rng, 12)
        weights[rng.choice(12, size=8, replace=False)] = 0.0
        instance = restoration("feeder", weights, upstream_parents(), 6)
        plans = solver(instance, 1, None)
        good += plans.sum() == 6 and abs(float(instance.score(plans)[0]) - instance.optimum()) < 1e-9
    return "30 of 30", f"{good} of 30", good == 30


def brute_force_closed_sets(weights, parent, capacity, costs):
    n = len(parent)
    best = 0.0
    for chosen in itertools.product([0, 1], repeat=n):
        closed = all(not chosen[j] or parent[j] == -1 or chosen[parent[j]] for j in range(n))
        if closed and np.dot(chosen, costs) <= capacity:
            best = max(best, float(np.dot(chosen, weights)))
    return best


def hours_budget_equals_brute_force(solve):
    rng = np.random.default_rng(SEED)
    equal = 0
    for _ in range(10):
        parent = random_tree_parents(rng, 10)
        weights, hours = sample_instance(rng, 10)
        costs = hours.astype(int)
        equal += abs(solve(weights, parent, 20, costs)[0] - brute_force_closed_sets(weights, parent, 20, costs)) < 1e-9
    return "10 of 10", f"{equal} of 10", equal == 10


def one_weight_changes_the_choice(solve):
    parent = [-1, 0, 0, 1]
    before = solve(np.array([0.1, 0.1, 0.5, 0.2]), parent, 3)[1]
    after = solve(np.array([0.1, 0.1, 0.5, 5.0]), parent, 3)[1]
    got = f"line 3 chosen {int(before[3])} then {int(after[3])}"
    return "line 3 chosen 0 then 1", got, (int(before[3]), int(after[3])) == (0, 1)


def order_keeps_subtrees_together(order_function):
    rng = np.random.default_rng(SEED)
    good = 0
    for _ in range(20):
        parent = random_tree_parents(rng, 30)
        order, size = order_function(parent)
        position = {line: i for i, line in enumerate(order)}
        inside = all(parent[line] == -1 or position[parent[line]] < position[line]
                     <= position[parent[line]] + size[position[parent[line]]] - 1 for line in range(30))
        good += inside and sorted(order) == list(range(30))
    return "20 of 20", f"{good} of 20", good == 20


def order_by_index(parent):
    n = len(parent)
    return list(range(n)), [1] * n


CASES = [
    (order_keeps_subtrees_together, depth_first_order, order_by_index, "lines in index order with subtrees of size 1"),
    (equals_enumeration_on_the_feeder, best_closed_set, table_without_skipping,
     "a table that does not skip the subtree of a line left out"),
    (equals_enumeration_on_random_trees, best_closed_set, table_without_skipping,
     "a table that does not skip the subtree of a line left out"),
    (plan_has_the_count_and_the_value, tree_optimum, zero_weight_lines_left_out, "lines of zero weight left out of the plan"),
    (hours_budget_equals_brute_force, best_closed_set, table_with_unit_costs, "costs ignored"),
    (one_weight_changes_the_choice, best_closed_set, table_that_ignores_weights, "weights ignored"),
]


if __name__ == "__main__":
    sys.exit(0 if run_checks(CASES, Path(__file__).with_suffix(".csv")) else 1)
