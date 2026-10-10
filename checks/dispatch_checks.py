import sys
from pathlib import Path

import numpy as np

from briskit.dispatch import dispatch_times, energised_times, lines_below_each, rho_factors
from briskit.feeder import all_plans, plan_index, random_tree_parents, sample_instance, upstream_parents
from briskit.scheduling import energization_cost, list_schedule, priority_dispatch, single_crew_order
from checks.record import run_checks

SEED = 20261010
INSTANCES = 300


def feeder_cases():
    rng = np.random.default_rng(SEED)
    parent = upstream_parents()
    return parent, [sample_instance(rng, len(parent)) for _ in range(INSTANCES)]


def every_line_below(line, below):
    found = []
    for child in below[line]:
        found += [child] + every_line_below(child, below)
    return found


def rho_by_masks(weights, hours, parent):
    below = lines_below_each(parent)
    rho = np.zeros(len(parent))
    for line in range(len(parent)):
        under = every_line_below(line, below)
        best = weights[line] / hours[line]
        for mask in range(1, 2 ** len(under)):
            chosen = {under[k] for k in range(len(under)) if (mask >> k) & 1}
            if all(parent[j] == line or parent[j] in chosen for j in chosen):
                members = [line] + sorted(chosen)
                best = max(best, weights[members].sum() / hours[members].sum())
        rho[line] = best
    return rho


def ratio_of_the_line_alone(weights, hours, parent):
    return weights / hours


def every_unstarted_line_is_a_candidate(priority, hours, parent, crews=2):
    n = len(parent)
    finish = {}
    free_at = [0.0] * crews
    t = 0.0
    while len(finish) < n:
        for crew in range(crews):
            if free_at[crew] > t + 1e-12:
                continue
            line = max((j for j in range(n) if j not in finish), key=lambda j: priority[j], default=None)
            if line is None:
                continue
            finish[line] = t + hours[line]
            free_at[crew] = finish[line]
        t = min((f for f in finish.values() if f > t + 1e-12), default=t)
    completed = np.array([finish[j] for j in range(n)])
    return completed, energised_times(completed, parent)


def priority_ignored(priority, hours, parent, crews=2):
    return dispatch_times(np.zeros(len(parent)), hours, parent, crews)


def cost_with_energised_candidates(priority, weights, hours, parent):
    return float(weights @ dispatch_times(priority, hours, parent, 2, "energised")[1])


def rho_matches_subtree_masks(rho):
    rng = np.random.default_rng(SEED)
    worst = 0.0
    for parent in [upstream_parents()] * 20 + [random_tree_parents(rng, 9) for _ in range(40)]:
        weights, hours = sample_instance(rng, len(parent))
        worst = max(worst, float(np.abs(rho(weights, hours, parent) - rho_by_masks(weights, hours, parent)).max()))
    return "difference below 1e-12", f"{worst:.1e}", worst < 1e-12


def second_implementation_matches(times):
    parent, cases = feeder_cases()
    worst = 0.0
    for weights, hours in cases:
        for priority in (weights, weights / hours):
            worst = max(worst, abs(float(weights @ times(priority, hours, parent)[1])
                                   - priority_dispatch(priority, weights, hours, parent)))
    return "difference below 1e-9", f"{worst:.1e}", worst < 1e-9


def conversion_cost(weights, hours, parent, plans):
    order = single_crew_order(weights, hours, parent)
    return energization_cost(plans, order, weights, hours, parent)[plan_index(list_schedule(order, hours))]


def rho_dispatch_equals_conversion(priorities):
    parent, cases = feeder_cases()
    plans = all_plans(len(parent))
    equal = sum(abs(priority_dispatch(priorities(weights, hours, parent), weights, hours, parent)
                    - conversion_cost(weights, hours, parent, plans)) < 1e-9 for weights, hours in cases)
    return f"{INSTANCES} of {INSTANCES}", f"{equal} of {INSTANCES}", equal == INSTANCES


def dispatch_obeys_theorem_5(dispatch):
    parent, cases = feeder_cases()
    plans = all_plans(len(parent))
    equal = sum(abs(dispatch(rho_factors(weights, hours, parent), weights, hours, parent)
                    - conversion_cost(weights, hours, parent, plans)) < 1e-9 for weights, hours in cases)
    return f"{INSTANCES} of {INSTANCES}", f"{equal} of {INSTANCES}", equal == INSTANCES


def raising_one_weight_changes_the_order(times):
    parent = [-1, 0, 0, 1]
    hours = np.ones(4)
    before = times(np.array([1.0, 2.0, 1.0, 0.5]), hours, parent, 1)[0][2]
    after = times(np.array([1.0, 2.0, 3.0, 0.5]), hours, parent, 1)[0][2]
    return "line 2 finishes at 3 then 2", f"line 2 finishes at {before:.0f} then {after:.0f}", (before, after) == (3.0, 2.0)


def area_above_curve(weights, moments):
    order = np.argsort(moments)
    restored = np.concatenate([[0.0], np.cumsum(weights[order])]) / weights.sum()
    edges = np.concatenate([[0.0], moments[order]])
    return float(np.sum((1.0 - restored[:-1]) * np.diff(edges)) * weights.sum())


def curve_area_equals_customer_hours(moments):
    parent, cases = feeder_cases()
    worst = 0.0
    for weights, hours in cases:
        for priority in (weights, weights / hours, rho_factors(weights, hours, parent)):
            completed, live = dispatch_times(priority, hours, parent)
            worst = max(worst, abs(area_above_curve(weights, moments(completed, live)) - float(weights @ live)))
    return "difference below 1e-9", f"{worst:.1e}", worst < 1e-9


CASES = [
    (rho_matches_subtree_masks, rho_factors, ratio_of_the_line_alone, "the ratio of the line alone"),
    (second_implementation_matches, dispatch_times, every_unstarted_line_is_a_candidate,
     "every unstarted line is a candidate"),
    (rho_dispatch_equals_conversion, rho_factors, ratio_of_the_line_alone, "priority w / p in place of rho"),
    (dispatch_obeys_theorem_5, priority_dispatch, cost_with_energised_candidates,
     "candidates need an energised node or a line under repair upstream"),
    (raising_one_weight_changes_the_order, dispatch_times, priority_ignored, "priority ignored"),
    (curve_area_equals_customer_hours, lambda completed, live: live, lambda completed, live: completed,
     "completion times in place of energisation times"),
]


if __name__ == "__main__":
    sys.exit(0 if run_checks(CASES, Path(__file__).with_suffix(".csv")) else 1)
