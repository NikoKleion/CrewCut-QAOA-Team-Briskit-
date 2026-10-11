import sys
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

from briskit.cost_circuit import cost_circuit, cost_scale
from briskit.feeder import sample_instance
from briskit.instances import crew_assignment
from briskit.one_hot import (assignments, cost_table, one_hot_circuit, one_hot_fields, one_hot_terms, penalty_bound,
                             penalty_bounds)
from briskit.qaoa import qaoa_probabilities
from briskit.scheduling import smith_order, weighted_completion_cost
from checks.record import run_checks

SEED = 20261010


def terms_with_weight_and_hours_swapped(weights, hours, order, crews, penalty):
    constant, single, pair = one_hot_terms(weights, hours, order, crews, penalty)
    n = len(weights)
    for a in range(n):
        for b in range(a + 1, n):
            k, j = order[a], order[b]
            for m in range(crews):
                first, second = sorted((k * crews + m, j * crews + m))
                pair[first, second] = weights[k] * hours[j]
    return constant, single, pair


def terms_with_penalty_on_the_count_itself(weights, hours, order, crews, penalty):
    _, single, pair = one_hot_terms(weights, hours, order, crews, penalty)
    return 0.0, single + 2 * np.repeat(np.broadcast_to(penalty, (len(weights),)), crews), pair


def terms_without_penalty_pairs(weights, hours, order, crews, penalty):
    constant, single, pair = one_hot_terms(weights, hours, order, crews, penalty)
    for j in range(len(weights)):
        pair[j * crews:(j + 1) * crews, j * crews:(j + 1) * crews] = 0.0
    return constant, single, pair


def mean_pair_coefficient(weights, hours, order):
    w, p = np.asarray(weights)[order], np.asarray(hours)[order]
    n = len(w)
    return float(np.mean([w[b] * p[a] for a in range(n) for b in range(a + 1, n)]))


def bounds_in_schedule_order(weights, hours, order):
    return penalty_bounds(weights, hours, order)[order]


def assignments_grouped_by_crew(n, crews):
    index = np.arange(2 ** (n * crews))
    crew = np.full((index.size, n), -1, dtype=np.int8)
    for j in range(n):
        on = np.stack([(index >> (m * n + j)) & 1 for m in range(crews)], axis=1)
        one = on.sum(axis=1) == 1
        crew[one, j] = on[one].argmax(axis=1)
    return (crew >= 0).all(axis=1), crew


def circuit_without_fields(single, pair, angles, measure=True):
    fields, couplings = one_hot_fields(single, pair)
    return cost_circuit(np.zeros_like(fields), couplings, angles, measure)


def schedule_cost(crew, order, weights, hours, crews):
    return sum(weighted_completion_cost([j for j in order if crew[j] == m], weights, hours) for m in range(crews))


def valid_states_have_the_schedule_cost(build):
    rng = np.random.default_rng(SEED)
    worst, states = 0.0, 0
    for n, crews in ((5, 2), (4, 3)):
        valid, crew = assignments(n, crews)
        for _ in range(5):
            weights, hours = sample_instance(rng, n)
            order = smith_order(weights, hours)
            table = cost_table(*build(weights, hours, order, crews, 7.0))
            for state in np.nonzero(valid)[0]:
                worst = max(worst, abs(table[state] - schedule_cost(crew[state], order, weights, hours, crews)))
                states += 1
    return "difference below 1e-9", f"largest difference {worst:.1e} over {states} states", worst < 1e-9


def penalty_moves_only_invalid_states(build):
    n, crews, penalty = 4, 3, np.array([3.0, 1.0, 2.0, 5.0])
    weights, hours = sample_instance(np.random.default_rng(SEED), n)
    order = smith_order(weights, hours)
    low = cost_table(*build(weights, hours, order, crews, penalty))
    high = cost_table(*build(weights, hours, order, crews, 2 * penalty))
    index = np.arange(low.size)
    on = np.stack([sum((index >> (j * crews + m)) & 1 for m in range(crews)) for j in range(n)], axis=1)
    deviation = float(np.abs(high - low - ((1 - on) ** 2) @ penalty).max())
    moved = float(np.abs((high - low)[(on == 1).all(axis=1)]).max())
    return ("valid states unmoved, the others up by the penalty times the squared miscount",
            f"largest deviation {deviation:.1e}, largest move of a valid state {moved:.1e}",
            deviation < 1e-9 and moved < 1e-9)


def minimum_is_a_plan_at_the_bound(bound):
    valid, _ = assignments(6, 2)
    good = 0
    for k in range(30):
        weights, hours = sample_instance(np.random.default_rng([SEED, k]), 6)
        order = smith_order(weights, hours)
        table = cost_table(*one_hot_terms(weights, hours, order, 2, 1.001 * bound(weights, hours, order)))
        lowest = int(np.argmin(table))
        good += bool(valid[lowest]) and abs(table[lowest] - crew_assignment("check", weights, hours).optimum()) < 1e-9
    return "30 of 30", f"{good} of 30", good == 30


def minimum_is_a_plan_at_each_repairs_own_bound(bounds):
    valid, _ = assignments(6, 2)
    good = 0
    for k in range(30):
        weights, hours = sample_instance(np.random.default_rng([SEED, k]), 6)
        order = smith_order(weights, hours)
        table = cost_table(*one_hot_terms(weights, hours, order, 2, 1.001 * bounds(weights, hours, order)))
        lowest = int(np.argmin(table))
        good += bool(valid[lowest]) and abs(table[lowest] - crew_assignment("check", weights, hours).optimum()) < 1e-9
    return "30 of 30", f"{good} of 30", good == 30


def two_crews_agree_with_one_qubit_per_repair(decode):
    rng = np.random.default_rng(SEED)
    valid, crew = decode(6, 2)
    states = np.nonzero(valid)[0]
    plan = crew[states].astype(np.int64) @ (2 ** np.arange(6))
    equal, plans = 0, 0
    for _ in range(5):
        weights, hours = sample_instance(rng, 6)
        table = cost_table(*one_hot_terms(weights, hours, smith_order(weights, hours), 2, 4.0))
        equal += int(np.sum(np.abs(table[states] - crew_assignment("check", weights, hours).table()[plan]) < 1e-9))
        plans += len(states)
    return f"{plans} of {plans}", f"{equal} of {plans}", equal == plans and plans == 5 * 64


def circuit_gives_the_table_distribution(builder):
    rng = np.random.default_rng(SEED)
    weights, hours = sample_instance(rng, 4)
    constant, single, pair = one_hot_terms(weights, hours, smith_order(weights, hours), 2, 5.0)
    angles = rng.uniform(-0.8, 0.8, 4)
    table = cost_table(constant, single, pair) / cost_scale(*one_hot_fields(single, pair))
    got = Statevector(builder(single, pair, angles, measure=False)).probabilities()
    worst = float(np.abs(got - qaoa_probabilities(table, angles)).max())
    return "difference below 1e-12", f"largest difference {worst:.1e}", worst < 1e-12


def counts_follow_the_formula(build):
    good, cases = 0, 0
    for n, crews in ((6, 2), (8, 2), (12, 2), (5, 3), (4, 4)):
        weights, hours = sample_instance(np.random.default_rng([SEED, n, crews]), n)
        _, single, pair = build(weights, hours, smith_order(weights, hours), crews, 3.0)
        expected = crews * n * (n - 1) // 2 + n * crews * (crews - 1) // 2
        good += len(single) == n * crews and int(np.count_nonzero(pair)) == expected
        cases += 1
    return f"{cases} of {cases}", f"{good} of {cases}", good == cases


CASES = [
    (valid_states_have_the_schedule_cost, one_hot_terms, terms_with_weight_and_hours_swapped,
     "pair term with the weight of the earlier repair and the hours of the later one"),
    (penalty_moves_only_invalid_states, one_hot_terms, terms_with_penalty_on_the_count_itself,
     "penalty on the squared count, without the 1"),
    (minimum_is_a_plan_at_the_bound, penalty_bound, mean_pair_coefficient, "the mean pair coefficient as the bound"),
    (minimum_is_a_plan_at_each_repairs_own_bound, penalty_bounds, bounds_in_schedule_order,
     "each repair given the bound of the repair at its place in the schedule order"),
    (two_crews_agree_with_one_qubit_per_repair, assignments, assignments_grouped_by_crew,
     "qubits grouped by crew in the decoder"),
    (circuit_gives_the_table_distribution, one_hot_circuit, circuit_without_fields, "single-qubit terms left out"),
    (counts_follow_the_formula, one_hot_terms, terms_without_penalty_pairs, "penalty pairs left out"),
]


if __name__ == "__main__":
    sys.exit(0 if run_checks(CASES, Path(__file__).with_suffix(".csv")) else 1)
