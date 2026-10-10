import sys
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

from briskit.cost_circuit import cost_circuit, cost_scale, ising_terms
from briskit.feeder import all_plans, sample_instance, upstream_parents
from briskit.instances import restoration
from briskit.pair_cost import ridge_coefficients, selection_features
from briskit.qaoa import qaoa_probabilities, standardize, train_qaoa
from briskit.solvers import allowed, fitted_merit
from checks.record import run_checks

SEED = 20261010


def random_coefficients(rng, n):
    return np.concatenate([rng.uniform(-1.0, 1.0, 1 + n), rng.uniform(-0.5, 0.5, n * (n - 1) // 2)])


def selector_cost(coefficients, n, sense, count):
    plans = all_plans(n)
    merit = fitted_merit(coefficients, plans, sense)
    if count is None:
        return -merit, 0.0
    chosen = plans.sum(axis=1)
    penalty = float(merit[chosen == count].std())
    return -merit + penalty * (chosen - count) ** 2, penalty


def table_from_terms(fields, couplings):
    z = 1.0 - 2.0 * all_plans(len(fields))
    return z @ fields + np.einsum("ri,ij,rj->r", z, couplings, z)


def distance(first, second):
    return 0.5 * float(np.abs(first - second).sum())


def terms_without_the_penalty(coefficients, n, sense, penalty=0.0, count=None):
    return ising_terms(coefficients, n, sense)


def scale_from_pair_terms_alone(fields, couplings):
    return float(np.sqrt(np.sum(couplings ** 2)))


def circuit_without_single_terms(fields, couplings, angles, measure=True):
    return cost_circuit(np.zeros_like(fields), couplings, angles, measure)


def terms_reproduce_the_table(terms):
    rng = np.random.default_rng(SEED)
    n, worst = 8, 0.0
    for sense, count in (("max", 4), ("min", 3), ("min", None)):
        coefficients = random_coefficients(rng, n)
        cost, penalty = selector_cost(coefficients, n, sense, count)
        difference = cost - table_from_terms(*terms(coefficients, n, sense, penalty, count))
        worst = max(worst, float(difference.max() - difference.min()))
    return "differs from the table by one constant, spread below 1e-9", f"spread {worst:.1e}", worst < 1e-9


def scale_equals_table_deviation(scale):
    rng = np.random.default_rng(SEED)
    n, worst = 10, 0.0
    for sense, count in (("max", 5), ("min", None)):
        coefficients = random_coefficients(rng, n)
        cost, penalty = selector_cost(coefficients, n, sense, count)
        fields, couplings = ising_terms(coefficients, n, sense, penalty, count)
        worst = max(worst, abs(scale(fields, couplings) / cost.std() - 1))
    return "relative difference below 1e-12", f"{worst:.1e}", worst < 1e-12


def circuit_matches_table_simulator(build):
    rng = np.random.default_rng(SEED)
    worst = 0.0
    for n in (8, 12):
        for depth in (1, 2, 3):
            coefficients = random_coefficients(rng, n)
            cost, penalty = selector_cost(coefficients, n, "max", n // 2)
            angles = rng.uniform(-0.8, 0.8, 2 * depth)
            circuit = build(*ising_terms(coefficients, n, "max", penalty, n // 2), angles, measure=False)
            worst = max(worst, distance(Statevector(circuit).probabilities(),
                                        qaoa_probabilities(standardize(cost), angles)))
    return "total variation below 1e-9 at 8 and 12 qubits, depths 1 to 3", f"{worst:.1e}", worst < 1e-9


def circuit_matches_trained_selector(terms):
    rng = np.random.default_rng(SEED)
    weights, _ = sample_instance(rng, 12)
    instance = restoration("feeder", weights, upstream_parents(), 6)
    rows, plans = allowed(instance)
    sample = rows[rng.choice(len(rows), size=150, replace=False)]
    coefficients = ridge_coefficients(selection_features(plans[sample]), instance.score(plans[sample]))
    merit = fitted_merit(coefficients, plans, instance.sense)
    penalty = float(merit[rows].std())
    cost = -merit + penalty * (plans.sum(axis=1) - instance.count) ** 2
    angles, probability = train_qaoa(cost, 2, rng, starts=2)
    circuit = cost_circuit(*terms(coefficients, 12, instance.sense, penalty, instance.count), angles, measure=False)
    gap = distance(Statevector(circuit).probabilities(), probability)
    return "total variation below 1e-9 from the trained distribution", f"{gap:.1e}", gap < 1e-9


def one_coefficient_changes_the_distribution(build):
    rng = np.random.default_rng(SEED)
    n = 8
    coefficients = random_coefficients(rng, n)
    raised = coefficients.copy()
    raised[1] += 0.5
    angles = rng.uniform(-0.8, 0.8, 4)
    before = Statevector(build(*ising_terms(coefficients, n, "min"), angles, measure=False)).probabilities()
    after = Statevector(build(*ising_terms(raised, n, "min"), angles, measure=False)).probabilities()
    moved = distance(before, after)
    return "distribution moves by more than 1e-6", f"{moved:.1e}", moved > 1e-6


CASES = [
    (terms_reproduce_the_table, ising_terms, terms_without_the_penalty, "the penalty left out of the terms"),
    (scale_equals_table_deviation, cost_scale, scale_from_pair_terms_alone, "scale from the pair terms alone"),
    (circuit_matches_table_simulator, cost_circuit, circuit_without_single_terms, "single terms left out"),
    (circuit_matches_trained_selector, ising_terms, terms_without_the_penalty, "the penalty left out of the circuit"),
    (one_coefficient_changes_the_distribution, cost_circuit, circuit_without_single_terms,
     "a circuit that ignores the single terms"),
]


if __name__ == "__main__":
    sys.exit(0 if run_checks(CASES, Path(__file__).with_suffix(".csv")) else 1)
