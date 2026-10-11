# Checks of the error pattern simulator, each run on the real code and on a broken version
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import saved_run

import numpy as np

from briskit.qaoa import qaoa_probabilities, standardize
from checks.record import run_checks
from crew_pairs import cost_table, instance, pair_coefficients
from pauli_trajectories import noisy_distribution

SEED = 20261010
PAULIS = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0])]


def case(repairs):
    weights, hours = instance(repairs, 0)
    pairs = pair_coefficients(weights, hours)
    return pairs, cost_table(weights, hours, pairs), np.random.default_rng(SEED).uniform(-0.8, 0.8, 2)


def on(operators, n):
    # operators: {qubit: 2 by 2 matrix}; qubit q is bit q of the state index
    out = np.eye(1)
    for q in range(n - 1, -1, -1):
        out = np.kron(out, operators.get(q, np.eye(2)))
    return out


def depolarise(rho, qubits, rate, n):
    kinds = [kind for kind in itertools.product(range(4), repeat=len(qubits)) if any(kind)]
    share = rate / 4 ** len(qubits)
    out = (1 - len(kinds) * share) * rho
    for kind in kinds:
        error = on({q: PAULIS[k] for q, k in zip(qubits, kind)}, n)
        out = out + share * error @ rho @ error.conj().T
    return out


def kraus_distribution(n, pairs, scale, angles, one_qubit_rate, two_qubit_rate):
    plus = np.full(2 ** n, 2.0 ** (-n / 2))
    rho = np.outer(plus, plus).astype(complex)
    for q in range(n):
        rho = depolarise(rho, [q], one_qubit_rate, n)
    for (i, j), c in pairs.items():
        zz = np.diag(on({i: PAULIS[3], j: PAULIS[3]}, n))
        gate = np.diag(np.exp(-1j * angles[0] * c / (2 * scale) * zz))
        rho = depolarise(gate @ rho @ gate.conj().T, [i, j], two_qubit_rate, n)
    for q in range(n):
        gate = on({q: np.cos(angles[1]) * PAULIS[0] - 1j * np.sin(angles[1]) * PAULIS[1]}, n)
        rho = depolarise(gate @ rho @ gate.conj().T, [q], one_qubit_rate, n)
    return np.real(np.diag(rho))


def with_doubled_pair_angle(n, pairs, scale, angles, one_qubit_rate, two_qubit_rate, patterns, rng):
    return noisy_distribution(n, pairs, scale / 2, angles, one_qubit_rate, two_qubit_rate, patterns, rng)


def with_an_error_at_the_full_rate(n, pairs, scale, angles, one_qubit_rate, two_qubit_rate, patterns, rng):
    return noisy_distribution(n, pairs, scale, angles, one_qubit_rate, 16 * two_qubit_rate / 15, patterns, rng)


def without_one_qubit_errors(n, pairs, scale, angles, one_qubit_rate, two_qubit_rate, patterns, rng):
    return noisy_distribution(n, pairs, scale, angles, 0.0, two_qubit_rate, patterns, rng)


def without_normalising(n, pairs, scale, angles, one_qubit_rate, two_qubit_rate, patterns, rng):
    return noisy_distribution(n, pairs, scale, angles, one_qubit_rate, two_qubit_rate, patterns, rng) * patterns / (patterns + 1)


def no_error_gives_the_table_distribution(simulate):
    pairs, table, _ = case(8)
    angles = np.random.default_rng(SEED).uniform(-0.8, 0.8, 4)
    got = simulate(8, pairs, table.std(), angles, 0.0, 0.0, 1, np.random.default_rng(SEED))
    worst = float(np.abs(got - qaoa_probabilities(standardize(table), angles)).max())
    return "difference below 1e-12", f"largest difference {worst:.1e}", worst < 1e-12


def distance_to_the_kraus_channel(simulate, one_qubit_rate, two_qubit_rate, patterns):
    pairs, table, angles = case(3)
    exact = kraus_distribution(3, pairs, table.std(), angles, one_qubit_rate, two_qubit_rate)
    got = simulate(3, pairs, table.std(), angles, one_qubit_rate, two_qubit_rate, patterns, np.random.default_rng(SEED))
    return float(np.abs(got - exact).max())


def two_qubit_errors_follow_the_kraus_channel(simulate):
    # at rate 0.3 the channel moves by 0.0020 between r and 16 r / 15
    worst = distance_to_the_kraus_channel(simulate, 0.0, 0.3, 160000)
    return "largest difference below 0.001", f"largest difference {worst:.4f}", worst < 0.001


def one_qubit_errors_follow_the_kraus_channel(simulate):
    worst = distance_to_the_kraus_channel(simulate, 0.6, 0.0, 40000)
    return "largest difference below 0.004", f"largest difference {worst:.4f}", worst < 0.004


def every_average_is_a_distribution(simulate):
    pairs, table, _ = case(8)
    angles = np.random.default_rng(SEED).uniform(-0.8, 0.8, 4)
    got = simulate(8, pairs, table.std(), angles, 0.1, 0.2, 20, np.random.default_rng(SEED))
    return "sums to 1 within 1e-12", f"sum off by {abs(got.sum() - 1):.1e}", abs(got.sum() - 1) < 1e-12 and got.min() >= 0


CASES = [
    (no_error_gives_the_table_distribution, noisy_distribution, with_doubled_pair_angle, "pair gate angle doubled"),
    (two_qubit_errors_follow_the_kraus_channel, noisy_distribution, with_an_error_at_the_full_rate,
     "a two-qubit error with chance r in place of 15 r / 16"),
    (one_qubit_errors_follow_the_kraus_channel, noisy_distribution, without_one_qubit_errors, "one-qubit errors left out"),
    (every_average_is_a_distribution, noisy_distribution, without_normalising, "average taken over one pattern too many"),
]


if __name__ == "__main__":
    sys.exit(0 if run_checks(CASES, Path(__file__).with_suffix(".csv")) else 1)
