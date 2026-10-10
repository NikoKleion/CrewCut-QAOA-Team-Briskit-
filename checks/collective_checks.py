import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector, state_fidelity
from scipy.linalg import expm

from briskit.collective import (collective_block, collective_layer, collective_qaoa_circuit, eigen_blocks_of,
                                level_blocks)
from briskit.feeder import all_plans, sample_instance
from briskit.pair_cost import level_vectors, pair_coefficients
from briskit.qaoa import qaoa_circuit, qaoa_probabilities, standardize
from briskit.scheduling import pairwise_cost, smith_order
from checks.record import run_checks

SEED = 20261010


def z_on(qubit, n):
    out = np.array([[1.0]])
    for position in reversed(range(n)):
        out = np.kron(out, np.diag([1.0, -1.0]) if position == qubit else np.eye(2))
    return out


def pairwise_layer(n, pairs, gamma, scale):
    layer = QuantumCircuit(n)
    for (i, j), c in pairs.items():
        layer.rzz(gamma * c / scale, i, j)
    return layer


def leveled_repairs(rng, n, levels):
    hours = rng.integers(1, 11, n).astype(float)
    ratios = np.sort(rng.uniform(0.2, 3.0, levels))[::-1]
    level = rng.integers(0, levels, n)
    level[:levels] = np.arange(levels)
    return ratios[level] * hours, hours


def block_with_half_the_angle(amplitudes, angle):
    return collective_block(amplitudes, angle / 2)


def levels_with_one_block_left_out(weights, hours):
    strengths, amplitudes = level_blocks(weights, hours)
    return strengths[:-1], amplitudes[:-1]


def levels_with_one_sign_flipped(weights, hours):
    strengths, amplitudes = level_blocks(weights, hours)
    amplitudes = amplitudes.copy()
    amplitudes[-1, 0] *= -1
    return strengths, amplitudes


def eigen_blocks_all_positive(pairs, n):
    strengths, amplitudes = eigen_blocks_of(pairs, n)
    return np.abs(strengths), amplitudes


def three_eigen_blocks_of_seven(pairs, n):
    return eigen_blocks_of(pairs, n, keep=3)


def circuit_with_one_amplitude_too_large(n, strengths, amplitudes, scale, angles, basis="z", measure=True):
    amplitudes = amplitudes.copy()
    amplitudes[-1, 0] *= 1.1
    return collective_qaoa_circuit(n, strengths, amplitudes, scale, angles, basis, measure)


def circuit_without_final_hadamards(n, strengths, amplitudes, scale, angles, basis="z", measure=True):
    circuit = collective_qaoa_circuit(n, strengths, amplitudes, scale, angles, basis, measure)
    if basis == "x":
        circuit.h(range(n))
    return circuit


def blocks_that_ignore_the_hours(weights, hours):
    return level_blocks(weights, np.ones(len(weights)))


def blocks_without_rounding(weights, hours):
    gaps, vectors = level_vectors(hours, weights / hours)
    return gaps / 4, vectors


def block_matches_matrix_exponential(block):
    rng = np.random.default_rng(SEED)
    n, equal = 4, 0
    for _ in range(10):
        amplitudes = rng.uniform(-3.0, 3.0, n)
        strength = rng.uniform(-0.5, 0.5)
        collective = sum(amplitudes[q] * z_on(q, n) for q in range(n))
        target = Operator(expm(-1j * strength * collective @ collective))
        equal += Operator(block(amplitudes, strength)).equiv(target)
    return "10 of 10 equal up to phase", f"{equal} of 10", equal == 10


def level_layer_equals_pairwise(blocks):
    rng = np.random.default_rng(SEED)
    n, gamma = 7, 0.37
    weights, hours = leveled_repairs(rng, n, 3)
    pairs = pair_coefficients(weights, hours)
    scale = max(pairs.values())
    same = Operator(collective_layer(n, *blocks(weights, hours), gamma, scale)).equiv(
        Operator(pairwise_layer(n, pairs, gamma, scale)))
    return "equal up to phase", "equal up to phase" if same else "different", same


def eigen_layer_equals_pairwise(blocks):
    rng = np.random.default_rng(SEED)
    n, gamma = 7, 0.41
    pairs = {(i, j): float(rng.uniform(-1.0, 1.0)) for i in range(n) for j in range(i + 1, n)}
    strengths, amplitudes = blocks(pairs, n)
    same = Operator(collective_layer(n, strengths, amplitudes, gamma, 1.0)).equiv(
        Operator(pairwise_layer(n, pairs, gamma, 1.0)))
    signs = f"{int(np.sum(strengths < 0))} negative and {int(np.sum(strengths > 0))} positive blocks"
    return "equal up to phase", f"{'equal up to phase' if same else 'different'} with {signs}", same


def circuit_matches_table_and_pairwise(build):
    rng = np.random.default_rng(SEED)
    n = 12
    weights, hours = sample_instance(rng, n)
    cost = pairwise_cost(all_plans(n), smith_order(weights, hours), weights, hours)
    angles = rng.uniform(-0.8, 0.8, 4)
    strengths, amplitudes = level_blocks(weights, hours)
    collective = Statevector(build(n, strengths, amplitudes, cost.std(), angles, measure=False))
    pairwise = Statevector(qaoa_circuit(n, pair_coefficients(weights, hours), cost.std(), angles, measure=False))
    distance = 0.5 * float(np.abs(collective.probabilities() - qaoa_probabilities(standardize(cost), angles)).sum())
    infidelity = abs(1.0 - state_fidelity(collective, pairwise))
    return ("both below 1e-10", f"distance to the table simulator {distance:.1e}, 1 - fidelity {infidelity:.1e}",
            distance < 1e-10 and infidelity < 1e-10)


def x_form_matches_z_form(build):
    rng = np.random.default_rng(SEED)
    n = 8
    weights, hours = sample_instance(rng, n)
    cost = pairwise_cost(all_plans(n), smith_order(weights, hours), weights, hours)
    angles = rng.uniform(-0.8, 0.8, 4)
    strengths, amplitudes = level_blocks(weights, hours)
    z_form = Statevector(build(n, strengths, amplitudes, cost.std(), angles, "z", measure=False))
    x_form = Statevector(build(n, strengths, amplitudes, cost.std(), angles, "x", measure=False))
    distance = 0.5 * float(np.abs(z_form.probabilities() - x_form.probabilities()).sum())
    return "distance below 1e-10", f"{distance:.1e}", distance < 1e-10


def layer_changes_with_one_hour(blocks):
    rng = np.random.default_rng(SEED)
    n, gamma = 6, 0.37
    weights, hours = leveled_repairs(rng, n, 2)
    longer = hours.copy()
    longer[0] += 1.0
    changed = not Operator(collective_layer(n, *blocks(weights, hours), gamma, 10.0)).equiv(
        Operator(collective_layer(n, *blocks(weights, longer), gamma, 10.0)))
    return "layer changes", "layer changes" if changed else "layer unchanged", changed


def three_levels_give_three_blocks(blocks):
    rng = np.random.default_rng(SEED)
    sample_instance(rng, 12)
    weights, hours = leveled_repairs(rng, 12, 3)
    count = len(blocks(weights, hours)[0])
    return 3, count, count == 3


CASES = [
    (block_matches_matrix_exponential, collective_block, block_with_half_the_angle, "half the angle"),
    (level_layer_equals_pairwise, level_blocks, levels_with_one_block_left_out, "one block left out"),
    (level_layer_equals_pairwise, level_blocks, levels_with_one_sign_flipped, "one amplitude with its sign flipped"),
    (eigen_layer_equals_pairwise, eigen_blocks_of, eigen_blocks_all_positive, "negative eigenvalues taken as positive"),
    (eigen_layer_equals_pairwise, eigen_blocks_of, three_eigen_blocks_of_seven, "three blocks of seven"),
    (circuit_matches_table_and_pairwise, collective_qaoa_circuit, circuit_with_one_amplitude_too_large,
     "one amplitude 10% too large"),
    (x_form_matches_z_form, collective_qaoa_circuit, circuit_without_final_hadamards,
     "no Hadamards before measurement"),
    (layer_changes_with_one_hour, level_blocks, blocks_that_ignore_the_hours, "blocks that ignore the hours"),
    (three_levels_give_three_blocks, level_blocks, blocks_without_rounding, "ratios not rounded"),
]


if __name__ == "__main__":
    sys.exit(0 if run_checks(CASES, Path(__file__).with_suffix(".csv")) else 1)
