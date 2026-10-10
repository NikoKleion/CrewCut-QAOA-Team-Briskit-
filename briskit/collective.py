import numpy as np
from qiskit import QuantumCircuit

from .pair_cost import eigen_blocks, level_vectors


def level_blocks(weights, hours):
    hours = np.asarray(hours, dtype=float)
    gaps, vectors = level_vectors(hours, np.round(np.asarray(weights, dtype=float) / hours, 12))
    return gaps / 4, vectors


def pair_matrix(pairs, n):
    matrix = np.zeros((n, n))
    for (i, j), c in pairs.items():
        matrix[i, j] = matrix[j, i] = c
    return matrix


def eigen_blocks_of(pairs, n, keep=None):
    values, vectors = eigen_blocks(pair_matrix(pairs, n))
    keep = n if keep is None else keep
    return values[:keep] / 4, vectors[:, :keep].T


def collective_block(amplitudes, angle, basis="z"):
    n = len(amplitudes)
    block = QuantumCircuit(n, name="collective")
    for i in range(n):
        for j in range(i + 1, n):
            turn = 4 * angle * amplitudes[i] * amplitudes[j]
            if turn == 0:
                continue
            if basis == "z":
                block.rzz(turn, i, j)
            else:
                block.rxx(turn, i, j)
    return block.to_gate()


def collective_layer(n, strengths, amplitudes, gamma, scale, basis="z"):
    layer = QuantumCircuit(n)
    for strength, block in zip(strengths, amplitudes):
        layer.append(collective_block(block, gamma * strength / scale, basis), range(n))
    return layer


def collective_qaoa_circuit(n, strengths, amplitudes, scale, angles, basis="z", measure=True):
    depth = len(angles) // 2
    circuit = QuantumCircuit(n, n) if measure else QuantumCircuit(n)
    if basis == "z":
        circuit.h(range(n))
    for layer in range(depth):
        gamma, beta = angles[layer], angles[depth + layer]
        # cost layer
        circuit.compose(collective_layer(n, strengths, amplitudes, gamma, scale, basis), inplace=True)
        # mixer
        if basis == "z":
            circuit.rx(2 * beta, range(n))
        else:
            circuit.rz(2 * beta, range(n))
    if basis == "x":
        circuit.h(range(n))
    if measure:
        circuit.measure(range(n), range(n))
    return circuit
