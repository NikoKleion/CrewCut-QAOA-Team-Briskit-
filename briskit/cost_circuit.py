import numpy as np
from qiskit import QuantumCircuit


def ising_terms(coefficients, n, sense, penalty=0.0, count=None):
    sign = -1.0 if sense == "max" else 1.0
    upper = np.triu_indices(n, 1)
    single = sign * np.asarray(coefficients[1:n + 1], dtype=float)
    pair = np.zeros((n, n))
    pair[upper] = sign * np.asarray(coefficients[n + 1:], dtype=float)
    if count is not None:
        single = single + penalty * (1 - 2 * count)
        pair[upper] += 2 * penalty
    pair = pair + pair.T
    return -single / 2 - pair.sum(axis=1) / 4, np.triu(pair, 1) / 4


def cost_scale(fields, couplings):
    return float(np.sqrt(np.sum(fields ** 2) + np.sum(couplings ** 2)))


def cost_circuit(fields, couplings, angles, measure=True):
    n = len(fields)
    depth = len(angles) // 2
    scale = cost_scale(fields, couplings)
    circuit = QuantumCircuit(n, n) if measure else QuantumCircuit(n)
    circuit.h(range(n))
    for layer in range(depth):
        gamma, beta = angles[layer], angles[depth + layer]
        # cost layer
        for i in range(n):
            circuit.rz(2 * gamma * fields[i] / scale, i)
        for i in range(n):
            for j in range(i + 1, n):
                if couplings[i, j] != 0:
                    circuit.rzz(2 * gamma * couplings[i, j] / scale, i, j)
        # mixer
        circuit.rx(2 * beta, range(n))
    if measure:
        circuit.measure(range(n), range(n))
    return circuit
