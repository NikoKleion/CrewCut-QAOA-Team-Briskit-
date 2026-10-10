import numpy as np
from qiskit import QuantumCircuit
from scipy.optimize import minimize


def qaoa_probabilities(cost, angles):
    n = int(np.log2(cost.size))
    depth = len(angles) // 2
    state = np.full(cost.size, 1 / np.sqrt(cost.size), dtype=complex)
    for k in range(depth):
        state = state * np.exp(-1j * angles[k] * cost)
        state = state.reshape((2,) * n)
        beta = angles[depth + k]
        for q in range(n):
            state = np.cos(beta) * state - 1j * np.sin(beta) * np.flip(state, axis=q)
        state = state.reshape(-1)
    return np.abs(state) ** 2


def standardize(cost):
    return (cost - cost.mean()) / cost.std()


def train_qaoa(cost, depth, rng, starts=6, maxiter=None):
    z = standardize(cost)
    energy = lambda a: float(qaoa_probabilities(z, a) @ z)
    options = {} if maxiter is None else {"maxiter": maxiter}
    best = None
    for s in range(starts):
        ramp = np.linspace(0.2, 1.0, depth)
        start = np.concatenate([0.4 * ramp, 0.4 * ramp[::-1]]) if s == 0 else rng.uniform(-0.8, 0.8, 2 * depth)
        result = minimize(energy, start, method="BFGS", options=options)
        if best is None or result.fun < best.fun:
            best = result
    return best.x, qaoa_probabilities(z, best.x)


def expected_best_of(probabilities, values, shots):
    order = np.argsort(values)
    cumulative = np.clip(np.cumsum(probabilities[order]), 0, 1)
    previous = np.concatenate([[0.0], cumulative[:-1]])
    return float(np.sum(values[order] * ((1 - previous) ** shots - (1 - cumulative) ** shots)))


def qaoa_circuit(n, pairs, scale, angles, measure=True):
    depth = len(angles) // 2
    circuit = QuantumCircuit(n, n) if measure else QuantumCircuit(n)
    circuit.h(range(n))
    for layer in range(depth):
        gamma, beta = angles[layer], angles[depth + layer]
        # cost layer
        for (i, j), c in pairs.items():
            circuit.rzz(gamma * c / scale, i, j)
        # mixer
        circuit.rx(2 * beta, range(n))
    if measure:
        circuit.measure(range(n), range(n))
    return circuit
