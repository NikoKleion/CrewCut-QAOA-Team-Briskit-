# QAOA on a pair cost under depolarisation after every gate: the outcome distribution averaged over sampled error patterns (WIP)
import numpy as np


def signs(n):
    index = np.arange(2 ** n)
    return [1.0 - 2.0 * ((index >> q) & 1) for q in range(n)]


def pauli(state, kind, qubit, sign, index):
    if kind in (1, 2):
        state = state[index ^ (1 << qubit)]
    if kind in (2, 3):
        state = state * sign[qubit]
    return state


def one_qubit_errors(state, n, rate, sign, index, rng):
    # each of the 3 Pauli matrices with chance rate / 4
    for q in np.nonzero(rng.random(n) < 3 * rate / 4)[0]:
        state = pauli(state, int(rng.integers(1, 4)), int(q), sign, index)
    return state


def noisy_distribution(n, pairs, scale, angles, one_qubit_rate, two_qubit_rate, patterns, rng):
    depth = len(angles) // 2
    index = np.arange(2 ** n)
    sign = signs(n)
    # pair gate: exp(-i gamma c / (2 scale) s_i s_j), the cost c on plans where i and j share a crew
    phases = [[np.exp(-1j * angles[layer] * c / (2 * scale) * sign[i] * sign[j]) for (i, j), c in pairs.items()]
              for layer in range(depth)]
    total = np.zeros(2 ** n)
    for _ in range(patterns):
        state = np.full(2 ** n, 2.0 ** (-n / 2), dtype=complex)
        state = one_qubit_errors(state, n, one_qubit_rate, sign, index, rng)
        for layer in range(depth):
            for (i, j), phase in zip(pairs, phases[layer]):
                state = state * phase
                # each of the 15 Pauli products with chance rate / 16
                if rng.random() < 15 * two_qubit_rate / 16:
                    kind = int(rng.integers(1, 16))
                    state = pauli(pauli(state, kind % 4, i, sign, index), kind // 4, j, sign, index)
            beta = angles[depth + layer]
            state = state.reshape((2,) * n)
            for q in range(n):
                state = np.cos(beta) * state - 1j * np.sin(beta) * np.flip(state, axis=q)
            state = one_qubit_errors(state.reshape(-1), n, one_qubit_rate, sign, index, rng)
        total += np.abs(state) ** 2
    return total / patterns
