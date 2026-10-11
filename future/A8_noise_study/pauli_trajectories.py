# QAOA on a pair cost with a random Pauli error after each pair gate: the distribution averaged over error trajectories (WIP)
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


def noisy_distribution(n, pairs, scale, angles, rate, trajectories, rng):
    depth = len(angles) // 2
    index = np.arange(2 ** n)
    sign = signs(n)
    # pair gate: exp(-i gamma c / (2 scale) s_i s_j), the cost c on plans where i and j share a crew
    phases = [[np.exp(-1j * angles[layer] * c / (2 * scale) * sign[i] * sign[j]) for (i, j), c in pairs.items()]
              for layer in range(depth)]
    total = np.zeros(2 ** n)
    for _ in range(trajectories):
        state = np.full(2 ** n, 2.0 ** (-n / 2), dtype=complex)
        for layer in range(depth):
            for (i, j), phase in zip(pairs, phases[layer]):
                state = state * phase
                if rng.random() < rate:
                    kind = int(rng.integers(1, 16))
                    state = pauli(pauli(state, kind % 4, i, sign, index), kind // 4, j, sign, index)
            beta = angles[depth + layer]
            state = state.reshape((2,) * n)
            for q in range(n):
                state = np.cos(beta) * state - 1j * np.sin(beta) * np.flip(state, axis=q)
            state = state.reshape(-1)
        total += np.abs(state) ** 2
    return total / trajectories
