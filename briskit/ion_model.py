import numpy as np
from scipy.linalg import expm

PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
IDENTITY = np.eye(2, dtype=complex)


def spin_operator(i, n):
    out = np.array([[1.0 + 0j]])
    for k in range(n):
        out = np.kron(out, PAULI_X if k == i else IDENTITY)
    return out


def collective_drive(strength, periods, detuning=1.0, levels=30, steps=300):
    n = len(strength)
    period = 2 * np.pi / abs(detuning)
    spins = [spin_operator(i, n) for i in range(n)]
    lower = np.diag(np.sqrt(np.arange(1, levels)), 1).astype(complex)
    dt = period / steps
    unitary = np.eye(2 ** n * levels, dtype=complex)
    for k in range(int(round(max(periods) * steps))):
        t = (k + 0.5) * dt
        on = [i for i in range(n) if t < periods[i] * period]
        if not on:
            break
        spin = sum(strength[i] * spins[i] for i in on)
        mode = lower * np.exp(-1j * detuning * t) + lower.conj().T * np.exp(1j * detuning * t)
        unitary = expm(-1j * np.kron(spin, mode) * dt) @ unitary
    ground = np.arange(2 ** n) * levels
    block = unitary[np.ix_(ground, ground)]
    residual = 1.0 - np.linalg.norm(block) ** 2 / 2 ** n
    return block, residual


def target_operator(strength, periods, detuning=1.0, sign=1):
    n = len(strength)
    period = 2 * np.pi / abs(detuning)
    spins = [spin_operator(i, n) for i in range(n)]
    phase = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for i in range(n):
        for j in range(i + 1, n):
            phase += (2 * strength[i] * strength[j] / detuning) * min(periods[i], periods[j]) * period * (spins[i] @ spins[j])
    return expm(sign * 1j * phase)


def overlap(block, target):
    return abs(np.trace(target.conj().T @ block)) / block.shape[0]
