import itertools

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate
from qiskit.quantum_info import Statevector, state_fidelity

from briskit.feeder import all_plans
from briskit.pair_cost import level_vectors

SEED = 20261010


def leveled_instance(rng, n, levels):
    hours = rng.integers(1, 11, n).astype(float)
    ratios = np.sort(rng.uniform(0.2, 3.0, levels))[::-1]
    level = rng.integers(0, levels, n)
    level[:levels] = np.arange(levels)
    ratio = ratios[level]
    return hours, ratio * hours, ratio


def ordered_cost(plan, hours, weights, ratio):
    total = 0.0
    for crew in (0, 1):
        jobs = sorted((j for j in range(len(hours)) if plan[j] == crew), key=lambda j: -ratio[j])
        t = 0.0
        for j in jobs:
            t += hours[j]
            total += weights[j] * t
    return total


def algebra_errors(rng, levels, cases=20, n=9):
    worst_matrix, worst_cost, ranks = 0.0, 0.0, set()
    for _ in range(cases):
        hours, weights, ratio = leveled_instance(rng, n, levels)
        coupling = np.outer(hours, hours) * np.minimum.outer(ratio, ratio)
        gaps, vectors = level_vectors(hours, ratio)
        rebuilt = sum(gaps[k] * np.outer(vectors[k], vectors[k]) for k in range(levels))
        worst_matrix = max(worst_matrix, np.abs(coupling - rebuilt).max())
        ranks.add(int(np.linalg.matrix_rank(coupling)))
        constant = None
        for plan in itertools.product([0, 1], repeat=n):
            spins = 1.0 - 2.0 * np.array(plan)
            form = 0.25 * sum(gaps[k] * (vectors[k] @ spins) ** 2 for k in range(levels))
            difference = ordered_cost(plan, hours, weights, ratio) - form
            constant = difference if constant is None else constant
            worst_cost = max(worst_cost, abs(difference - constant))
    return worst_matrix, worst_cost, sorted(ranks)


def layer_fidelity(rng, n=10, levels=3, gamma=0.37):
    hours, weights, ratio = leveled_instance(rng, n, levels)
    coupling = np.outer(hours, hours) * np.minimum.outer(ratio, ratio)
    scale = 1.0 / coupling.max()
    coupling = coupling * scale
    gaps, vectors = level_vectors(hours, ratio)
    gaps = gaps * scale

    pairwise = QuantumCircuit(n)
    pairwise.h(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            pairwise.rzz(2 * gamma * 0.5 * coupling[i, j], i, j)

    spins = 1.0 - 2.0 * all_plans(n)
    collective = QuantumCircuit(n)
    collective.h(range(n))
    for k in range(levels):
        total = spins @ vectors[k]
        collective.append(DiagonalGate(list(np.exp(-1j * gamma * 0.25 * gaps[k] * total ** 2))), range(n))

    return state_fidelity(Statevector(pairwise), Statevector(collective))


def main():
    rng = np.random.default_rng(SEED)
    print("pair matrix and cost against brute force, 9 repairs, every one of the 512 assignments, 20 cases per row")
    for levels in (1, 2, 4):
        matrix, cost, ranks = algebra_errors(rng, levels)
        print(f"   {levels} level(s)  pair matrix against the sum over levels {matrix:.1e}  rank {ranks}"
              f"  cost against brute force up to one constant {cost:.1e}")
    print("one cost layer, 10 repairs, 3 levels: one rotation per pair against one operation per level")
    print(f"   state fidelity {layer_fidelity(rng):.12f}")
    print(f"   entangling operations in the layer: {10 * 9 // 2} per pair, 3 per level")


if __name__ == "__main__":
    main()
