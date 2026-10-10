import numpy as np
from qiskit import transpile
from qiskit.transpiler import CouplingMap

from briskit.hardware import complete_cost_layer, two_qubit_count, two_qubit_depth, unaffected_fraction

SEED = 20261010


def main(sizes=(12, 22, 29), seeds=5):
    rng = np.random.default_rng(SEED)
    heavy_hex = CouplingMap.from_heavy_hex(7)
    print(f"one cost layer with a term on every pair; heavy hex has {heavy_hex.size()} qubits, best of {seeds} seeds")
    for n in sizes:
        layer = complete_cost_layer(n, rng)
        ion = transpile(layer, basis_gates=["rzz", "rx", "ry", "rz"], optimization_level=1)
        best = None
        for seed in range(seeds):
            compiled = transpile(layer, coupling_map=heavy_hex, basis_gates=["cz", "rz", "sx", "x"],
                                 optimization_level=3, seed_transpiler=seed)
            stats = (two_qubit_count(compiled), two_qubit_depth(compiled))
            if best is None or stats[0] < best[0]:
                best = stats
        gates = two_qubit_count(ion)
        print(f"   {n:2d} repairs: all pairs {gates:3d} gates, depth {two_qubit_depth(ion):3d},"
              f" shots without a two-qubit error {100 * unaffected_fraction(gates):.0f}%"
              f" | heavy hex {best[0]:4d} gates, depth {best[1]:4d} | ratio {best[0] / gates:.2f}")


if __name__ == "__main__":
    main()
