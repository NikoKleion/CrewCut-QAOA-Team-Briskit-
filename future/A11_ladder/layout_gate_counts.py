# Two-qubit gates of one cost layer for both forms of crew assignment: as written, with every pair connected, and compiled to a heavy hex layout (WIP)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saved_run import saved_run

from qiskit import transpile
from qiskit.transpiler import CouplingMap

from briskit.hardware import two_qubit_count, two_qubit_depth
from briskit.one_hot import one_hot_circuit, one_hot_terms, penalty_bound
from briskit.qaoa import qaoa_circuit
from briskit.scheduling import smith_order
from crew_pairs import instance, pair_coefficients

SIZES = (6, 8, 12)
SEEDS = 3
ANGLES = [0.1, 0.1]


def counts(circuit, heavy_hex):
    connected = transpile(circuit, basis_gates=["rzz", "rx", "ry", "rz"], optimization_level=1)
    compiled = [transpile(circuit, coupling_map=heavy_hex, basis_gates=["cz", "rz", "sx", "x"], optimization_level=3,
                          seed_transpiler=seed) for seed in range(SEEDS)]
    fewest = min(compiled, key=two_qubit_count)
    return (circuit.num_qubits, two_qubit_count(connected), two_qubit_depth(connected),
            two_qubit_count(fewest), two_qubit_depth(fewest))


def main(sizes=SIZES):
    heavy_hex = CouplingMap.from_heavy_hex(7)
    print(f"one cost layer and one mixer layer; heavy hex has {heavy_hex.size()} qubits, fewest gates of {SEEDS} seeds")
    print("   repairs   form                            qubits   every pair connected: gates, depth   heavy hex: gates, depth")
    for n in sizes:
        weights, hours = instance(n, 0)
        order = smith_order(weights, hours)
        _, single, pair = one_hot_terms(weights, hours, order, 2, penalty_bound(weights, hours, order))
        forms = (("one qubit per repair", qaoa_circuit(n, pair_coefficients(weights, hours), 1.0, ANGLES, measure=False)),
                 ("one qubit per repair and crew", one_hot_circuit(single, pair, ANGLES, measure=False)))
        for label, circuit in forms:
            qubits, gates, depth, hex_gates, hex_depth = counts(circuit, heavy_hex)
            print(f"   {n:7d}   {label:30s} {qubits:6d}   {gates:21d}, {depth:5d}   {hex_gates:16d}, {hex_depth:5d}")


if __name__ == "__main__":
    saved_run(__file__, main)
