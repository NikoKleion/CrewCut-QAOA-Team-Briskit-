from qiskit import QuantumCircuit

TWO_QUBIT_ERROR_FORTE = 0.00495
TWO_QUBIT_NAMES = {"rzz", "rxx", "cz", "cx", "ecr", "swap"}


def complete_cost_layer(n, rng):
    circuit = QuantumCircuit(n)
    for i in range(n):
        for j in range(i + 1, n):
            circuit.rzz(float(rng.uniform(0.05, 1.0)), i, j)
    for i in range(n):
        circuit.rx(0.4, i)
    return circuit


def two_qubit_count(circuit):
    return sum(1 for inst in circuit.data if inst.operation.name in TWO_QUBIT_NAMES)


def two_qubit_depth(circuit):
    return circuit.depth(lambda inst: inst.operation.name in TWO_QUBIT_NAMES)


def unaffected_fraction(gates, rate=TWO_QUBIT_ERROR_FORTE):
    return (1 - rate) ** gates
