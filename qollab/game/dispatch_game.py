import time

import numpy as np
from qiskit import QuantumCircuit, transpile

from game_instance import (CREWCUT_ANGLES, EXACT, HOURS, INSTANCES, PENALTY, PENALTY_NAME, REPAIRS, RULE_BEST_ON,
                           STANDARD_ANGLES, URGENCY)


def smith_order():
    return sorted(range(REPAIRS), key=lambda j: (-URGENCY[j] / HOURS[j], j))


def pair_coefficients():
    order = smith_order()
    return {(order[a], order[b]): URGENCY[order[b]] * HOURS[order[a]]
            for a in range(REPAIRS) for b in range(a + 1, REPAIRS)}


def plan_costs():
    index = np.arange(2 ** REPAIRS)
    cost = np.full(index.size, float(np.dot(URGENCY, HOURS)))
    for (i, j), c in pair_coefficients().items():
        cost += c * (((index >> i) & 1) == ((index >> j) & 1))
    return cost


def crewcut_circuit(measure=True):
    pairs = pair_coefficients()
    scale = float(np.sqrt(sum((c / 2) ** 2 for c in pairs.values())))
    depth = len(CREWCUT_ANGLES) // 2
    circuit = QuantumCircuit(REPAIRS, REPAIRS) if measure else QuantumCircuit(REPAIRS)
    circuit.h(range(REPAIRS))
    for layer in range(depth):
        gamma, beta = CREWCUT_ANGLES[layer], CREWCUT_ANGLES[depth + layer]
        # cost layer
        for (i, j), c in pairs.items():
            circuit.rzz(gamma * c / scale, i, j)
        # mixer
        circuit.rx(2 * beta, range(REPAIRS))
    if measure:
        circuit.measure(range(REPAIRS), range(REPAIRS))
    return circuit


def standard_terms():
    size = 2 * REPAIRS
    order = smith_order()
    single = np.zeros(size)
    pair = np.zeros((size, size))
    for j in range(REPAIRS):
        single[2 * j] = single[2 * j + 1] = URGENCY[j] * HOURS[j] - PENALTY[j]
        # one repair on both crews
        pair[2 * j, 2 * j + 1] = 2 * PENALTY[j]
    for a in range(REPAIRS):
        for b in range(a + 1, REPAIRS):
            k, j = order[a], order[b]
            # two repairs on one crew, k first
            for crew in (0, 1):
                first, second = sorted((2 * k + crew, 2 * j + crew))
                pair[first, second] = URGENCY[j] * HOURS[k]
    return -single / 2 - (pair + pair.T).sum(axis=1) / 4, pair / 4


def standard_circuit(measure=True):
    fields, couplings = standard_terms()
    scale = float(np.sqrt(np.sum(fields ** 2) + np.sum(couplings ** 2)))
    size = 2 * REPAIRS
    depth = len(STANDARD_ANGLES) // 2
    circuit = QuantumCircuit(size, size) if measure else QuantumCircuit(size)
    circuit.h(range(size))
    for layer in range(depth):
        gamma, beta = STANDARD_ANGLES[layer], STANDARD_ANGLES[depth + layer]
        # cost layer
        for q in range(size):
            circuit.rz(2 * gamma * fields[q] / scale, q)
        for q in range(size):
            for r in range(q + 1, size):
                if couplings[q, r] != 0:
                    circuit.rzz(2 * gamma * couplings[q, r] / scale, q, r)
        # mixer
        circuit.rx(2 * beta, range(size))
    if measure:
        circuit.measure(range(size), range(size))
    return circuit


def decode(index):
    plan = 0
    for j in range(REPAIRS):
        group = (index >> (2 * j)) & 3
        if group not in (1, 2):
            return -1
        plan |= (group - 1) << j
    return plan


def two_qubit_count(circuit):
    return sum(1 for inst in circuit.data if inst.operation.num_qubits == 2)


def compile_for(circuit, device):
    try:
        return transpile(circuit, device, optimization_level=1, seed_transpiler=1), ""
    except Exception as error:
        return circuit, f"not compiled locally ({type(error).__name__}); the circuit was sent as written"


def backend_label(device):
    name = getattr(device, "name", type(device).__name__)
    name = name() if callable(name) else name
    noise_model = getattr(getattr(device, "options", None), "noise_model", None)
    return f"{name}, noise model {noise_model}" if isinstance(noise_model, str) and noise_model else str(name)


def run_one(device, circuit, shots, standard):
    started = time.perf_counter()
    compiled, note = compile_for(circuit, device)
    counts = device.run(compiled, shots=shots).result().get_counts()
    plans = {}
    for bits, count in counts.items():
        index = int(bits.replace(" ", ""), 2)
        plan = decode(index) if standard else index
        plans[plan] = plans.get(plan, 0) + int(count)
    return {"qubits": circuit.num_qubits, "gates": two_qubit_count(circuit), "compiled": two_qubit_count(compiled),
            "note": note, "seconds": time.perf_counter() - started,
            "plans": [[int(plan), count] for plan, count in sorted(plans.items())]}


def setup():
    return {"repairs": REPAIRS, "hours": [float(h) for h in HOURS], "urgency": [float(w) for w in URGENCY],
            "order": smith_order(), "costs": [float(c) for c in plan_costs()], "exact": EXACT,
            "penalty_name": PENALTY_NAME, "rule_best_on": RULE_BEST_ON, "instances": INSTANCES,
            "crewcut_gates": len(pair_coefficients()), "standard_gates": int(np.count_nonzero(standard_terms()[1]))}


def run_circuits(device, shots):
    shots = int(shots)
    label = backend_label(device)
    one_job = "ionq" in label.lower()
    result = {"backend": label, "shots": shots, "one_job": one_job,
              "crewcut": run_one(device, crewcut_circuit(), shots, False)}
    if not one_job:
        result["standard"] = run_one(device, standard_circuit(), shots, True)
    return result
