import json
import re
import time
from datetime import datetime, timezone

import numpy as np
from qiskit import QuantumCircuit, transpile

URGENCY = [0.8722, 5.0, 0.0296, 0.2913, 0.7377, 0.1033, 0.3519, 0.2544, 0.339, 0.4554, 0.3582, 0.4066]
HOURS = [1, 4, 6, 5, 7, 9, 7, 3, 2, 4, 4, 10]

# share of pair terms kept, then depth: gammas, then betas
ANGLES = {
    100: {
        1: [0.881586, -0.245265],
        2: [-0.66792, -2.161198, 0.299712, 0.134016],
        3: [0.592971, 1.445382, 2.12517, -0.361986, -0.18392, -0.12558],
    },
    50: {
        1: [0.911572, -0.260962],
        2: [0.699394, 2.108281, -0.314733, -0.163365],
        3: [-0.590098, -1.434354, -2.029828, 0.378219, 0.203911, 0.141021],
    },
    25: {
        1: [-1.007158, 0.321466],
        2: [-0.797142, -1.698469, 0.383024, 0.239191],
        3: [0.678894, 1.378866, 1.522507, -0.420582, -0.264119, -0.192449],
    },
}


def smith_order(w, p):
    return sorted(range(len(w)), key=lambda j: (-w[j] / p[j], j))


def pair_coefficients(w, p):
    order = smith_order(w, p)
    n = len(w)
    pairs = {}
    for a in range(n):
        for b in range(a + 1, n):
            i, j = order[a], order[b]
            pairs[(i, j)] = w[j] * p[i]
    return pairs


def cost_table(w, p, pairs):
    n = len(w)
    index = np.arange(2 ** n)
    bit = [(index >> q) & 1 for q in range(n)]
    cost = np.full(index.size, float(np.dot(w, p)))
    for (i, j), c in pairs.items():
        cost += c * (bit[i] == bit[j])
    return cost


def list_schedule(w, p):
    load, plan = [0.0, 0.0], 0
    for j in smith_order(w, p):
        crew = 0 if load[0] <= load[1] else 1
        load[crew] += p[j]
        plan |= crew << j
    return plan


def qaoa_circuit(n, pairs, scale, angles):
    depth = len(angles) // 2
    circuit = QuantumCircuit(n, n)
    circuit.h(range(n))
    for layer in range(depth):
        gamma, beta = angles[layer], angles[depth + layer]
        # cost layer
        for (i, j), c in pairs.items():
            circuit.rzz(gamma * c / scale, i, j)
        # mixer
        circuit.rx(2 * beta, range(n))
    circuit.measure(range(n), range(n))
    return circuit


def two_qubit_count(circuit):
    return sum(1 for inst in circuit.data if inst.operation.num_qubits == 2)


def compile_for(circuit, device):
    try:
        return transpile(circuit, device, optimization_level=1, seed_transpiler=1), None
    except Exception as error:
        return circuit, f"not compiled locally ({type(error).__name__}); the circuit was sent as written"


def largest_pairs(pairs, keep):
    ranked = sorted(pairs.items(), key=lambda item: -item[1])
    return dict(ranked[: round(keep / 100 * len(ranked))])


def backend_label(device):
    name = getattr(device, "name", type(device).__name__)
    name = name() if callable(name) else name
    noise_model = getattr(getattr(device, "options", None), "noise_model", None)
    return f"{name}, noise model {noise_model}" if isinstance(noise_model, str) and noise_model else str(name)


def run_circuit(device, shots, depth, keep):
    started = time.perf_counter()
    date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    w, p = np.array(URGENCY), np.array(HOURS, dtype=float)
    n = len(w)
    pairs = pair_coefficients(w, p)
    cost = cost_table(w, p, pairs)
    best = cost.min()
    kept = largest_pairs(pairs, keep)

    circuit = qaoa_circuit(n, kept, cost_table(w, p, kept).std(), ANGLES[keep][depth])
    compiled, compile_note = compile_for(circuit, device)
    counts = device.run(compiled, shots=shots).result().get_counts()

    plans = {int(bits.replace(" ", ""), 2): c for bits, c in counts.items()}
    measured = np.array(list(plans.keys()))
    weight = np.array(list(plans.values()), dtype=float)
    measured_cost = cost[measured]

    rule = cost[list_schedule(w, p)]
    best_measured = measured_cost.min()
    optimal = cost <= best * (1 + 1e-12)
    hit = weight[optimal[measured]].sum() / weight.sum()

    lines = [f"backend: {backend_label(device)}",
             f"date: {date}",
             f"shots: {shots}   depth: {depth}   share of pair terms kept: {keep}%",
             f"repairs: {n}   plans: {cost.size}   shots returned: {int(weight.sum())}   distinct plans measured: {measured.size}",
             f"pair terms in the cost layer: {len(kept)} of {len(pairs)}",
             f"two-qubit gates as written: {two_qubit_count(circuit)}"
             f"   after compiling for this backend: {compile_note or two_qubit_count(compiled)}",
             f"best possible plan                 {best:9.3f}",
             f"classical rule                     {rule:9.3f}   (+{100 * (rule / best - 1):.2f}%)",
             f"best plan among the shots          {best_measured:9.3f}   (+{100 * (best_measured / best - 1):.2f}%)",
             f"average over the shots             {float(np.average(measured_cost, weights=weight)):9.3f}",
             f"average over all plans (random)    {cost.mean():9.3f}",
             f"shots that are a best plan         {100 * hit:8.2f}%   (random: {100 * optimal.mean():.3f}%)"]
    for k in measured[np.argsort(measured_cost)][:3]:
        crews = "".join("B" if (int(k) >> q) & 1 else "A" for q in range(n))
        lines.append(f"plan {crews}   cost {cost[k]:.3f}")
    lines.append(f"seconds for circuit, job and scoring: {time.perf_counter() - started:.1f}")
    return lines


def record_name(lines, depth, keep):
    label = re.sub(r"[^a-z0-9]+", "_", lines[0].split(": ", 1)[1].lower()).strip("_")
    return f"qollab_run_{label}_keep{keep}_depth{depth}.txt"


def save_record(lines, name):
    text = "\n".join(lines) + "\n"
    try:
        # https://qollab.xyz/learn/docs/runtime-environment: sendFile places a file in the console with a copy button
        from js import sendFile
    except ImportError:
        folder = globals().get("record_folder")
        if folder:
            with open(folder + name, "w", encoding="utf-8") as file:
                file.write(text)
            print(f"record saved as {folder + name}")
        return
    sendFile(name, "text/plain", text)
    print(f"record {name} is in the console to download")


# https://qollab.xyz/learn/docs/runtime-environment: main() takes annotated int, float, str or bool settings
def main(shots: int = 2000, depth: int = 2, keep: int = 100, figures: bool = True):
    if keep not in ANGLES or depth not in ANGLES[keep]:
        print(f"keep must be one of {sorted(ANGLES)} and depth one of {sorted(ANGLES[100])}")
        return
    started = time.perf_counter()
    lines = run_circuit(backend, shots, depth, keep)
    print("\n".join(lines))
    save_record(["project file: qollab/main.py"] + lines, record_name(lines, depth, keep))
    circuit_seconds = time.perf_counter() - started
    figure_seconds = 0.0
    if figures:
        figure_started = time.perf_counter()
        draw_saved_figures()
        figure_seconds = time.perf_counter() - figure_started
    print()
    print(f"run time: circuit and scoring {circuit_seconds:.1f} s, figures {figure_seconds:.1f} s,"
          f" total {circuit_seconds + figure_seconds:.1f} s")


def draw_saved_figures():
    try:
        from project_figures import draw_all
    except ImportError as error:
        print(f"figures skipped: {error}")
        return
    # https://qollab.xyz/learn/docs/runtime-environment: a data file is loaded only when its path is a plain string
    with open("figure_data.json", encoding="utf-8") as file:
        data = json.load(file)
    print()
    print("figures from the cached results")
    draw_all(data, globals().get("figure_folder", ""))
