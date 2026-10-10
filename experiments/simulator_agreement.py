import numpy as np
from qiskit.quantum_info import Statevector

from briskit.feeder import all_plans, sample_instance
from briskit.pair_cost import pair_coefficients
from briskit.qaoa import qaoa_circuit, qaoa_probabilities, standardize
from briskit.scheduling import pairwise_cost, smith_order

SEED = 20261010


def main(n=10):
    rng = np.random.default_rng(SEED)
    plans = all_plans(n)
    worst = 0.0
    for depth in (1, 2, 3):
        w, p = sample_instance(rng, n)
        cost = pairwise_cost(plans, smith_order(w, p), w, p)
        angles = rng.uniform(-0.8, 0.8, 2 * depth)
        ours = qaoa_probabilities(standardize(cost), angles)
        circuit = qaoa_circuit(n, pair_coefficients(w, p), cost.std(), angles, measure=False)
        theirs = Statevector(circuit).probabilities()
        worst = max(worst, 0.5 * np.abs(ours - theirs).sum())
    print(f"state vector simulator against the Qiskit circuit, {n} qubits, depths 1 to 3")
    print(f"   largest total variation distance {worst:.2e}")


if __name__ == "__main__":
    main()
