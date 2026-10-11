# Sparse cost layer under two-qubit gate errors: gain kept with 100%, 50% and 25% of the pair terms at the Forte 1 and Aria 1 error rates, 12 repairs (WIP)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saved_run import saved_run

import numpy as np

from briskit.hardware import TWO_QUBIT_ERROR_FORTE
from briskit.qaoa import train_qaoa
from crew_pairs import SEED, cost_table, instance, largest_pairs, pair_coefficients
from pauli_trajectories import noisy_distribution

REPAIRS = 12
INSTANCES = 30
KEEPS = (100, 50, 25)
DEPTH = 2
STARTS = 6
TRAJECTORIES = 300
# error per pair gate; the Aria 1 figure is the one used for the Qollab runs of 10 October and has no entry in SOURCES.md yet
RATES = {"no error": 0.0, "Forte 1": TWO_QUBIT_ERROR_FORTE, "Aria 1": 0.0133}


def main(instances=INSTANCES, trajectories=TRAJECTORIES):
    rows = {}
    for k in range(instances):
        weights, hours = instance(REPAIRS, k)
        pairs = pair_coefficients(weights, hours)
        full = cost_table(weights, hours, pairs)
        best, random_mean = full.min(), full.mean()
        for keep in KEEPS:
            kept = largest_pairs(pairs, keep)
            pruned = cost_table(weights, hours, kept)
            angles, _ = train_qaoa(pruned, DEPTH, np.random.default_rng([SEED, REPAIRS, k, DEPTH, keep]), starts=STARTS)
            for name, rate in RATES.items():
                stream = np.random.default_rng([SEED, REPAIRS, k, DEPTH, keep, int(rate * 1e6)])
                runs = 1 if rate == 0 else trajectories
                probability = noisy_distribution(REPAIRS, kept, pruned.std(), angles, rate, runs, stream)
                gain = (random_mean - probability @ full) / (random_mean - best)
                rows.setdefault((name, keep), []).append((gain, probability[full <= best * (1 + 1e-12)].sum(), len(kept)))
    print(f"{REPAIRS} repairs, two crews, {instances} instances, depth {DEPTH}, angles trained without errors,"
          f" {trajectories} error trajectories for each circuit, median over instances [quartiles]")
    print("a random two-qubit Pauli error after each pair gate with the given chance; single-qubit and readout errors not modelled")
    print("   error rate          pair terms kept   pair gates   gain over random plans kept   best plan shots")
    for name, rate in RATES.items():
        for keep in KEEPS:
            gain, hit, gates = (np.percentile([row[i] for row in rows[name, keep]], [25, 50, 75]) for i in range(3))
            print(f"   {name:9s} {rate:7.5f}   {keep:14d}%   {DEPTH * int(gates[1]):10d}   {100 * gain[1]:6.1f}% [{100 * gain[0]:5.1f}, {100 * gain[2]:5.1f}]"
                  f"        {100 * hit[1]:5.2f}% [{100 * hit[0]:5.2f}, {100 * hit[2]:5.2f}]")


if __name__ == "__main__":
    saved_run(__file__, main)
