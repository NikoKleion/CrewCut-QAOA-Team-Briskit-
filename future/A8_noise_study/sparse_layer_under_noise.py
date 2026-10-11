# Sparse cost layer under IonQ's documented depolarisation model: gain kept with 100%, 50% and 25% of the pair terms, 12 repairs, 30 instances (WIP)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saved_run import saved_run

import numpy as np

from briskit.qaoa import train_qaoa
from crew_pairs import SEED, cost_table, instance, largest_pairs, pair_coefficients
from pauli_trajectories import noisy_distribution

REPAIRS = 12
INSTANCES = 30
KEEPS = (100, 50, 25)
DEPTH = 2
STARTS = 6
PATTERNS = 300
# one-qubit and two-qubit rates of IonQ's noise model page; the Forte 1 pair is the model before September 2025
RATES = {"no error": (0.0, 0.0),
         "Aria 1": (0.0005, 0.0133),
         "Forte 1, old model": (0.0002666666666666667, 0.004949333333333333)}


def main(instances=INSTANCES, patterns=PATTERNS):
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
            for tag, (name, (one_qubit, two_qubit)) in enumerate(RATES.items()):
                stream = np.random.default_rng([SEED, REPAIRS, k, DEPTH, keep, tag])
                runs = 1 if two_qubit == 0 else patterns
                probability = noisy_distribution(REPAIRS, kept, pruned.std(), angles, one_qubit, two_qubit, runs, stream)
                gain = (random_mean - probability @ full) / (random_mean - best)
                rows.setdefault((name, keep), []).append((gain, probability[full <= best * (1 + 1e-12)].sum(), len(kept)))
    print(f"{REPAIRS} repairs, two crews, {instances} instances, depth {DEPTH}, angles trained without errors,"
          f" {patterns} error patterns for each circuit, median over instances [quartiles]")
    print("depolarisation after every gate of the circuit as written (H, one RZZ for each pair term, RX); no readout error")
    print("   noise model            pair terms kept   pair gates   gain over random plans kept   best plan shots")
    for name in RATES:
        for keep in KEEPS:
            gain, hit, gates = (np.percentile([row[i] for row in rows[name, keep]], [25, 50, 75]) for i in range(3))
            print(f"   {name:20s}   {keep:14d}%   {DEPTH * int(gates[1]):10d}   {100 * gain[1]:6.1f}% [{100 * gain[0]:5.1f}, {100 * gain[2]:5.1f}]"
                  f"        {100 * hit[1]:5.2f}% [{100 * hit[0]:5.2f}, {100 * hit[2]:5.2f}]")
    print("the Forte 1 rates are those of the model IonQ replaced in September 2025; they do not describe the present Forte 1 backend")


if __name__ == "__main__":
    saved_run(__file__, main)
