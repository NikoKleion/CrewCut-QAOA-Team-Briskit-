# Sparse cost layer on the ideal simulator: result on the full cost with 100%, 50% and 25% of the pair terms, 30 instances of 12 repairs (WIP)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saved_run import saved_run

import numpy as np

from briskit.qaoa import train_qaoa
from crew_pairs import SEED, cost_table, instance, largest_pairs, pair_coefficients

REPAIRS = 12
INSTANCES = 30
KEEPS = (100, 50, 25)
DEPTHS = (1, 2)
STARTS = 6


def main(instances=INSTANCES):
    rows = {}
    for k in range(instances):
        weights, hours = instance(REPAIRS, k)
        pairs = pair_coefficients(weights, hours)
        full = cost_table(weights, hours, pairs)
        best, random_mean = full.min(), full.mean()
        for keep in KEEPS:
            kept = largest_pairs(pairs, keep)
            pruned = cost_table(weights, hours, kept)
            for depth in DEPTHS:
                stream = np.random.default_rng([SEED, REPAIRS, k, depth, keep])
                _, probability = train_qaoa(pruned, depth, stream, starts=STARTS)
                gain = (random_mean - probability @ full) / (random_mean - best)
                hit = probability[full <= best * (1 + 1e-12)].sum()
                rows.setdefault((depth, keep), []).append((gain, hit, len(kept)))
    print(f"{REPAIRS} repairs, two crews, {instances} instances, {STARTS} starts, trained on the kept pair terms,"
          f" scored on the full cost, median over instances [quartiles]")
    print("   depth   pair terms kept   pair gates per layer   gain over random plans kept   best plan shots")
    for depth in DEPTHS:
        for keep in KEEPS:
            gain, hit, gates = (np.percentile([row[i] for row in rows[depth, keep]], [25, 50, 75]) for i in range(3))
            print(f"   {depth:5d}   {keep:14d}%   {int(gates[1]):20d}   {100 * gain[1]:6.1f}% [{100 * gain[0]:5.1f}, {100 * gain[2]:5.1f}]"
                  f"        {100 * hit[1]:5.2f}% [{100 * hit[0]:5.2f}, {100 * hit[2]:5.2f}]")


if __name__ == "__main__":
    saved_run(__file__, main)
