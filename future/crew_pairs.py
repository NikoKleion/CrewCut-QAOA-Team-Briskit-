# Crew assignment for two crews as a pair cost: instances, pair terms, the largest pair terms, cost of every plan
import numpy as np

from briskit.feeder import sample_instance
from briskit.scheduling import smith_order

SEED = 20261010


def instance(repairs, k):
    return sample_instance(np.random.default_rng([SEED, repairs, k]), repairs)


def pair_coefficients(weights, hours):
    order = smith_order(weights, hours)
    n = len(weights)
    return {(order[a], order[b]): float(weights[order[b]] * hours[order[a]]) for a in range(n) for b in range(a + 1, n)}


def largest_pairs(pairs, keep):
    ranked = sorted(pairs.items(), key=lambda item: -item[1])
    return dict(ranked[: round(keep / 100 * len(ranked))])


def cost_table(weights, hours, pairs):
    index = np.arange(2 ** len(weights))
    cost = np.full(index.size, float(np.dot(weights, hours)))
    for (i, j), c in pairs.items():
        cost += c * (((index >> i) & 1) == ((index >> j) & 1))
    return cost
