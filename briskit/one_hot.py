import numpy as np

from .cost_circuit import cost_circuit


def one_hot_terms(weights, hours, order, crews, penalty):
    n = len(weights)
    penalty = np.broadcast_to(np.asarray(penalty, dtype=float), (n,))
    size = n * crews
    single = np.zeros(size)
    pair = np.zeros((size, size))
    for j in range(n):
        for m in range(crews):
            single[j * crews + m] = weights[j] * hours[j] - penalty[j]
            # one repair on two crews
            for other in range(m + 1, crews):
                pair[j * crews + m, j * crews + other] = 2 * penalty[j]
    for a in range(n):
        for b in range(a + 1, n):
            k, j = order[a], order[b]
            # two repairs on one crew, k first
            for m in range(crews):
                first, second = sorted((k * crews + m, j * crews + m))
                pair[first, second] = weights[j] * hours[k]
    return float(penalty.sum()), single, pair


def penalty_bounds(weights, hours, order):
    w = np.asarray(weights, dtype=float)[order]
    p = np.asarray(hours, dtype=float)[order]
    bounds = np.empty(len(order))
    bounds[order] = w * np.cumsum(p) + p * (w.sum() - np.cumsum(w))
    return bounds


def penalty_bound(weights, hours, order):
    return float(penalty_bounds(weights, hours, order).max())


def cost_table(constant, single, pair):
    index = np.arange(2 ** len(single))
    cost = np.full(index.size, float(constant))
    for q in range(len(single)):
        cost += single[q] * ((index >> q) & 1)
    for a, b in zip(*np.nonzero(pair)):
        cost += pair[a, b] * ((index >> a) & (index >> b) & 1)
    return cost


def assignments(n, crews):
    index = np.arange(2 ** (n * crews))
    crew = np.full((index.size, n), -1, dtype=np.int8)
    for j in range(n):
        group = (index >> (j * crews)) & (2 ** crews - 1)
        for m in range(crews):
            crew[group == 1 << m, j] = m
    return (crew >= 0).all(axis=1), crew


def one_hot_fields(single, pair):
    return -single / 2 - (pair + pair.T).sum(axis=1) / 4, pair / 4


def one_hot_circuit(single, pair, angles, measure=True):
    fields, couplings = one_hot_fields(single, pair)
    return cost_circuit(fields, couplings, angles, measure)
