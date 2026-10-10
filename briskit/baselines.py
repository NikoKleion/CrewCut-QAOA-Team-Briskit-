import heapq

import numpy as np


def bit_flip_descent(values, start, n):
    current, evaluations = start, 1
    improved = True
    while improved:
        improved = False
        for q in range(n):
            neighbour = current ^ (1 << q)
            evaluations += 1
            if values[neighbour] < values[current] - 1e-12:
                current, improved = neighbour, True
    return values[current], evaluations


def descent_endpoints(cost, n):
    idx = np.arange(cost.size)
    step, best = idx.copy(), cost.copy()
    for q in range(n):
        neighbour = idx ^ (1 << q)
        better = cost[neighbour] < best - 1e-12
        step[better], best[better] = neighbour[better], cost[neighbour][better]
    end = step
    while True:
        further = end[end]
        if np.array_equal(further, end):
            return end
        end = further


def karmarkar_karp(numbers):
    heap = [(-x, i) for i, x in enumerate(numbers)]
    heapq.heapify(heap)
    edges = []
    while len(heap) > 1:
        a, i = heapq.heappop(heap)
        b, j = heapq.heappop(heap)
        edges.append((i, j))
        heapq.heappush(heap, (a - b, i))
    side = {heap[0][1]: 1}
    for i, j in reversed(edges):
        side[j] = -side[i]
    return np.array([side[k] for k in range(len(numbers))])


def longest_first_difference(numbers):
    load = [0.0, 0.0]
    for x in sorted(numbers, reverse=True):
        load[int(load[1] < load[0])] += x
    return abs(load[0] - load[1])


def thermal_distributions(values, betas):
    z = (values - values.min()) / values.std()
    out = []
    for beta in betas:
        weight = np.exp(-beta * z)
        out.append(weight / weight.sum())
    return out


def matched_thermal(values, betas, target_mean):
    candidates = thermal_distributions(values, betas)
    return candidates[int(np.argmin([abs(float(c @ values) - target_mean) for c in candidates]))]


def greedy_marginal(value_table, n, count):
    chosen = 0
    for _ in range(count):
        gains = [value_table[chosen | (1 << j)] if not (chosen >> j) & 1 else -np.inf for j in range(n)]
        top = max(gains)
        chosen |= 1 << next(j for j in range(n) if gains[j] >= top - 1e-12)
    return chosen
