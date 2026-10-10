import numpy as np


def depth_first_order(parent):
    n = len(parent)
    below = [[] for _ in range(n)]
    roots = []
    for line, above in enumerate(parent):
        if above == -1:
            roots.append(line)
        else:
            below[above].append(line)
    order = []
    stack = roots[::-1]
    while stack:
        line = stack.pop()
        order.append(line)
        stack.extend(reversed(below[line]))
    size = np.ones(n, dtype=int)
    for line in reversed(order):
        if parent[line] != -1:
            size[parent[line]] += size[line]
    return order, [int(size[line]) for line in order]


def best_closed_set(weights, parent, capacity, costs=None):
    n = len(parent)
    weights = np.asarray(weights, dtype=float)
    costs = np.ones(n, dtype=int) if costs is None else np.asarray(costs, dtype=int)
    order, size = depth_first_order(parent)
    table = np.zeros((n + 1, capacity + 1))
    for i in range(n - 1, -1, -1):
        line = order[i]
        cost = int(costs[line])
        # line left out: nothing below it counts
        table[i] = table[i + size[i]]
        # line taken
        if cost <= capacity:
            table[i, cost:] = np.maximum(table[i, cost:], weights[line] + table[i + 1, : capacity + 1 - cost])
    chosen = np.zeros(n, dtype=np.int8)
    i, c = 0, capacity
    while i < n:
        line = order[i]
        cost = int(costs[line])
        if cost <= c and weights[line] + table[i + 1, c - cost] >= table[i, c] - 1e-12:
            chosen[line] = 1
            i, c = i + 1, c - cost
        else:
            i += size[i]
    return float(table[0, capacity]), chosen


def tree_optimum(instance, budget, rng):
    return best_closed_set(instance.data["weights"], instance.data["parent"], instance.count)[1][None, :]
