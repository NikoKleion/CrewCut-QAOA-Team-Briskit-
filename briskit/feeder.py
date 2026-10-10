import numpy as np

IEEE13_LINES = [(650, 632), (632, 633), (633, 634), (632, 645), (645, 646), (632, 671),
                (671, 680), (671, 684), (684, 611), (684, 652), (671, 692), (692, 675)]


def upstream_parents(lines=IEEE13_LINES):
    into = {b: j for j, (a, b) in enumerate(lines)}
    return [into.get(a, -1) for a, b in lines]


def random_tree_parents(rng, n):
    return [-1] + [int(rng.integers(0, j)) for j in range(1, n)]


def topological_order(parent):
    n = len(parent)
    out, seen = [], set()
    while len(out) < n:
        for j in range(n):
            if j not in seen and (parent[j] == -1 or parent[j] in seen):
                out.append(j)
                seen.add(j)
    return out


def sample_instance(rng, n):
    weights = rng.uniform(0.0, 1.0, n)
    weights[rng.integers(n)] = 5.0
    hours = rng.integers(1, 11, n).astype(float)
    return weights, hours


def all_plans(n):
    idx = np.arange(2 ** n)
    return ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)


def plan_index(plan):
    return int(np.dot(np.asarray(plan).astype(int), 2 ** np.arange(len(plan))))


def restoration_value(plans, weights, parent):
    live = np.zeros(plans.shape, dtype=float)
    for j in topological_order(parent):
        live[:, j] = plans[:, j] if parent[j] == -1 else plans[:, j] * live[:, parent[j]]
    return live @ weights
