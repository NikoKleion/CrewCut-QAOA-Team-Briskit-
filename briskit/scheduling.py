import itertools

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .feeder import topological_order


def smith_order(weights, hours):
    return sorted(range(len(weights)), key=lambda j: (-weights[j] / hours[j], j))


def single_crew_order(weights, hours, parent):
    n = len(weights)
    root = parent.index(-1)
    w, p = [float(x) for x in weights], [float(x) for x in hours]
    seq = {j: [j] for j in range(n)}
    group = list(range(n))
    alive = set(range(n)) - {root}
    while alive:
        j = max(alive, key=lambda k: w[k] / p[k])
        i = group[parent[j]]
        w[i] += w[j]
        p[i] += p[j]
        seq[i] = seq[i] + seq[j]
        for k in seq[j]:
            group[k] = i
        alive.remove(j)
    return seq[root]


def weighted_completion_cost(order, weights, hours):
    t, cost = 0.0, 0.0
    for j in order:
        t += hours[j]
        cost += weights[j] * t
    return cost


def brute_force_single_crew(weights, hours, parent):
    n = len(weights)
    best = np.inf
    for perm in itertools.permutations(range(n)):
        pos = {j: k for k, j in enumerate(perm)}
        if all(parent[j] == -1 or pos[parent[j]] < pos[j] for j in range(n)):
            best = min(best, weighted_completion_cost(perm, weights, hours))
    return best


def completion_times(plans, order, hours):
    rows = np.arange(plans.shape[0])
    load = np.zeros((plans.shape[0], 2))
    done = np.zeros(plans.shape)
    for j in order:
        load[rows, plans[:, j]] += hours[j]
        done[:, j] = load[rows, plans[:, j]]
    return done


def energization_cost(plans, order, weights, hours, parent):
    done = completion_times(plans, order, hours)
    live = np.zeros_like(done)
    for j in topological_order(parent):
        live[:, j] = done[:, j] if parent[j] == -1 else np.maximum(done[:, j], live[:, parent[j]])
    return live @ weights


def pairwise_cost(plans, order, weights, hours):
    n = len(weights)
    cost = np.full(plans.shape[0], float(np.dot(weights, hours)))
    for a in range(n):
        for b in range(a + 1, n):
            i, j = order[a], order[b]
            cost += weights[j] * hours[i] * (plans[:, i] == plans[:, j])
    return cost


def list_schedule(order, hours):
    load = [0.0, 0.0]
    plan = np.zeros(len(order), dtype=np.int8)
    for j in order:
        crew = 0 if load[0] <= load[1] else 1
        plan[j] = crew
        load[crew] += hours[j]
    return plan


def level_by_level(hours, level):
    signs = np.zeros(len(hours))
    carry = 0.0
    for k in np.unique(level):
        members = np.where(level == k)[0]
        best = None
        for trial in itertools.product([1.0, -1.0], repeat=len(members)):
            balance = carry + float(np.dot(hours[members], trial))
            if best is None or abs(balance) < abs(best[0]):
                best = (balance, trial)
        carry = best[0]
        signs[members] = best[1]
    return ((1 - signs) / 2).astype(np.int8)


def priority_dispatch(priority, weights, hours, parent, crews=2):
    n = len(weights)
    started, finish = set(), {}
    free_at = [0.0] * crews
    t = 0.0
    while len(started) < n:
        done = {j for j, f in finish.items() if f <= t + 1e-12}
        active = set(started) - done
        for c in range(crews):
            if free_at[c] > t + 1e-12:
                continue
            ready = [j for j in range(n) if j not in started
                     and (parent[j] == -1 or parent[j] in done or parent[j] in active)]
            if not ready:
                continue
            j = max(ready, key=lambda k: priority[k])
            started.add(j)
            active.add(j)
            finish[j] = t + hours[j]
            free_at[c] = finish[j]
        t = min(f for f in finish.values() if f > t + 1e-12)
    live = np.zeros(n)
    for j in topological_order(parent):
        live[j] = finish[j] if parent[j] == -1 else max(finish[j], live[parent[j]])
    return float(np.dot(weights, live))


def exact_optimum(weights, hours, parent, crews=2, time_limit=120):
    n = len(weights)
    horizon = int(hours.sum())
    length = hours.astype(int)
    x = lambda l, t: l * horizon + (t - 1)
    u = lambda l, t: n * horizon + l * horizon + (t - 1)
    size = 2 * n * horizon
    objective = np.zeros(size)
    lower, upper = np.zeros(size), np.ones(size)
    integrality = np.zeros(size)
    integrality[: n * horizon] = 1
    for l in range(n):
        for t in range(1, horizon + 1):
            if t < length[l]:
                upper[x(l, t)] = 0
            objective[u(l, t)] = -weights[l]
    rows = n + horizon + 2 * n * horizon
    matrix = lil_matrix((rows, size))
    low, high = np.full(rows, -np.inf), np.zeros(rows)
    r = 0
    for l in range(n):
        for t in range(1, horizon + 1):
            matrix[r, x(l, t)] = 1
        low[r], high[r] = 1, 1
        r += 1
    for s in range(1, horizon + 1):
        for l in range(n):
            for t in range(s, min(horizon, s + length[l] - 1) + 1):
                matrix[r, x(l, t)] = 1
        high[r] = crews
        r += 1
    for l in range(n):
        for t in range(1, horizon + 1):
            matrix[r, u(l, t)] = 1
            for tau in range(1, t + 1):
                matrix[r, x(l, tau)] = -1
            r += 1
            if parent[l] != -1:
                matrix[r, u(l, t)] = 1
                matrix[r, u(parent[l], t)] = -1
            r += 1
    result = milp(objective, constraints=LinearConstraint(matrix.tocsr(), low, high),
                  integrality=integrality, bounds=Bounds(lower, upper), options={"time_limit": time_limit})
    if result.x is None:
        return None, False
    cost = sum(weights[l] * (horizon + 1 - sum(result.x[u(l, t)] for t in range(1, horizon + 1)))
               for l in range(n))
    return cost, result.status == 0
