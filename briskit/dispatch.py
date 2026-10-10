import numpy as np


def lines_below_each(parent):
    below = [[] for _ in parent]
    for line, above in enumerate(parent):
        if above != -1:
            below[above].append(line)
    return below


def subtree_sums(line, weights, hours, below):
    sums = [(weights[line], hours[line])]
    for child in below[line]:
        under = subtree_sums(child, weights, hours, below)
        sums = sums + [(w + wc, p + pc) for w, p in sums for wc, pc in under]
    return sums


def rho_factors(weights, hours, parent):
    below = lines_below_each(parent)
    return np.array([max(w / p for w, p in subtree_sums(line, weights, hours, below))
                     for line in range(len(parent))])


def energised_times(completed, parent):
    live = np.zeros(len(parent))
    waiting = set(range(len(parent)))
    while waiting:
        for line in sorted(waiting):
            if parent[line] == -1 or parent[line] not in waiting:
                above = 0.0 if parent[line] == -1 else live[parent[line]]
                live[line] = max(completed[line], above)
                waiting.remove(line)
    return live


def dispatch_times(priority, hours, parent, crews=2, candidates="selected"):
    if candidates not in ("selected", "energised"):
        raise ValueError(candidates)
    n = len(parent)
    finish = {}
    free_at = [0.0] * crews
    t = 0.0
    while len(finish) < n:
        for crew in range(crews):
            if free_at[crew] > t + 1e-12:
                continue
            done = {j for j, f in finish.items() if f <= t + 1e-12}
            active = set(finish) - done

            def energised(line):
                while line != -1:
                    if line not in done:
                        return False
                    line = parent[line]
                return True

            if candidates == "selected":
                allowed = [j for j in range(n) if j not in finish and (parent[j] == -1 or parent[j] in finish)]
            else:
                allowed = [j for j in range(n) if j not in finish
                           and (parent[j] == -1 or energised(parent[j]) or parent[j] in active)]
            if not allowed:
                continue
            line = max(allowed, key=lambda j: priority[j])
            finish[line] = t + hours[line]
            free_at[crew] = finish[line]
        t = min(f for f in finish.values() if f > t + 1e-12)
    completed = np.array([finish[j] for j in range(n)])
    return completed, energised_times(completed, parent)


def restored_share(weights, live, moment):
    return float(weights[live <= moment + 1e-12].sum() / weights.sum())
