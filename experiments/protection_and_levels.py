import itertools

import numpy as np

from briskit.feeder import all_plans, plan_index
from briskit.pair_cost import level_vectors
from briskit.scheduling import level_by_level, list_schedule, pairwise_cost, smith_order

SEED = 20261010


def protection_form_error(rng, cases=40, n=8):
    worst = 0.0
    for _ in range(cases):
        hours = rng.integers(1, 11, n).astype(float)
        weights = rng.uniform(0.1, 3.0, n)
        failure = rng.uniform(0.05, 0.7, n)
        ratio = weights / hours
        exposure = failure * hours
        order = np.argsort(-ratio)
        unprotected = rng.integers(0, 2, n)
        exact = 0.0
        for failed in itertools.product([0, 1], repeat=n):
            f = np.array(failed)
            chance = np.prod(np.where(f == 1, failure * unprotected, 1 - failure * unprotected))
            t = cost = 0.0
            for j in order:
                if f[j]:
                    t += hours[j]
                    cost += weights[j] * t
            exact += chance * cost
        gaps, vectors = level_vectors(exposure, ratio)
        form = float(np.sum(unprotected * (failure * weights * hours - 0.5 * exposure ** 2 * ratio)))
        form += 0.5 * float(gaps @ (vectors @ unprotected) ** 2)
        worst = max(worst, abs(exact - form))
    return worst


def leveled_jobs(rng, levels, per_level):
    hours = rng.integers(1, 11, levels * per_level).astype(float)
    ratios = np.sort(rng.uniform(0.2, 3.0, levels))[::-1]
    level = np.repeat(np.arange(levels), per_level)
    return hours, ratios[level] * hours, level


def level_solver_gaps(rng, levels, per_level, instances):
    plans = all_plans(levels * per_level)
    rule, solver = [], []
    for _ in range(instances):
        hours, weights, level = leveled_jobs(rng, levels, per_level)
        order = smith_order(weights, hours)
        cost = pairwise_cost(plans, order, weights, hours)
        best = cost.min()
        rule.append(cost[plan_index(list_schedule(order, hours))] / best - 1)
        solver.append(cost[plan_index(level_by_level(hours, level))] / best - 1)
    return np.array(rule), np.array(solver)


def main(instances=20):
    rng = np.random.default_rng(SEED)
    print("protection: expected cost over every failure set against the sum over levels, 40 cases of 8 lines")
    print(f"   largest difference {protection_form_error(rng):.1e}")
    print(f"18 repairs in urgency levels, {instances} instances per row, percent above the optimum, mean (worst)")
    print("   levels x repairs   classical rule         level by level")
    for levels, per_level in ((2, 9), (3, 6), (6, 3)):
        rule, solver = level_solver_gaps(rng, levels, per_level, instances)
        print(f"   {levels} x {per_level}              {100 * rule.mean():6.3f}% ({100 * rule.max():6.3f}%)"
              f"     {100 * solver.mean():6.3f}% ({100 * solver.max():6.3f}%)"
              f"   optimal in {int(np.sum(solver < 1e-12))} of {instances}")


if __name__ == "__main__":
    main()
