import itertools

import numpy as np

from briskit.feeder import all_plans, plan_index, random_tree_parents, sample_instance, upstream_parents
from briskit.qaoa import expected_best_of, train_qaoa
from briskit.scheduling import (brute_force_single_crew, energization_cost, exact_optimum, list_schedule,
                                pairwise_cost, single_crew_order, smith_order, weighted_completion_cost)

SEED = 20261010


def single_crew_order_error(rng, cases=40, n=7):
    worst = 0.0
    for _ in range(cases):
        parent = random_tree_parents(rng, n)
        w, p = sample_instance(rng, n)
        order = single_crew_order(w, p, parent)
        worst = max(worst, abs(weighted_completion_cost(order, w, p) - brute_force_single_crew(w, p, parent)))
    return worst


def pair_form_error(rng, cases=20, n=7):
    plans = all_plans(n)
    worst = 0.0
    for _ in range(cases):
        w, p = sample_instance(rng, n)
        cost = pairwise_cost(plans, smith_order(w, p), w, p)
        for row in range(0, 2 ** n, 5):
            crews = [[j for j in range(n) if plans[row, j] == c] for c in (0, 1)]
            exact = sum(min((weighted_completion_cost(perm, w, p) for perm in itertools.permutations(crew)),
                            default=0.0) for crew in crews)
            worst = max(worst, abs(exact - cost[row]))
    return worst


def feeder_instances(rng, parent, count):
    plans = all_plans(len(parent))
    instances = []
    for _ in range(count):
        w, p = sample_instance(rng, len(parent))
        order = single_crew_order(w, p, parent)
        true = energization_cost(plans, order, w, p, parent)
        pair = pairwise_cost(plans, order, w, p)
        assert np.all(pair <= true + 1e-9)
        listed = true[plan_index(list_schedule(order, p))]
        instances.append((w, p, true, pair, listed))
    return instances


def main(instances=300, exact=12, sampled=20):
    rng = np.random.default_rng(SEED)

    print("single crew order against brute force, 40 random trees of 7 repairs")
    print(f"   largest difference {single_crew_order_error(rng):.2e}")

    print("pair cost for independent repairs against brute force over every order on each crew, 20 cases of 7 repairs")
    print(f"   largest difference {pair_form_error(rng):.2e}")

    parent = upstream_parents()
    cases = feeder_instances(rng, parent, instances)
    listed = np.array([c[4] / c[2].min() - 1 for c in cases])
    closed = np.array([c[2][np.argmin(c[3])] / c[2].min() - 1 for c in cases])
    better = np.array([np.mean(c[2] < c[4] - 1e-9) for c in cases])
    print(f"IEEE 13 node feeder, 12 repairs, two crews, {instances} instances, percent above the best assignment")
    print(f"   conversion algorithm                 mean {100 * listed.mean():.2f}%  median {100 * np.median(listed):.2f}%"
          f"  worst {100 * listed.max():.2f}%")
    print(f"   conversion algorithm already best    {100 * np.mean(listed < 1e-9):.0f}% of instances")
    print(f"   best plan of the closed-form pair cost  mean {100 * closed.mean():.2f}%  worst {100 * closed.max():.2f}%")
    print(f"   share of assignments better than the conversion algorithm  {100 * better.mean():.2f}%")

    print(f"exact optimum over all schedules, first {exact} instances")
    rows = []
    for k in range(exact):
        w, p, true, pair, listed_cost = cases[k]
        optimum, proven = exact_optimum(w, p, parent)
        if optimum is None:
            continue
        rows.append((listed_cost / optimum - 1, true.min() / optimum - 1))
        print(f"   instance {k:2d}  optimum {optimum:8.3f}  conversion algorithm {100 * rows[-1][0]:+6.2f}%"
              f"  best assignment {100 * rows[-1][1]:+6.2f}%  proven {proven}", flush=True)
    if rows:
        rows = np.array(rows)
        print(f"   mean  conversion algorithm +{100 * rows[:, 0].mean():.2f}%  best assignment +{100 * rows[:, 1].mean():.2f}%")

    print(f"QAOA on the closed-form pair cost, {sampled} instances, no noise")
    for depth in (1, 2, 3):
        hit, shots, hit_random, shots_random = [], [], [], []
        for w, p, true, pair, listed_cost in cases[:sampled]:
            _, prob = train_qaoa(pair, depth, rng)
            best = true.min()
            uniform = np.full(true.size, 1 / true.size)
            optimal = true <= best * (1 + 1e-12)
            hit.append(prob[optimal].sum())
            hit_random.append(uniform[optimal].sum())
            shots.append(expected_best_of(prob, true, 100) / best - 1)
            shots_random.append(expected_best_of(uniform, true, 100) / best - 1)
        print(f"   depth {depth}  chance of a best assignment {100 * np.mean(hit):.2f}% (random {100 * np.mean(hit_random):.3f}%)"
              f"  best of 100 shots +{100 * np.mean(shots):.2f}% (random +{100 * np.mean(shots_random):.2f}%)", flush=True)


if __name__ == "__main__":
    main()
