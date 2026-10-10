import itertools

import numpy as np

from briskit.baselines import greedy_marginal
from briskit.feeder import all_plans, plan_index, restoration_value, sample_instance, upstream_parents
from briskit.pair_cost import (coupling_matrix, eigen_blocks, fit_coefficients, pair_features, ridge_coefficients,
                               selection_features)
from briskit.scheduling import energization_cost, single_crew_order

SEED = 20261010


def restore_within_budget(rng, count, instances=200, samples=300):
    parent = upstream_parents()
    n = len(parent)
    plans = all_plans(n)
    subsets = np.array(list(itertools.combinations(range(n), count)))
    chosen = np.zeros((len(subsets), n))
    chosen[np.arange(len(subsets))[:, None], subsets] = 1.0
    models = {"fitted cost, single-line terms": selection_features(chosen, pairs=False),
              "fitted cost, single and pair terms": selection_features(chosen)}
    loss = {"largest weights first": [], **{name: [] for name in models}, "greedy by marginal gain": []}
    for _ in range(instances):
        w, _ = sample_instance(rng, n)
        value = restoration_value(chosen, w, parent)
        table = restoration_value(plans, w, parent)
        optimum = value.max()
        largest = np.zeros(n, dtype=int)
        largest[np.argsort(-w)[:count]] = 1
        loss["largest weights first"].append(1 - table[plan_index(largest)] / optimum)
        rows = rng.choice(len(chosen), size=min(samples, len(chosen)), replace=False)
        for name, features in models.items():
            coefficients = ridge_coefficients(features[rows], value[rows])
            loss[name].append(1 - value[np.argmax(features @ coefficients)] / optimum)
        loss["greedy by marginal gain"].append(1 - table[greedy_marginal(table, n, count)] / optimum)
    return loss


def collective_blocks(rng, instances=12, samples=300, counts=(1, 2, 3, 4, 6, 12)):
    parent = upstream_parents()
    n = len(parent)
    plans = all_plans(n)
    features = pair_features(plans)
    spins = 1.0 - 2.0 * plans
    share, gap = {k: [] for k in counts}, {k: [] for k in counts}
    for _ in range(instances):
        w, p = sample_instance(rng, n)
        true = energization_cost(plans, single_crew_order(w, p, parent), w, p, parent)
        rows = rng.choice(true.size, size=samples, replace=False)
        values, vectors = eigen_blocks(coupling_matrix(fit_coefficients(features, true, rows), n))
        for k in counts:
            share[k].append(np.abs(values[:k]).sum() / np.abs(values).sum())
            model = 0.5 * ((spins @ vectors[:, :k]) ** 2 @ values[:k])
            gap[k].append(true[np.argmin(model)] / true.min() - 1)
    return share, gap


def main():
    rng = np.random.default_rng(SEED)
    print("IEEE 13 node feeder, restore a fixed number of 12 damaged lines, 200 instances")
    print("a line counts only if every line upstream of it is restored; value lost against the optimum, mean")
    for count in (4, 6):
        loss = restore_within_budget(rng, count)
        print(f"   restore {count}: " + ";  ".join(f"{name} {100 * np.mean(values):.1f}%" for name, values in loss.items()))
    print("IEEE 13 node feeder, crew assignment, fitted pair cost written as collective blocks, 12 instances")
    share, gap = collective_blocks(rng)
    for k in share:
        print(f"   {k:2d} block(s): {100 * np.mean(share[k]):5.1f}% of the pair cost,"
              f" best plan {100 * np.mean(gap[k]):.2f}% above the best assignment")


if __name__ == "__main__":
    main()
