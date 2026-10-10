import numpy as np

from briskit.baselines import greedy_marginal, matched_thermal
from briskit.feeder import all_plans, plan_index, random_tree_parents, restoration_value, sample_instance
from briskit.pair_cost import ridge_coefficients, selection_features
from briskit.qaoa import expected_best_of, train_qaoa

SEED = 20261010


def main(n=18, count=8, instances=4, samples=600, shots=100):
    rng = np.random.default_rng(SEED)
    plans = all_plans(n)
    size = plans.sum(axis=1)
    feasible = np.where(size == count)[0]
    features = selection_features(plans[feasible])
    upper = np.triu_indices(n, 1)
    betas = np.geomspace(0.01, 30, 60)
    table = {}
    add = lambda name, value: table.setdefault(name, []).append(value)
    for r in range(instances):
        parent = random_tree_parents(rng, n)
        w, _ = sample_instance(rng, n)
        value = restoration_value(plans, w, parent)
        allowed = value[feasible]
        optimum = allowed.max()
        lost = lambda v: 1 - v / optimum

        largest = np.zeros(n, dtype=int)
        largest[np.argsort(-w)[:count]] = 1
        add("largest weights first", lost(value[plan_index(largest)]))
        add("greedy by marginal gain", lost(value[greedy_marginal(value, n, count)]))

        rows = rng.choice(len(feasible), size=samples, replace=False)
        coefficients = ridge_coefficients(features[rows], allowed[rows])
        fitted = features @ coefficients
        ranked = np.argsort(-fitted)
        add("fitted pair cost, its best set", lost(allowed[ranked[0]]))
        add(f"fitted pair cost, its {shots} best sets", lost(allowed[ranked[:shots]].max()))
        add(f"best of the {samples} fitting sets", lost(allowed[rows].max()))

        uniform = np.full(len(feasible), 1 / len(feasible))
        add(f"random feasible sets, best of {shots}", lost(-expected_best_of(uniform, -allowed, shots)))

        fitted_all = coefficients[0] + plans @ coefficients[1:n + 1]
        for c, i, j in zip(coefficients[n + 1:], upper[0], upper[1]):
            fitted_all += c * plans[:, i] * plans[:, j]
        cost = -fitted_all + fitted.std() * (size - count) ** 2
        for depth in (1, 2):
            _, prob = train_qaoa(cost, depth, rng, starts=2)
            inside = prob[feasible]
            add(f"QAOA depth {depth}: share of shots with the right count", inside.sum())
            inside = inside / inside.sum()
            add(f"QAOA depth {depth}, best of {shots} feasible shots", lost(-expected_best_of(inside, -allowed, shots)))
            if depth == 2:
                thermal = matched_thermal(-fitted, betas, -float(inside @ fitted))
                add(f"thermal sampler at the same mean fitted value, best of {shots}",
                    lost(-expected_best_of(thermal, -allowed, shots)))
        print(f"   instance {r + 1} of {instances}", flush=True)
    print(f"restore {count} of {n} lines on random trees, {len(feasible)} feasible sets, {instances} instances")
    print("value lost against the optimum, mean")
    for name, values in table.items():
        digits = 1 if "share" in name else 2
        print(f"   {name:62s} {100 * np.mean(values):6.{digits}f}%")


if __name__ == "__main__":
    main()
