import numpy as np

from briskit.baselines import bit_flip_descent
from briskit.feeder import all_plans, plan_index, random_tree_parents, sample_instance, upstream_parents
from briskit.pair_cost import fit_pair_cost, pair_features
from briskit.qaoa import expected_best_of, train_qaoa
from briskit.scheduling import energization_cost, list_schedule, pairwise_cost, single_crew_order

SEED = 20261010


def study(label, n, parents, instances, samples, depths, rng):
    plans = all_plans(n)
    features = pair_features(plans)
    table = {}
    add = lambda name, value: table.setdefault(name, []).append(value)
    for r in range(instances):
        parent = parents(rng)
        w, p = sample_instance(rng, n)
        order = single_crew_order(w, p, parent)
        true = energization_cost(plans, order, w, p, parent)
        best = true.min()
        gap = lambda value: value / best - 1
        uniform = np.full(true.size, 1 / true.size)
        listed = plan_index(list_schedule(order, p))
        add("conversion algorithm", gap(true[listed]))
        descent, evaluations = bit_flip_descent(true, listed, n)
        add("bit-flip descent from the conversion algorithm", gap(descent))
        add("evaluations used by the descent", evaluations)
        for shots in (100, 1000):
            add(f"random plans, best of {shots}", gap(expected_best_of(uniform, true, shots)))
        closed = pairwise_cost(plans, order, w, p)
        add("closed-form pair cost, its best plan", gap(true[np.argmin(closed)]))
        models = {"closed-form pair cost": closed}
        for m in samples:
            rows = rng.choice(true.size, size=m, replace=False)
            fit = fit_pair_cost(features, true, rows)
            models[f"pair cost fitted on {m} plans"] = fit
            add(f"pair cost fitted on {m} plans, its best plan", gap(true[np.argmin(fit)]))
            add(f"best of the {m} fitting plans", gap(true[rows].min()))
        for name, model in models.items():
            for depth in depths:
                _, prob = train_qaoa(model, depth, rng, starts=2)
                for shots in (100, 1000):
                    add(f"QAOA depth {depth}, {name}, best of {shots}", gap(expected_best_of(prob, true, shots)))
        print(f"   {label}: instance {r + 1} of {instances}", flush=True)
    print(f"{label}, {n} repairs, {instances} instances, mean percent above the best assignment")
    for name, values in table.items():
        if name.startswith("evaluations"):
            print(f"   {name:62s} {np.mean(values):8.0f}")
        else:
            print(f"   {name:62s} {100 * np.mean(values):8.2f}%")


def main(feeder_instances=20, tree_instances=3):
    rng = np.random.default_rng(SEED)
    feeder = upstream_parents()
    study("IEEE 13 node feeder", 12, lambda g: feeder, feeder_instances, (150, 300), (1, 2, 3), rng)
    study("random trees", 18, lambda g: random_tree_parents(g, 18), tree_instances, (500, 1500), (1, 2), rng)


if __name__ == "__main__":
    main()
