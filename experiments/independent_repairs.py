import numpy as np

from briskit.feeder import all_plans, plan_index, sample_instance
from briskit.qaoa import expected_best_of, train_qaoa
from briskit.scheduling import list_schedule, pairwise_cost, smith_order

SEED = 20261010


def study(n, instances, depths, rng):
    plans = all_plans(n)
    table = {}
    add = lambda name, value: table.setdefault(name, []).append(value)
    for r in range(instances):
        w, p = sample_instance(rng, n)
        order = smith_order(w, p)
        cost = pairwise_cost(plans, order, w, p)
        best = cost.min()
        gap = lambda value: value / best - 1
        uniform = np.full(cost.size, 1 / cost.size)
        add("classical rule (urgency per hour, first free crew)", gap(cost[plan_index(list_schedule(order, p))]))
        for shots in (100, 1000):
            add(f"random plans, best of {shots}", gap(expected_best_of(uniform, cost, shots)))
        for depth in depths:
            _, prob = train_qaoa(cost, depth, rng, starts=3)
            add(f"QAOA depth {depth}, chance one shot is optimal (random {2 / cost.size:.5f})",
                float(prob[cost <= best * (1 + 1e-12)].sum()))
            for shots in (100, 1000):
                add(f"QAOA depth {depth}, best of {shots}", gap(expected_best_of(prob, cost, shots)))
        print(f"   {n} repairs: instance {r + 1} of {instances}", flush=True)
    print(f"{n} independent repairs, {2 ** n} plans, {instances} instances, mean percent above the optimum")
    for name, values in table.items():
        if "chance" in name:
            print(f"   {name:62s} {np.mean(values):9.5f}")
        else:
            print(f"   {name:62s} {100 * np.mean(values):8.3f}%")


def main(small=10, large=3):
    rng = np.random.default_rng(SEED)
    study(12, small, (1, 2, 3), rng)
    study(18, large, (1, 2), rng)


if __name__ == "__main__":
    main()
