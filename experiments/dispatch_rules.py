import numpy as np

from briskit.feeder import all_plans, plan_index, sample_instance, upstream_parents
from briskit.scheduling import energization_cost, list_schedule, priority_dispatch, single_crew_order

SEED = 20261010


def main(instances=300):
    rng = np.random.default_rng(SEED)
    parent = upstream_parents()
    n = len(parent)
    plans = all_plans(n)
    table = {"FirstEnergy rule": [], "EEI rule": [], "conversion algorithm": [], "best plan in the encoding": []}
    for _ in range(instances):
        w, p = sample_instance(rng, n)
        order = single_crew_order(w, p, parent)
        true = energization_cost(plans, order, w, p, parent)
        table["FirstEnergy rule"].append(priority_dispatch(w, w, p, parent))
        table["EEI rule"].append(priority_dispatch(w / p, w, p, parent))
        table["conversion algorithm"].append(true[plan_index(list_schedule(order, p))])
        table["best plan in the encoding"].append(true.min())
    best = np.array(table["best plan in the encoding"])
    print(f"IEEE 13 node feeder, 12 repairs, two crews, {instances} instances")
    print("customer-hours without power, relative to the best plan in the encoding")
    for name, values in table.items():
        gap = np.array(values) / best - 1
        print(f"   {name:28s} mean +{100 * gap.mean():6.2f}%   median +{100 * np.median(gap):6.2f}%"
              f"   worst +{100 * gap.max():6.2f}%")
    for name in ("FirstEnergy rule", "EEI rule", "conversion algorithm"):
        total = np.array(table[name]).sum()
        print(f"   customer-hours saved by the best plan against the {name}: {100 * (1 - best.sum() / total):.1f}%")


if __name__ == "__main__":
    main()
