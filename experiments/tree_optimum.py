import numpy as np

from briskit import benchmark
from briskit.feeder import random_tree_parents, sample_instance, upstream_parents
from briskit.instances import restoration
from briskit.tree_knapsack import best_closed_set, tree_optimum

SEED = 20261010
SCRIPT = "experiments/tree_optimum.py"


def main(instances=30, large=2500):
    print(f"IEEE 13 node feeder, {instances} instances, exact tree method through the harness, one plan returned")
    for count in (4, 6):
        rows = []
        for k in range(instances):
            weights, _ = sample_instance(np.random.default_rng([SEED, k]), 12)
            instance = restoration(f"restore {count} of 12, instance {k}", weights, upstream_parents(), count)
            rows += benchmark.run("tree optimum", [instance], {"exact tree method": tree_optimum}, [1], [k], SCRIPT)
        exact = sum(row["regret"] < 1e-12 for row in rows)
        scored = sorted({row["evaluations"] for row in rows})
        print(f"   restore {count} of 12: optimum on {exact} of {len(rows)} instances, plans scored per run {scored}")
    rng = np.random.default_rng(SEED)
    parent = random_tree_parents(rng, large)
    weights, hours = sample_instance(rng, large)
    print(f"random tree of {large} lines, no value table possible")
    for count in (large // 10, large // 2):
        value, chosen = best_closed_set(weights, parent, count)
        print(f"   restore {count:4d}: best value {value:10.3f} of {weights.sum():.3f}, lines chosen {int(chosen.sum())},"
              f" table entries {large * (count + 1)}")
    budget = int(hours.sum() // 10)
    value, chosen = best_closed_set(weights, parent, budget, hours.astype(int))
    print(f"   {budget} repair hours of {int(hours.sum())}: best value {value:10.3f}, lines chosen {int(chosen.sum())},"
          f" hours used {int(hours[chosen == 1].sum())}")


if __name__ == "__main__":
    main()
