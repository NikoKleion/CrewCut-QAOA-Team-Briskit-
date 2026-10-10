import numpy as np

from briskit.feeder import all_plans, sample_instance
from briskit.pair_cost import level_count
from briskit.scheduling import pairwise_cost

SEED = 20261010


def main(n=12, instances=40):
    rng = np.random.default_rng(SEED)
    plans = all_plans(n)
    print(f"{n} independent repairs, {instances} instances per row")
    print("   step   levels before -> after   rounded cost / true cost over all plans   plan found, above the optimum (worst)")
    for step in (0.10, 0.25, 0.50, 1.00):
        before, after, highest, lowest, worst = [], [], 0.0, np.inf, 0.0
        for _ in range(instances):
            w, p = sample_instance(rng, n)
            ratio = w / p
            grid = np.ceil(np.log(ratio / ratio.min()) / np.log(1 + step) - 1e-12)
            rounded_ratio = ratio.min() * (1 + step) ** grid
            order = sorted(range(n), key=lambda j: (-ratio[j], j))
            rounded_order = sorted(range(n), key=lambda j: (-rounded_ratio[j], -ratio[j], j))
            true = pairwise_cost(plans, order, w, p)
            rounded = pairwise_cost(plans, rounded_order, rounded_ratio * p, p)
            quotient = rounded / true
            lowest = min(lowest, quotient.min())
            highest = max(highest, quotient.max())
            worst = max(worst, true[np.argmin(rounded)] / true.min() - 1)
            before.append(level_count(ratio))
            after.append(level_count(rounded_ratio))
        print(f"   {step:4.2f}   {np.mean(before):5.1f} -> {np.mean(after):4.1f}"
              f"            {lowest:.4f} to {highest:.4f} (bound {1 + step:.2f})"
              f"             {100 * worst:.2f}% (bound {100 * step:.0f}%)")


if __name__ == "__main__":
    main()
