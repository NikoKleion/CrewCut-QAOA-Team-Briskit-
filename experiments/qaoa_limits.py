import numpy as np

from briskit.baselines import descent_endpoints, karmarkar_karp, longest_first_difference
from briskit.feeder import all_plans, sample_instance
from briskit.hardware import unaffected_fraction
from briskit.pair_cost import pair_coefficients
from briskit.qaoa import expected_best_of, train_qaoa
from briskit.scheduling import pairwise_cost, smith_order

SEED = 20261010
STARTS = 2
ITERATIONS = 40


def trained(cost, depth, rng):
    return train_qaoa(cost, depth, rng, starts=STARTS, maxiter=ITERATIONS)[1]


def independent_cost(rng, n, one_level):
    if one_level:
        p = rng.uniform(1.0, 10.0, n)
        w = p.copy()
    else:
        w, p = sample_instance(rng, n)
    return w, p, pairwise_cost(all_plans(n), smith_order(w, p), w, p)


def number_partitioning(rng, n=14, instances=6):
    spins = 1.0 - 2.0 * all_plans(n)
    table = {}
    add = lambda name, value: table.setdefault(name, []).append(value)
    for _ in range(instances):
        hours = rng.uniform(1.0, 10.0, n)
        difference = np.abs(spins @ hours)
        add("optimum", difference.min())
        add("longest first, to the less loaded crew", longest_first_difference(hours))
        add("Karmarkar-Karp differencing", abs(float(karmarkar_karp(hours) @ hours)))
        uniform = np.full(difference.size, 1 / difference.size)
        add("random plans, best of 100", expected_best_of(uniform, difference, 100))
        add("random plans, best of 1000", expected_best_of(uniform, difference, 1000))
        for depth in (1, 2, 3):
            prob = trained(difference ** 2, depth, rng)
            add(f"QAOA depth {depth}, best of 100", expected_best_of(prob, difference, 100))
            add(f"QAOA depth {depth}, best of 1000", expected_best_of(prob, difference, 1000))
    print(f"one urgency level, {n} repairs, {instances} instances: difference in hours between the two crews, mean")
    for name, values in table.items():
        print(f"   {name:42s} {np.mean(values):9.4f}")


def local_search_starts(rng, sizes=(12, 14), instances=6):
    print("steepest bit-flip descent from a starting plan: chance it ends at the best plan")
    for n in sizes:
        table = {"random start": [], "QAOA depth 1 start": [], "QAOA depth 2 start": [], "QAOA depth 3 start": []}
        for _ in range(instances):
            w, p, cost = independent_cost(rng, n, False)
            end = descent_endpoints(cost, n)
            reaches = cost[end] <= cost.min() * (1 + 1e-12)
            table["random start"].append(reaches.mean())
            for depth in (1, 2, 3):
                table[f"QAOA depth {depth} start"].append(float(trained(cost, depth, rng)[reaches].sum()))
        print(f"   {n} repairs: " + ";  ".join(f"{name} {100 * np.mean(values):.1f}%" for name, values in table.items()))


def sparse_cost_layer(rng, n=14, instances=5):
    plans = all_plans(n)
    table = {}
    for _ in range(instances):
        w, p = sample_instance(rng, n)
        true = pairwise_cost(plans, smith_order(w, p), w, p)
        pairs = sorted(pair_coefficients(w, p).items(), key=lambda item: -item[1])
        uniform = np.full(true.size, 1 / true.size)
        for keep in (1.0, 0.75, 0.5, 0.25):
            kept = pairs[: int(round(keep * len(pairs)))]
            pruned = np.zeros(true.size)
            for (i, j), c in kept:
                pruned += c * (plans[:, i] == plans[:, j])
            for depth in (1, 2):
                prob = trained(pruned, depth, rng)
                gates = len(kept) * depth
                clean = unaffected_fraction(gates)
                noisy = clean * prob + (1 - clean) * uniform
                table.setdefault((depth, keep), []).append(
                    (expected_best_of(prob, true, 100) / true.min() - 1,
                     expected_best_of(noisy, true, 100) / true.min() - 1, clean, gates))
        table.setdefault("random", []).append(expected_best_of(uniform, true, 100) / true.min() - 1)
    print(f"{n} repairs, {instances} instances, best of 100 shots, percent above the optimum")
    print("each shot is unaffected or replaced by a random plan; keep is the share of pair terms kept, largest first")
    print(f"   random plans {100 * np.mean(table['random']):.3f}%")
    for depth in (1, 2):
        for keep in (1.0, 0.75, 0.5, 0.25):
            rows = np.array(table[(depth, keep)])
            print(f"   depth {depth}, keep {keep:4.2f}: {int(rows[0, 3]):3d} gates, shots unaffected {100 * rows[0, 2]:4.1f}%,"
                  f" without noise {100 * rows[:, 0].mean():.3f}%, with noise {100 * rows[:, 1].mean():.3f}%")


def optimal_plan_probability(rng, sizes=(8, 10, 12, 14), instances=6):
    print("chance one shot is the best plan, as a multiple of random sampling, median")
    for one_level in (False, True):
        table = {depth: [] for depth in (1, 2, 3)}
        for n in sizes:
            ratios = {depth: [] for depth in (1, 2, 3)}
            for _ in range(instances):
                w, p, cost = independent_cost(rng, n, one_level)
                optimal = cost <= cost.min() * (1 + 1e-12)
                for depth in (1, 2, 3):
                    ratios[depth].append(trained(cost, depth, rng)[optimal].sum() / (optimal.sum() / cost.size))
            for depth in (1, 2, 3):
                table[depth].append(np.median(ratios[depth]))
        print(f"   {'one urgency level' if one_level else 'distinct urgency ratios'}, sizes {sizes}")
        for depth in (1, 2, 3):
            slope = np.polyfit(sizes, np.log2(table[depth]), 1)[0]
            print(f"     depth {depth}: " + "  ".join(f"{value:6.1f}" for value in table[depth])
                  + f"    growth per added repair x{2 ** slope:.3f}")


def main():
    rng = np.random.default_rng(SEED)
    for part in (number_partitioning, local_search_starts, sparse_cost_layer, optimal_plan_probability):
        part(rng)
        print(flush=True)


if __name__ == "__main__":
    main()
