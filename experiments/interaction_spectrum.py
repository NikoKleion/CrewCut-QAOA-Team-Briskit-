import numpy as np

from briskit.feeder import all_plans, restoration_value, sample_instance, upstream_parents
from briskit.pair_cost import interaction_order_shares, protection_terms
from briskit.scheduling import energization_cost, pairwise_cost, single_crew_order, smith_order

SEED = 20261010


def main(instances=50):
    rng = np.random.default_rng(SEED)
    parent = upstream_parents()
    n = len(parent)
    plans = all_plans(n)
    unprotected = 1.0 - plans
    table = {"crew assignment, independent repairs": [], "crew assignment on the feeder": [],
             "budgeted restoration on the feeder": [], "protection, independent lines": []}
    for _ in range(instances):
        w, p = sample_instance(rng, n)
        table["crew assignment, independent repairs"].append(
            interaction_order_shares(pairwise_cost(plans, smith_order(w, p), w, p)))
        table["crew assignment on the feeder"].append(
            interaction_order_shares(energization_cost(plans, single_crew_order(w, p, parent), w, p, parent)))
        table["budgeted restoration on the feeder"].append(interaction_order_shares(restoration_value(plans, w, parent)))
        single, pair = protection_terms(w, p, rng.uniform(0.05, 0.6, n))
        expected = unprotected @ single + 0.5 * np.einsum("ri,ij,rj->r", unprotected, pair, unprotected)
        table["protection, independent lines"].append(interaction_order_shares(expected))
    print(f"share of the non-constant variance by interaction order, 12 lines, mean over {instances} instances")
    print(f"   {'decision':40s} order 1   order 2   order 3   order 4   order 5+")
    for name, values in table.items():
        print(f"   {name:40s} " + "   ".join(f"{100 * x:5.1f}%" for x in np.mean(values, axis=0)))


if __name__ == "__main__":
    main()
