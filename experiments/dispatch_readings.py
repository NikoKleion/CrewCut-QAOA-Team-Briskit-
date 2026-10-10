import numpy as np

from briskit.dispatch import dispatch_times, restored_share, rho_factors
from briskit.feeder import all_plans, sample_instance, upstream_parents
from briskit.scheduling import energization_cost, single_crew_order

SEED = 20261010


def main(instances=300):
    rng = np.random.default_rng(SEED)
    parent = upstream_parents()
    plans = all_plans(len(parent))
    rules = {"rho factor": lambda w, p: rho_factors(w, p, parent), "FirstEnergy": lambda w, p: w, "EEI": lambda w, p: w / p}
    gaps = {name: {"selected": [], "energised": []} for name in rules}
    differ = {name: 0 for name in rules}
    halfway = {name: [] for name in rules}
    for _ in range(instances):
        weights, hours = sample_instance(rng, len(parent))
        best = energization_cost(plans, single_crew_order(weights, hours, parent), weights, hours, parent).min()
        live = {}
        for name, rule in rules.items():
            priority = rule(weights, hours)
            cost = {}
            for reading in ("selected", "energised"):
                times = dispatch_times(priority, hours, parent, 2, reading)[1]
                cost[reading] = float(weights @ times)
                gaps[name][reading].append(cost[reading] / best - 1)
                if reading == "selected":
                    live[name] = times
            differ[name] += abs(cost["selected"] - cost["energised"]) > 1e-9
        half = live["rho factor"].max() / 2
        for name in rules:
            halfway[name].append(restored_share(weights, live[name], half))
    print(f"IEEE 13 node feeder, 12 repairs, two crews, {instances} instances")
    print("candidate lines: selected = the line upstream has been started;"
          " energised = its node is live or the line upstream is under repair")
    print("   rule          schedules differ   above the best assignment, selected   energised")
    for name in rules:
        print(f"   {name:12s}  {differ[name]:4d} of {instances}        {100 * np.mean(gaps[name]['selected']):6.2f}%"
              f"                          {100 * np.mean(gaps[name]['energised']):6.2f}%")
    print("share of weight restored at half the rho rule's time to full restoration, candidates selected")
    for name in rules:
        print(f"   {name:12s}  {100 * np.mean(halfway[name]):5.1f}%")


if __name__ == "__main__":
    main()
