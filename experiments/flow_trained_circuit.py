from briskit import benchmark
from briskit.flow import restoration, stored_instance, stored_value_table
from briskit.qaoa import train_qaoa
from briskit.solvers import cost_at_bound, fit_on_sample, fitted_merit
from briskit.trained import read_records, write_records
from experiments.flow_budget_experiment import SAMPLES
from experiments.trained_circuits import OUTPUT, ROOT, STARTS, code_version, without_code

EXPERIMENT = "E8"
NAME = "random_geometric_12_0"
COUNT = 6
SEED = 0
DEPTHS = (1, 2)


def flow_instance():
    stored = stored_instance(NAME)
    table, _ = stored_value_table(stored)
    return restoration(stored, table, COUNT), table


def train_record(instance, table, seed, depth):
    method = f"fitted cost, QAOA depth {depth}"
    rng = benchmark.stream(seed, instance.name, method)
    samples = SAMPLES[instance.size]
    coefficients, rows, plans = fit_on_sample(instance, samples, rng)
    cost, penalty = cost_at_bound(fitted_merit(coefficients, plans, instance.sense), rows, instance)
    angles, probability = train_qaoa(cost, depth, rng, starts=STARTS)
    record = {"experiment": EXPERIMENT, "instance": instance.name, "method": method, "seed": seed,
              "samples": samples, "starts": STARTS, "depth": depth, "size": instance.size, "sense": instance.sense,
              "count": instance.count, "values": [float(v) for v in table],
              "coefficients": [float(c) for c in coefficients], "penalty": penalty,
              "angles": [float(a) for a in angles]}
    return record, rows, probability, rng


def main():
    instance, table = flow_instance()
    optimum = instance.optimum()
    records = []
    print(f"{instance.name}, pair cost fitted on {SAMPLES[instance.size]} plans, seed {SEED}")
    print("   depth   right count   optimum, of the right count   mean regret, of the right count")
    for depth in DEPTHS:
        record, rows, probability, _ = train_record(instance, table, SEED, depth)
        inside = probability[rows] / probability[rows].sum()
        best = table[rows] >= optimum * (1 - 1e-12)
        print(f"   {depth}       {100 * probability[rows].sum():6.1f}%       {100 * inside[best].sum():6.2f}%"
              f"                      {100 * float(inside @ (1 - table[rows] / optimum)):6.2f}%")
        records.append(record)
    saved = read_records(OUTPUT) if OUTPUT.exists() else []
    own = [record for record in saved if record["experiment"] == EXPERIMENT]
    others = [record for record in saved if record["experiment"] != EXPERIMENT]
    if without_code(own) != records:
        write_records(OUTPUT, others + [dict(record, code=code_version()) for record in records])
    print(f"   records for depths 1 and 2 are in {OUTPUT.relative_to(ROOT).as_posix()}, scored by the value table"
          " stored in each record")


if __name__ == "__main__":
    main()
