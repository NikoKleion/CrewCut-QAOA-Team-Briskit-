import subprocess
from pathlib import Path

import numpy as np

from briskit import benchmark
from briskit.feeder import sample_instance, upstream_parents
from briskit.instances import restoration
from briskit.qaoa import train_qaoa
from briskit.solvers import fit_on_sample, fitted_merit, penalised_cost
from briskit.trained import read_records, write_records

SEED = 20261010
SAMPLES = 150
STARTS = 2
INSTANCE = 0
EXPERIMENT = "budget experiment"
ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "qollab" / "trained_circuits.json"


def budget_instance(k):
    weights, _ = sample_instance(np.random.default_rng([SEED, k]), 12)
    return restoration(f"restore 6 of 12, instance {k}", weights, upstream_parents(), 6)


def train_record(instance, seed, depth):
    method = f"fitted cost, QAOA depth {depth}"
    rng = benchmark.stream(seed, instance.name, method)
    coefficients, rows, plans = fit_on_sample(instance, SAMPLES, rng)
    cost, penalty = penalised_cost(fitted_merit(coefficients, plans, instance.sense), rows, instance)
    angles, probability = train_qaoa(cost, depth, rng, starts=STARTS)
    record = {"experiment": EXPERIMENT, "instance": instance.name, "method": method, "seed": seed,
              "samples": SAMPLES, "starts": STARTS, "depth": depth, "size": instance.size, "sense": instance.sense,
              "count": instance.count, "weights": [float(w) for w in instance.data["weights"]],
              "parent": [int(a) for a in instance.data["parent"]], "coefficients": [float(c) for c in coefficients],
              "penalty": penalty, "angles": [float(a) for a in angles]}
    return record, rows, probability, rng


def code_version():
    done = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else "unknown"


def without_code(records):
    return [{key: value for key, value in record.items() if key != "code"} for record in records]


def main():
    instance = budget_instance(INSTANCE)
    table = instance.table()
    optimum = instance.optimum()
    records = []
    print(f"{instance.name}, pair cost fitted on {SAMPLES} plans, seed {INSTANCE}")
    print("   depth   right count   optimum, of the right count   mean regret, of the right count")
    for depth in (1, 2, 3):
        record, rows, probability, _ = train_record(instance, INSTANCE, depth)
        inside = probability[rows] / probability[rows].sum()
        best = table[rows] >= optimum * (1 - 1e-12)
        print(f"   {depth}       {100 * probability[rows].sum():6.1f}%       {100 * inside[best].sum():6.2f}%"
              f"                      {100 * float(inside @ (1 - table[rows] / optimum)):6.2f}%")
        records.append(record)
    saved = read_records(OUTPUT) if OUTPUT.exists() else []
    own = [record for record in saved if record["experiment"] == EXPERIMENT]
    others = [record for record in saved if record["experiment"] != EXPERIMENT]
    if without_code(own) != records:
        write_records(OUTPUT, [dict(record, code=code_version()) for record in records] + others)
    print(f"   records for depths 1, 2 and 3 are in {OUTPUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
