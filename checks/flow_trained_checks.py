import sys
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

from briskit import benchmark
from briskit.feeder import all_plans
from briskit.solvers import fitted, qaoa
from briskit.trained import record_circuit, record_value
from checks.record import run_checks
from experiments.flow_budget_experiment import SAMPLES
from experiments.flow_trained_circuit import SEED, flow_instance, train_record

DEPTH = 2
BUDGET = 50


def circuit_without_penalty(record, measure=True):
    return record_circuit(dict(record, penalty=0.0, count=None), measure)


def record_with_another_seed(instance, table, seed, depth):
    return train_record(instance, table, seed + 1, depth)


def value_with_values_reversed(record, plans):
    return record_value(dict(record, values=record["values"][::-1]), plans)


def record_gives_the_trained_distribution(build):
    instance, table = flow_instance()
    record, rows, probability, _ = train_record(instance, table, SEED, DEPTH)
    gap = 0.5 * float(np.abs(Statevector(build(record, measure=False)).probabilities() - probability).sum())
    return "total variation below 1e-9", f"{gap:.1e}", gap < 1e-9


def record_is_the_benchmark_run(train):
    instance, table = flow_instance()
    record, rows, probability, rng = train(instance, table, SEED, DEPTH)
    inside = probability[rows] / probability[rows].sum()
    replayed = all_plans(instance.size)[rows[rng.choice(len(rows), size=BUDGET, p=inside)]]
    stream = benchmark.stream(SEED, instance.name, record["method"])
    solved = fitted(SAMPLES[instance.size], qaoa(DEPTH, at_bound=True))(instance, BUDGET, stream)
    same = int(np.sum(np.all(replayed == solved, axis=1)))
    return f"{BUDGET} of {BUDGET} plans equal", f"{same} of {BUDGET}", same == BUDGET


def values_match_the_flow_table(score):
    instance, table = flow_instance()
    record = train_record(instance, table, SEED, 1)[0]
    plans = all_plans(instance.size)
    worst = float(np.abs(score(record, plans) - instance.score(plans)).max())
    return "difference below 1e-12", f"{worst:.1e}", worst < 1e-12


CASES = [
    (record_gives_the_trained_distribution, record_circuit, circuit_without_penalty, "the penalty left out"),
    (record_is_the_benchmark_run, train_record, record_with_another_seed, "another seed"),
    (values_match_the_flow_table, record_value, value_with_values_reversed, "values in reversed order"),
]


if __name__ == "__main__":
    sys.exit(0 if run_checks(CASES, Path(__file__).with_suffix(".csv")) else 1)
