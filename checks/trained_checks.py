import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

from briskit import benchmark
from briskit.feeder import all_plans
from briskit.solvers import fitted, qaoa
from briskit.trained import read_records, record_circuit, record_value, within_limits, write_records
from checks.record import run_checks
from experiments.trained_circuits import INSTANCE, OUTPUT, SAMPLES, budget_instance, train_record

DEPTH = 2
BUDGET = 50


def write_rounded(path, records):
    rounded = [{key: [round(x, 6) for x in value] if key in ("coefficients", "angles") else value
                for key, value in record.items()} for record in records]
    write_records(path, rounded)


def circuit_without_penalty(record, measure=True):
    return record_circuit(dict(record, penalty=0.0, count=None), measure)


def record_with_another_seed(instance, seed, depth):
    return train_record(instance, seed + 1, depth)


def value_with_weights_reversed(record, plans):
    return record_value(dict(record, weights=record["weights"][::-1]), plans)


def always_within_limits(size_in_bytes, qubits):
    return True


def records_survive_the_file(write):
    record = train_record(budget_instance(INSTANCE), INSTANCE, DEPTH)[0]
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "records.json"
        write(path, [record])
        same = read_records(path) == [record]
    return "read back equal", "read back equal" if same else "read back different", same


def record_gives_the_trained_distribution(build):
    record, rows, probability, _ = train_record(budget_instance(INSTANCE), INSTANCE, DEPTH)
    gap = 0.5 * float(np.abs(Statevector(build(record, measure=False)).probabilities() - probability).sum())
    return "total variation below 1e-9", f"{gap:.1e}", gap < 1e-9


def record_is_the_benchmark_run(train):
    instance = budget_instance(INSTANCE)
    record, rows, probability, rng = train(instance, INSTANCE, DEPTH)
    inside = probability[rows] / probability[rows].sum()
    replayed = all_plans(instance.size)[rows[rng.choice(len(rows), size=BUDGET, p=inside)]]
    stream = benchmark.stream(INSTANCE, instance.name, record["method"])
    solved = fitted(SAMPLES, qaoa(DEPTH))(instance, BUDGET, stream)
    same = int(np.sum(np.all(replayed == solved, axis=1)))
    return f"{BUDGET} of {BUDGET} plans equal", f"{same} of {BUDGET}", same == BUDGET


def values_match_the_feeder_model(score):
    instance = budget_instance(INSTANCE)
    record = train_record(instance, INSTANCE, 1)[0]
    plans = all_plans(instance.size)
    worst = float(np.abs(score(record, plans) - instance.score(plans)).max())
    return "difference below 1e-12", f"{worst:.1e}", worst < 1e-12


def limits_are_enforced(limits):
    records = read_records(OUTPUT)
    saved = limits(OUTPUT.stat().st_size, max(record["size"] for record in records))
    got = (saved, limits(3 * 1024 * 1024, 12), limits(1024, 25))
    return (True, False, False), got, got == (True, False, False)


CASES = [
    (records_survive_the_file, write_records, write_rounded, "numbers rounded to six decimals"),
    (record_gives_the_trained_distribution, record_circuit, circuit_without_penalty, "the penalty left out"),
    (record_is_the_benchmark_run, train_record, record_with_another_seed, "another seed"),
    (values_match_the_feeder_model, record_value, value_with_weights_reversed, "weights in reversed order"),
    (limits_are_enforced, within_limits, always_within_limits, "a test that always says yes"),
]


if __name__ == "__main__":
    sys.exit(0 if run_checks(CASES, Path(__file__).with_suffix(".csv")) else 1)
