import json

import numpy as np

from .benchmark import plan_indices
from .cost_circuit import cost_circuit, ising_terms
from .feeder import restoration_value

FILE_LIMIT = 2 * 1024 * 1024
QUBIT_LIMIT = 24


def write_records(path, records):
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        json.dump(records, file, indent=1)


def read_records(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def record_circuit(record, measure=True):
    fields, couplings = ising_terms(np.array(record["coefficients"]), record["size"], record["sense"],
                                    record["penalty"], record["count"])
    return cost_circuit(fields, couplings, record["angles"], measure)


def record_value(record, plans):
    if "values" in record:
        return np.array(record["values"])[plan_indices(plans)]
    return restoration_value(np.atleast_2d(plans), np.array(record["weights"]), record["parent"])


def within_limits(size_in_bytes, qubits):
    return size_in_bytes <= FILE_LIMIT and qubits <= QUBIT_LIMIT
