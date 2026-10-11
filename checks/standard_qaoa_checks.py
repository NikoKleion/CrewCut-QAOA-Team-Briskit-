import re
import sys
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

from briskit.cost_circuit import cost_circuit
from briskit.feeder import sample_instance
from briskit.instances import crew_assignment
from briskit.one_hot import cost_table, one_hot_circuit, one_hot_fields, one_hot_terms, penalty_bounds
from briskit.qaoa import qaoa_circuit, train_qaoa
from briskit.scheduling import smith_order, weighted_completion_cost
from checks.record import run_checks
from experiments.standard_qaoa import DEPTHS, SEED, STARTS, VARIANTS

ROOT = Path(__file__).resolve().parent.parent
REPAIRS = 6
INSTANCES = 30
FORMS = {"one qubit per repair": "project", "one penalty at the bound": ("uniform", 1.0),
         "one penalty at 4 times the bound": ("uniform", 4.0), "each repair's penalty at its own bound": ("own", 1.0)}
ROW = re.compile(r"^\s+(.*?)\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+([\d.]+)% \[\s*([\d.]+),\s*([\d.]+)\]\s+(\d+)\s+([\d.]+)%\s*$")


def saved_rows(text):
    rows, section = {}, None
    for line in text.split("\n"):
        head = re.match(r"^(\d+) repairs, two crews, \d+ instances, depth (\d+)", line)
        if head:
            section = (int(head.group(1)), int(head.group(2)))
        found = ROW.match(line)
        if found and section:
            rows[(*section, found.group(1).strip())] = (float(found.group(4)), float(found.group(5)), float(found.group(9)))
    return rows


def circuit_distribution(form, weights, hours, order, penalty, angles):
    n = len(weights)
    if form == "project":
        pairs = {(order[a], order[b]): weights[order[b]] * hours[order[a]] for a in range(n) for b in range(a + 1, n)}
        scale = float(np.sqrt(sum((c / 2) ** 2 for c in pairs.values())))
        return Statevector(qaoa_circuit(n, pairs, scale, angles, measure=False)).probabilities()
    _, single, pair = one_hot_terms(weights, hours, order, 2, penalty)
    return Statevector(one_hot_circuit(single, pair, angles, measure=False)).probabilities()


def circuit_without_single_terms(form, weights, hours, order, penalty, angles):
    if form == "project":
        return circuit_distribution(form, weights, hours, order, penalty, angles)
    _, single, pair = one_hot_terms(weights, hours, order, 2, penalty)
    fields, couplings = one_hot_fields(single, pair)
    return Statevector(cost_circuit(np.zeros_like(fields), couplings, angles, measure=False)).probabilities()


def crews_of(index, form, n):
    if form == "project":
        return [(index >> j) & 1 for j in range(n)]
    groups = [(index >> (2 * j)) & 3 for j in range(n)]
    return [group - 1 for group in groups] if all(group in (1, 2) for group in groups) else None


def schedule_cost(crews, order, weights, hours):
    return sum(weighted_completion_cost([j for j in order if crews[j] == m], weights, hours) for m in (0, 1))


def recomputed_medians(route):
    measures = {}
    for k in range(INSTANCES):
        weights, hours = sample_instance(np.random.default_rng([SEED, REPAIRS, k]), REPAIRS)
        order = smith_order(weights, hours)
        plans = [schedule_cost([(index >> j) & 1 for j in range(REPAIRS)], order, weights, hours) for index in range(2 ** REPAIRS)]
        best, random_mean = min(plans), float(np.mean(plans))
        bounds = penalty_bounds(weights, hours, order)
        for depth in DEPTHS:
            for label, form in FORMS.items():
                if form == "project":
                    tag, penalty, table = 0, None, crew_assignment("audit", weights, hours).table()
                else:
                    tag = VARIANTS[form][0]
                    penalty = form[1] * (bounds.max() if form[0] == "uniform" else bounds)
                    table = cost_table(*one_hot_terms(weights, hours, order, 2, penalty))
                angles, _ = train_qaoa(table, depth, np.random.default_rng([SEED, REPAIRS, k, depth, tag]), starts=STARTS)
                probability = route(form, weights, hours, order, penalty, angles)
                share = hit = mean = 0.0
                for index, chance in enumerate(probability):
                    crews = crews_of(index, form, REPAIRS)
                    if crews is not None:
                        cost = schedule_cost(crews, order, weights, hours)
                        share += chance
                        mean += chance * cost
                        hit += chance * (cost <= best * (1 + 1e-12))
                measures.setdefault((depth, label), []).append((100 * share, 100 * hit, 100 * (random_mean - mean / share) / (random_mean - best)))
    return {key: tuple(float(np.median([row[i] for row in rows])) for i in range(3)) for key, rows in measures.items()}


def saved_table_at_6_repairs_is_reproduced(route):
    saved = saved_rows((ROOT / "results" / "standard_qaoa.txt").read_text(encoding="utf-8"))
    equal, numbers = 0, 0
    for (depth, label), (share, hit, gain) in recomputed_medians(route).items():
        printed = saved[REPAIRS, depth, label]
        for got, want, digits in ((share, printed[0], 2), (hit, printed[1], 2), (gain, printed[2], 1)):
            numbers += 1
            equal += abs(got - want) <= 0.5001 * 10 ** -digits
    return f"{numbers} of {numbers} printed numbers", f"{equal} of {numbers}", equal == numbers


def saved_texts():
    return "\n".join((ROOT / "results" / f"{name}.txt").read_text(encoding="utf-8") for name in ("standard_qaoa", "standard_qaoa_10"))


def with_one_share_altered():
    return saved_texts().replace("93.19%     0.18% [ 0.18,  0.18]", "93.19%     0.28% [ 0.18,  0.18]", 1)


def ceiling_holds_at_every_size(text):
    rows = saved_rows(text)
    worst = 0.0
    for n in (6, 8, 10):
        share, hit, _ = rows[n, 1, "one penalty at 4 times the bound"]
        worst = max(worst, abs(hit - share * 2 / 2 ** n))
    return "best plan shots within 0.03 points of valid shots times 2 / 2^n", f"largest gap {worst:.3f} points", worst < 0.03


CASES = [
    (saved_table_at_6_repairs_is_reproduced, circuit_distribution, circuit_without_single_terms,
     "the standard form's circuit without its single-qubit terms"),
    (ceiling_holds_at_every_size, saved_texts(), with_one_share_altered(), "one saved best plan share altered"),
]


if __name__ == "__main__":
    sys.exit(0 if run_checks(CASES, Path(__file__).with_suffix(".csv")) else 1)
