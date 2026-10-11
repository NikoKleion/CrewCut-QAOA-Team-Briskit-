import sys
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "qollab" / "game"))

import dispatch_game as game
import game_instance as saved
from briskit.feeder import sample_instance
from briskit.instances import crew_assignment
from briskit.one_hot import assignments, cost_table, one_hot_terms
from briskit.qaoa import qaoa_probabilities, standardize
from briskit.scheduling import list_schedule, smith_order, weighted_completion_cost
from checks.record import run_checks
from experiments.standard_qaoa import SEED, shares

WEIGHTS, HOURS = np.array(saved.URGENCY), np.array(saved.HOURS)
N = saved.REPAIRS


def library_costs():
    return crew_assignment("game", WEIGHTS, HOURS).table()


def library_standard_table():
    return cost_table(*one_hot_terms(WEIGHTS, HOURS, smith_order(WEIGHTS, HOURS), 2, np.array(saved.PENALTY)))


def costs_with_weight_and_hours_swapped():
    order = game.smith_order()
    index = np.arange(2 ** N)
    cost = np.full(index.size, float(np.dot(WEIGHTS, HOURS)))
    for a in range(N):
        for b in range(a + 1, N):
            i, j = order[a], order[b]
            cost += WEIGHTS[i] * HOURS[j] * (((index >> i) & 1) == ((index >> j) & 1))
    return cost


def crewcut_with_the_angles_in_the_wrong_order(measure=True):
    kept = game.CREWCUT_ANGLES
    game.CREWCUT_ANGLES = kept[2:] + kept[:2]
    try:
        return game.crewcut_circuit(measure)
    finally:
        game.CREWCUT_ANGLES = kept


def standard_without_single_terms(measure=True):
    kept = game.standard_terms
    game.standard_terms = lambda: (np.zeros(2 * N), kept()[1])
    try:
        return game.standard_circuit(measure)
    finally:
        game.standard_terms = kept


def decode_grouped_by_crew(index):
    plan = 0
    for j in range(N):
        first, second = (index >> j) & 1, (index >> (N + j)) & 1
        if first + second != 1:
            return -1
        plan |= second << j
    return plan


def with_one_share_altered():
    return {"crewcut": dict(saved.EXACT["crewcut"], best=saved.EXACT["crewcut"]["best"] + 0.01),
            "standard": saved.EXACT["standard"]}


def game_costs_equal_the_library(costs):
    worst = float(np.abs(costs() - library_costs()).max())
    return "difference below 1e-9", f"largest difference {worst:.1e}", worst < 1e-9


def crewcut_circuit_gives_the_trained_distribution(builder):
    got = Statevector(builder(measure=False)).probabilities()
    worst = float(np.abs(got - qaoa_probabilities(standardize(library_costs()), saved.CREWCUT_ANGLES)).max())
    return "difference below 1e-10", f"largest difference {worst:.1e}", worst < 1e-10


def standard_circuit_gives_the_trained_distribution(builder):
    got = Statevector(builder(measure=False)).probabilities()
    worst = float(np.abs(got - qaoa_probabilities(standardize(library_standard_table()), saved.STANDARD_ANGLES)).max())
    return "difference below 1e-10", f"largest difference {worst:.1e}", worst < 1e-10


def decoder_agrees_with_the_library(decode):
    valid, crew = assignments(N, 2)
    plan = crew.astype(np.int64) @ (2 ** np.arange(N))
    equal = sum(decode(index) == (int(plan[index]) if valid[index] else -1) for index in range(valid.size))
    return f"{valid.size} of {valid.size} states", f"{equal} of {valid.size}", equal == valid.size


def saved_shares_are_those_of_the_trained_circuits(exact):
    project = library_costs()
    best, random_mean = project.min(), project.mean()
    crewcut = shares(qaoa_probabilities(standardize(project), saved.CREWCUT_ANGLES), project,
                     np.ones(project.size, dtype=bool), best, random_mean)
    table = library_standard_table()
    standard = shares(qaoa_probabilities(standardize(table), saved.STANDARD_ANGLES), table, assignments(N, 2)[0],
                      best, random_mean)
    worst = max(abs(got - exact[form][name]) for form, values in (("crewcut", crewcut), ("standard", standard))
                for name, got in zip(("valid", "best", "gain"), values))
    return "difference below 1e-9", f"largest difference {worst:.1e}", worst < 1e-9


def schedule_cost(plan, order, weights, hours):
    return sum(weighted_completion_cost([j for j in order if (plan >> j) & 1 == crew], weights, hours) for crew in (0, 1))


def rule_figures_are_recomputed(figures):
    best_on, gap_here = 0, None
    for k in range(saved.INSTANCES):
        weights, hours = sample_instance(np.random.default_rng([SEED, N, k]), N)
        order = smith_order(weights, hours)
        optimum = min(schedule_cost(plan, order, weights, hours) for plan in range(2 ** N))
        rule = list_schedule(order, hours)
        gap = schedule_cost(int(rule.astype(int) @ (2 ** np.arange(N))), order, weights, hours) / optimum - 1
        best_on += gap < 1e-12
        gap_here = gap if k == 0 else gap_here
    got = (int(best_on), round(float(gap_here), 9))
    want = (figures[0], round(figures[1], 9))
    return f"{want}", f"{got}", got == want


CASES = [
    (game_costs_equal_the_library, game.plan_costs, costs_with_weight_and_hours_swapped,
     "pair term with the weight of the earlier repair and the hours of the later one"),
    (crewcut_circuit_gives_the_trained_distribution, game.crewcut_circuit, crewcut_with_the_angles_in_the_wrong_order,
     "the angles of the mixer used for the cost layer"),
    (standard_circuit_gives_the_trained_distribution, game.standard_circuit, standard_without_single_terms,
     "single-qubit terms left out"),
    (decoder_agrees_with_the_library, game.decode, decode_grouped_by_crew, "qubits grouped by crew in the decoder"),
    (saved_shares_are_those_of_the_trained_circuits, saved.EXACT, with_one_share_altered(), "one saved share altered"),
    (rule_figures_are_recomputed, (saved.RULE_BEST_ON, saved.RULE_GAP), (saved.RULE_BEST_ON + 1, saved.RULE_GAP),
     "the count of instances altered"),
]


if __name__ == "__main__":
    sys.exit(0 if run_checks(CASES, Path(__file__).with_suffix(".csv")) else 1)
