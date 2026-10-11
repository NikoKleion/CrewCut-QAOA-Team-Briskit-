from pathlib import Path

import numpy as np

from briskit.feeder import sample_instance
from briskit.instances import crew_assignment
from briskit.one_hot import assignments, cost_table, one_hot_terms, penalty_bounds
from briskit.qaoa import train_qaoa
from briskit.scheduling import list_schedule, smith_order
from experiments.standard_qaoa import SEED, STARTS, VARIANTS, shares

REPAIRS = 8
INSTANCE = 0
INSTANCES = 30
DEPTH = 2
PENALTIES = [("uniform", 1.0), ("uniform", 2.0), ("uniform", 4.0), ("own", 1.0)]
OUTPUT = Path(__file__).resolve().parent.parent / "qollab" / "game_instance.py"


def rule_gap(weights, hours):
    instance = crew_assignment("rule", weights, hours)
    return float(instance.score(list_schedule(smith_order(weights, hours), hours))[0] / instance.optimum() - 1)


def trained_forms():
    weights, hours = sample_instance(np.random.default_rng([SEED, REPAIRS, INSTANCE]), REPAIRS)
    order = smith_order(weights, hours)
    project = crew_assignment("game", weights, hours).table()
    best, random_mean = project.min(), project.mean()
    stream = np.random.default_rng([SEED, REPAIRS, INSTANCE, DEPTH, 0])
    angles, probability = train_qaoa(project, DEPTH, stream, starts=STARTS)
    crewcut = (shares(probability, project, np.ones(project.size, dtype=bool), best, random_mean), angles)
    valid, _ = assignments(REPAIRS, 2)
    bounds = penalty_bounds(weights, hours, order)
    standard = {}
    for variant in PENALTIES:
        kind, factor = variant
        penalty = factor * (np.full(REPAIRS, bounds.max()) if kind == "uniform" else bounds)
        table = cost_table(*one_hot_terms(weights, hours, order, 2, penalty))
        stream = np.random.default_rng([SEED, REPAIRS, INSTANCE, DEPTH, VARIANTS[variant][0]])
        angles, probability = train_qaoa(table, DEPTH, stream, starts=STARTS)
        standard[variant] = (shares(probability, table, valid, best, random_mean), angles, penalty)
    return weights, hours, crewcut, standard


def numbers(values):
    return "[" + ", ".join(repr(float(value)) for value in values) + "]"


def exact(measures):
    return "{" + ", ".join(f'"{name}": {value!r}' for name, value in zip(("valid", "best", "gain"), measures)) + "}"


def main():
    gaps = [rule_gap(*sample_instance(np.random.default_rng([SEED, REPAIRS, k]), REPAIRS)) for k in range(INSTANCES)]
    rule_best_on = sum(gap < 1e-12 for gap in gaps)
    weights, hours, crewcut, standard = trained_forms()
    picked = max(PENALTIES, key=lambda variant: standard[variant][0][1])
    print(f"{REPAIRS} repairs, two crews, instance {INSTANCE} of the {INSTANCES} of experiments/standard_qaoa.py, depth {DEPTH}, {STARTS} starts")
    print(f"   the simple dispatch rule is a best plan on {rule_best_on} of {INSTANCES} instances; on instance {INSTANCE} it is"
          f" {100 * gaps[INSTANCE]:.2f}% above the best plan")
    print(f"   hours {[int(h) for h in hours]}")
    print(f"   urgency {[round(float(w), 4) for w in weights]}")
    print("   form                                          valid shots   best plan shots   gain kept, valid shots")
    print(f"   one qubit per repair                             {100 * crewcut[0][0]:7.2f}%   {100 * crewcut[0][1]:14.2f}%   {100 * crewcut[0][2]:8.1f}%")
    for variant in PENALTIES:
        share, hit, kept = standard[variant][0]
        mark = "   picked" if variant == picked else ""
        print(f"   standard, {VARIANTS[variant][1]:38s}{100 * share:7.2f}%   {100 * hit:14.2f}%   {100 * kept:8.1f}%{mark}")
    print("the standard form's penalty is picked by the share of best plan shots, which uses the answer")
    OUTPUT.write_text("\n".join([
        "# written by experiments/dispatch_game_instance.py",
        f"REPAIRS = {REPAIRS}",
        f"HOURS = {numbers(hours)}",
        f"URGENCY = {numbers(weights)}",
        f"CREWCUT_ANGLES = {numbers(crewcut[1])}",
        f'PENALTY_NAME = "{VARIANTS[picked][1]}"',
        f"PENALTY = {numbers(standard[picked][2])}",
        f"STANDARD_ANGLES = {numbers(standard[picked][1])}",
        f'EXACT = {{"crewcut": {exact(crewcut[0])}, "standard": {exact(standard[picked][0])}}}',
        f"RULE_BEST_ON = {rule_best_on}",
        f"INSTANCES = {INSTANCES}",
        f"RULE_GAP = {gaps[INSTANCE]!r}",
    ]) + "\n", encoding="utf-8", newline="\n")
    print(f"written: qollab/{OUTPUT.name}")


if __name__ == "__main__":
    main()
