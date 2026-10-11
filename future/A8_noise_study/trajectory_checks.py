# Checks of the error trajectory simulator, each run on the real code and on a broken version
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import saved_run

import numpy as np

from briskit.qaoa import qaoa_probabilities, standardize, train_qaoa
from checks.record import run_checks
from crew_pairs import cost_table, instance, pair_coefficients
from pauli_trajectories import noisy_distribution

SEED = 20261010
REPAIRS = 8


def case():
    weights, hours = instance(REPAIRS, 0)
    pairs = pair_coefficients(weights, hours)
    return pairs, cost_table(weights, hours, pairs), np.random.default_rng(SEED).uniform(-0.8, 0.8, 4)


def with_doubled_pair_angle(n, pairs, scale, angles, rate, trajectories, rng):
    return noisy_distribution(n, pairs, scale / 2, angles, rate, trajectories, rng)


def with_errors_ignored(n, pairs, scale, angles, rate, trajectories, rng):
    return noisy_distribution(n, pairs, scale, angles, 0.0, trajectories, rng)


def no_error_gives_the_table_distribution(simulate):
    pairs, table, angles = case()
    got = simulate(REPAIRS, pairs, table.std(), angles, 0.0, 1, np.random.default_rng(SEED))
    worst = float(np.abs(got - qaoa_probabilities(standardize(table), angles)).max())
    return "difference below 1e-12", f"largest difference {worst:.1e}", worst < 1e-12


def errors_pull_the_cost_toward_random_plans(simulate):
    pairs, table, _ = case()
    angles, _ = train_qaoa(table, 2, np.random.default_rng(SEED), starts=2)
    clean = simulate(REPAIRS, pairs, table.std(), angles, 0.0, 1, np.random.default_rng(SEED)) @ table
    noisy = simulate(REPAIRS, pairs, table.std(), angles, 0.05, 200, np.random.default_rng(SEED)) @ table
    share = float((table.mean() - noisy) / (table.mean() - clean))
    return "between 0.03 and 0.9 of the gain over random plans kept", f"{share:.3f} kept", 0.03 < share < 0.9


def every_trajectory_is_a_distribution(simulate):
    pairs, table, angles = case()
    got = simulate(REPAIRS, pairs, table.std(), angles, 0.2, 20, np.random.default_rng(SEED))
    return "sums to 1 within 1e-12", f"sum off by {abs(got.sum() - 1):.1e}", abs(got.sum() - 1) < 1e-12 and got.min() >= 0


def without_normalising(n, pairs, scale, angles, rate, trajectories, rng):
    return noisy_distribution(n, pairs, scale, angles, rate, trajectories, rng) * trajectories / (trajectories + 1)


CASES = [
    (no_error_gives_the_table_distribution, noisy_distribution, with_doubled_pair_angle, "pair gate angle doubled"),
    (errors_pull_the_cost_toward_random_plans, noisy_distribution, with_errors_ignored, "errors ignored"),
    (every_trajectory_is_a_distribution, noisy_distribution, without_normalising, "average taken over one trajectory too many"),
]


if __name__ == "__main__":
    sys.exit(0 if run_checks(CASES, Path(__file__).with_suffix(".csv")) else 1)
