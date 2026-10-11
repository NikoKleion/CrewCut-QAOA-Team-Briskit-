# Wall clock time on one machine: exhaustive search, one dispatch plan, training the circuit's angles, 12 repairs; exhaustive search at 12 to 20 repairs
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saved_run import saved_run

import time

import numpy as np

from briskit.feeder import sample_instance, upstream_parents
from briskit.instances import crew_assignment
from briskit.qaoa import train_qaoa
from briskit.scheduling import list_schedule, smith_order

SEED = 20261010
REPAIRS = 12
SIZES = (12, 14, 16, 18, 20)


def median_seconds(action, runs):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        action()
        times.append(time.perf_counter() - start)
    return np.median(times), min(times), max(times)


def line(label, timing):
    middle, low, high = timing
    return f"   {label:58s} {1000 * middle:10.2f} ms   ({1000 * low:.2f} to {1000 * high:.2f})"


def main():
    weights, hours = sample_instance(np.random.default_rng([SEED, 0]), REPAIRS)
    feeder = crew_assignment("feeder", weights, hours, upstream_parents())
    table = crew_assignment("independent", weights, hours).table()
    rng = np.random.default_rng(SEED)
    _, probability = train_qaoa(table, 1, rng)
    bits = np.arange(REPAIRS)

    def hundred_shots():
        drawn = rng.choice(table.size, size=100, p=probability)
        return feeder.score((drawn[:, None] >> bits) & 1)

    print(f"{REPAIRS} repairs, two crews, one instance, median of 5 runs (lowest to highest); times differ from run to run")
    print(line(f"exhaustive search, feeder model, {table.size} plans", median_seconds(lambda: feeder.table().min(), 5)))
    print(line("dispatch order and list schedule, one plan scored",
               median_seconds(lambda: feeder.score(list_schedule(smith_order(weights, hours), hours)), 5)))
    print(line("training the angles, one layer, table simulator", median_seconds(lambda: train_qaoa(table, 1, rng), 5)))
    print(line("drawing 100 plans from the trained distribution, scored", median_seconds(hundred_shots, 5)))
    print("exhaustive search on independent repairs by size, median of 3 runs")
    for n in SIZES:
        w, p = sample_instance(np.random.default_rng([SEED, n]), n)
        instance = crew_assignment("independent", w, p)
        print(line(f"{n} repairs, {2 ** n} plans", median_seconds(lambda: instance.table().min(), 3)))
    print("simulation on a classical machine; nothing here is the clock of a quantum processor")


if __name__ == "__main__":
    saved_run(__file__, main)
