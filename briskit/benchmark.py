import csv
import time
import zlib

import numpy as np

COLUMNS = ["experiment", "instance", "method", "budget", "seed", "value", "regret", "returned", "evaluations",
           "seconds", "script"]


def plan_indices(plans):
    plans = np.atleast_2d(np.asarray(plans)).astype(np.int64)
    return plans @ (np.int64(1) << np.arange(plans.shape[1], dtype=np.int64))


class Counted:
    def __init__(self, instance):
        self.name = instance.name
        self.size = instance.size
        self.sense = instance.sense
        self.count = instance.count
        self.data = instance.data
        self.record = {}
        self._instance = instance

    def score(self, plans):
        values = self._instance.score(plans)
        for index, value in zip(plan_indices(plans), values):
            self.record.setdefault(int(index), float(value))
        return values

    def used(self):
        return len(self.record)


def stream(seed, instance_name, method):
    return np.random.default_rng([seed, zlib.crc32(instance_name.encode()), zlib.crc32(method.encode())])


def accepted(plans, instance, budget):
    plans = np.asarray(plans)
    if plans.size == 0:
        return np.zeros((0, instance.size), dtype=np.int8)
    plans = np.atleast_2d(plans)
    if plans.shape[1] != instance.size:
        raise ValueError(f"a plan has {plans.shape[1]} entries, the instance has {instance.size} elements")
    if len(plans) > budget:
        raise ValueError(f"{len(plans)} plans returned for a budget of {budget}")
    if not np.isin(plans, (0, 1)).all():
        raise ValueError("a plan has an entry that is not 0 or 1")
    if instance.count is not None and (plans.sum(axis=1) != instance.count).any():
        raise ValueError(f"a plan does not have {instance.count} elements chosen")
    return plans.astype(np.int8)


def best(values, sense):
    values = list(values)
    if not values:
        return float("nan")
    return float(max(values) if sense == "max" else min(values))


def regret(value, optimum, sense):
    return (optimum - value) / optimum if sense == "max" else (value - optimum) / optimum


def run(experiment, instances, solvers, budgets, seeds, script, prefix_safe=(), trace=False):
    budgets = sorted(budgets)
    rows = []

    def record(instance, method, budget, seed, value, returned, optimum, evaluations, seconds):
        rows.append({"experiment": experiment, "instance": instance.name, "method": method, "budget": budget,
                     "seed": seed, "value": value, "regret": regret(value, optimum, instance.sense),
                     "returned": returned, "evaluations": evaluations, "seconds": seconds, "script": script})

    def solved(solve, counted, budget, seed, instance, method):
        started = time.perf_counter()
        plans = accepted(solve(counted, budget, stream(seed, instance.name, method)), instance, budget)
        return plans, time.perf_counter() - started

    for instance in instances:
        optimum = instance.optimum()
        for method, solve in solvers.items():
            for seed in seeds:
                if method in prefix_safe:
                    counted = Counted(instance)
                    plans, seconds = solved(solve, counted, budgets[-1], seed, instance, method)
                    scored = list(counted.record.values())
                    known = set(counted.record)
                    values = [float(value) for value in instance.score(plans)] if len(plans) else []
                    indices = [int(index) for index in plan_indices(plans)] if len(plans) else []
                    if trace:
                        for t in range(1, len(scored) + 1):
                            record(instance, method, 0, seed, best(scored[:t], instance.sense), float("nan"),
                                   optimum, t, seconds)
                    for budget in budgets:
                        if budget == 0 and trace and scored:
                            continue
                        record(instance, method, budget, seed, best(scored + values[:budget], instance.sense),
                               best(values[:budget], instance.sense), optimum,
                               len(known | set(indices[:budget])), seconds)
                else:
                    for budget in budgets:
                        counted = Counted(instance)
                        plans, seconds = solved(solve, counted, budget, seed, instance, method)
                        values = [float(value) for value in counted.score(plans)] if len(plans) else []
                        record(instance, method, budget, seed, best(counted.record.values(), instance.sense),
                               best(values, instance.sense), optimum, counted.used(), seconds)
    return rows


def write_rows(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for column in ("budget", "seed", "evaluations"):
            row[column] = int(row[column])
        for column in ("value", "regret", "returned", "seconds"):
            row[column] = float(row[column])
    return rows
