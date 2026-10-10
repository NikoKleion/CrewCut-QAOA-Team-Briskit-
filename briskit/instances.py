import numpy as np

from .feeder import all_plans, restoration_value
from .pair_cost import protection_terms
from .scheduling import energization_cost, pairwise_cost, single_crew_order, smith_order


class Instance:
    def __init__(self, name, size, sense, score, count=None, data=None):
        self.name = name
        self.size = size
        self.sense = sense
        self.count = count
        self.data = data or {}
        self._score = score

    def score(self, plans):
        return self._score(np.atleast_2d(np.asarray(plans)).astype(np.int8))

    def table(self):
        return self.score(all_plans(self.size))

    def feasible(self):
        plans = all_plans(self.size)
        if self.count is None:
            return np.ones(len(plans), dtype=bool)
        return plans.sum(axis=1) == self.count

    def optimum(self):
        values = self.table()[self.feasible()]
        return float(values.max() if self.sense == "max" else values.min())


def restoration(name, weights, parent, count):
    weights = np.asarray(weights, dtype=float)
    return Instance(name, len(weights), "max", lambda plans: restoration_value(plans, weights, parent), count,
                    {"weights": weights, "parent": list(parent)})


def crew_assignment(name, weights, hours, parent=None):
    weights, hours = np.asarray(weights, dtype=float), np.asarray(hours, dtype=float)
    if parent is None:
        order = smith_order(weights, hours)
        score = lambda plans: pairwise_cost(plans, order, weights, hours)
    else:
        order = single_crew_order(weights, hours, parent)
        score = lambda plans: energization_cost(plans, order, weights, hours, parent)
    return Instance(name, len(weights), "min", score, None,
                    {"weights": weights, "hours": hours, "parent": None if parent is None else list(parent)})


def protection(name, weights, hours, failure, count):
    weights, hours, failure = (np.asarray(x, dtype=float) for x in (weights, hours, failure))
    single, pair = protection_terms(weights, hours, failure)

    def score(plans):
        exposed = 1.0 - plans
        return exposed @ single + 0.5 * np.einsum("ri,ij,rj->r", exposed, pair, exposed)

    return Instance(name, len(weights), "min", score, count, {"weights": weights, "hours": hours, "failure": failure})
