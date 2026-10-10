import numpy as np

from .baselines import thermal_distributions
from .feeder import all_plans
from .pair_cost import ridge_coefficients, selection_features
from .qaoa import train_qaoa
from .scheduling import list_schedule, single_crew_order, smith_order


def allowed(instance):
    plans = all_plans(instance.size)
    if instance.count is None:
        return np.arange(len(plans)), plans
    rows = np.flatnonzero(plans.sum(axis=1) == instance.count)
    return rows, plans


def random_plans(instance, budget, rng):
    rows, plans = allowed(instance)
    return plans[rows[rng.integers(len(rows), size=budget)]]


def largest_weights(instance, budget, rng):
    plan = np.zeros(instance.size, dtype=np.int8)
    plan[np.argsort(-instance.data["weights"], kind="stable")[: instance.count]] = 1
    return plan[None, :]


def greedy(instance, budget, rng):
    plan = np.zeros(instance.size, dtype=np.int8)
    for _ in range(instance.count):
        free = np.flatnonzero(plan == 0)
        trials = np.repeat(plan[None, :], len(free), axis=0)
        trials[np.arange(len(free)), free] = 1
        scores = instance.score(trials)
        merit = scores if instance.sense == "max" else -scores
        plan[free[np.flatnonzero(merit >= merit.max() - 1e-12)[0]]] = 1
    return plan[None, :]


def conversion(instance, budget, rng):
    weights, hours, parent = instance.data["weights"], instance.data["hours"], instance.data["parent"]
    order = smith_order(weights, hours) if parent is None else single_crew_order(weights, hours, parent)
    return list_schedule(order, hours)[None, :]


def fitted_merit(coefficients, plans, sense):
    n = plans.shape[1]
    x = plans.astype(float)
    value = coefficients[0] + x @ coefficients[1:n + 1]
    first, second = np.triu_indices(n, 1)
    for c, i, j in zip(coefficients[n + 1:], first, second):
        value += c * x[:, i] * x[:, j]
    return value if sense == "max" else -value


def fit_on_sample(instance, samples, rng, pairs=True):
    rows, plans = allowed(instance)
    sample = rows[rng.choice(len(rows), size=min(samples, len(rows)), replace=False)]
    features = selection_features(plans[sample], pairs=pairs)
    coefficients = ridge_coefficients(features, instance.score(plans[sample]))
    if not pairs:
        coefficients = np.concatenate([coefficients, np.zeros(instance.size * (instance.size - 1) // 2)])
    return coefficients, rows, plans


def fitted(samples, select, pairs=True):
    def solve(instance, budget, rng):
        coefficients, rows, plans = fit_on_sample(instance, samples, rng, pairs)
        merit = fitted_merit(coefficients, plans, instance.sense)
        return plans[select(merit, rows, instance, budget, rng)]
    return solve


def best_first(merit, rows, instance, budget, rng):
    return rows[np.argsort(-merit[rows], kind="stable")[:budget]]


def thermal(beta):
    def select(merit, rows, instance, budget, rng):
        probability = thermal_distributions(-merit[rows], [beta])[0]
        return rows[rng.choice(len(rows), size=budget, p=probability)]
    return select


def penalised_cost(merit, rows, instance):
    cost = -merit
    penalty = 0.0
    if instance.count is not None:
        chosen = all_plans(instance.size).sum(axis=1)
        penalty = merit[rows].std()
        cost = cost + penalty * (chosen - instance.count) ** 2
    return cost, float(penalty)


def qaoa(depth, starts=2, maxiter=None):
    def select(merit, rows, instance, budget, rng):
        cost, _ = penalised_cost(merit, rows, instance)
        _, probability = train_qaoa(cost, depth, rng, starts=starts, maxiter=maxiter)
        inside = probability[rows] / probability[rows].sum()
        return rows[rng.choice(len(rows), size=budget, p=inside)]
    return select
