import numpy as np

from .scheduling import smith_order


def pair_coefficients(weights, hours):
    order = smith_order(weights, hours)
    n = len(weights)
    pairs = {}
    for a in range(n):
        for b in range(a + 1, n):
            i, j = order[a], order[b]
            pairs[(i, j)] = weights[j] * hours[i]
    return pairs


def level_vectors(hours, ratio):
    levels = np.unique(ratio)[::-1]
    gaps = np.append(levels[:-1] - levels[1:], levels[-1])
    vectors = np.array([np.where(ratio >= level, hours, 0.0) for level in levels])
    return gaps, vectors


def level_count(ratio):
    return len(np.unique(np.round(ratio, 12)))


def pair_features(plans):
    spins = 1.0 - 2.0 * plans.astype(float)
    upper = np.triu_indices(spins.shape[1], 1)
    return np.concatenate([np.ones((spins.shape[0], 1)), spins[:, upper[0]] * spins[:, upper[1]]], axis=1)


def fit_coefficients(features, values, rows):
    coefficients, *_ = np.linalg.lstsq(features[rows], values[rows], rcond=None)
    return coefficients


def fit_pair_cost(features, values, rows):
    return features @ fit_coefficients(features, values, rows)


def selection_features(plans, pairs=True):
    x = np.asarray(plans, dtype=float)
    columns = [np.ones((x.shape[0], 1)), x]
    if pairs:
        upper = np.triu_indices(x.shape[1], 1)
        columns.append(x[:, upper[0]] * x[:, upper[1]])
    return np.concatenate(columns, axis=1)


def ridge_coefficients(features, values, strength=1e-3):
    gram = features.T @ features + strength * np.eye(features.shape[1])
    return np.linalg.solve(gram, features.T @ values)


def coupling_matrix(coefficients, n):
    matrix = np.zeros((n, n))
    matrix[np.triu_indices(n, 1)] = coefficients[1:]
    return matrix + matrix.T


def eigen_blocks(matrix):
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(-np.abs(values))
    return values[order], vectors[:, order]


def walsh_hadamard(table):
    a = np.asarray(table, dtype=float).copy()
    h = 1
    while h < a.size:
        a = a.reshape(-1, 2, h)
        a = np.stack([a[:, 0] + a[:, 1], a[:, 0] - a[:, 1]], axis=1).reshape(-1)
        h *= 2
    return a / a.size


def interaction_order_shares(table, top=4):
    power = walsh_hadamard(table) ** 2
    order = np.array([bin(i).count("1") for i in range(power.size)])
    total = power[order > 0].sum()
    return [power[order == d].sum() / total for d in range(1, top + 1)] + [power[order > top].sum() / total]


def protection_terms(weights, hours, failure):
    order = smith_order(weights, hours)
    n = len(weights)
    single = failure * weights * hours
    pair = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            i, j = order[a], order[b]
            pair[i, j] = pair[j, i] = failure[i] * failure[j] * weights[j] * hours[i]
    return single, pair
