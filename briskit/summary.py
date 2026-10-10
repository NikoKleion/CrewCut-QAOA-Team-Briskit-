import numpy as np


def by_run(rows):
    runs = {}
    for row in rows:
        runs.setdefault((row["instance"], row["method"], row["seed"]), []).append(row)
    for group in runs.values():
        group.sort(key=lambda row: (row["evaluations"], row["budget"]))
    return runs


def regret_at(run, total):
    reached = [row for row in run if row["evaluations"] <= total]
    return reached[-1]["regret"] if reached else None


def regret_summary(runs, method, total):
    values = [regret_at(run, total) for (instance, name, seed), run in runs.items() if name == method]
    values = np.array([value for value in values if value is not None])
    if values.size == 0:
        return None
    lower, median, upper = np.percentile(values, [25, 50, 75])
    return {"runs": int(values.size), "median": float(median), "lower": float(lower), "upper": float(upper),
            "mean": float(values.mean())}


def runtime_to_target(runs, method, target, tolerance=1e-12):
    reached, unsuccessful, count = [], 0, 0
    for (instance, name, seed), run in runs.items():
        if name != method:
            continue
        count += 1
        hits = [row["evaluations"] for row in run if row["regret"] <= target + tolerance]
        if hits:
            reached.append(hits[0])
        else:
            unsuccessful += run[-1]["evaluations"]
    average = (unsuccessful + sum(reached)) / len(reached) if reached else None
    return {"runs": count, "reached": len(reached), "average": average}
