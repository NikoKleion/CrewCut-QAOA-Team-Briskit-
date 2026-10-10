import hashlib
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .feeder import all_plans

FAMILIES = ("random_geometric", "ring_with_chords", "hub_and_spoke")
DAMAGED_COUNTS = (12, 16)  # plan section 2.4
SEEDS = (0, 1, 2)  # plan section 2.4: three seeds
NODES = 16  # project choice, no source
TIER_WEIGHTS = (4.0, 2.0, 1.0)  # project choice, no source; three tiers from plan section 2.4
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "flow"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "flow"


def connected(node_count, links):
    parent = list(range(node_count))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v in links:
        parent[root(u)] = root(v)
    return len({root(a) for a in range(node_count)}) == 1


def random_geometric_network(rng, min_links, radius=0.35):  # radius: project choice, no source
    while True:
        positions = rng.uniform(0.0, 1.0, (NODES, 2))  # NetworkX random_geometric_graph: uniform in the unit cube
        links = [(u, v) for u in range(NODES) for v in range(u + 1, NODES)
                 if np.linalg.norm(positions[u] - positions[v]) <= radius]
        if len(links) >= min_links and connected(NODES, links):
            capacities = rng.integers(5, 16, len(links))  # project choice, no source
            return positions, links, capacities


def ring_with_chords_network(rng, min_links):
    angles = 2 * np.pi * np.arange(NODES) / NODES
    positions = 0.5 + 0.5 * np.column_stack([np.cos(angles), np.sin(angles)])
    links = [(u, (u + 1) % NODES) for u in range(NODES)]
    links = [(min(u, v), max(u, v)) for u, v in links]
    while len(links) < min_links:  # Newman-Watts: shortcuts added to the ring, none removed (NetworkX docs)
        u, v = sorted(int(a) for a in rng.choice(NODES, 2, replace=False))
        if (u, v) not in links:
            links.append((u, v))
    capacities = rng.integers(5, 16, len(links))  # project choice, no source
    return positions, links, capacities


def hub_and_spoke_network(rng, min_links, hubs=3, second_hub_share=0.75):  # project choice, no source
    while True:
        positions = np.vstack([rng.uniform(0.35, 0.65, (hubs, 2)),  # project choice: hubs near the centre
                               rng.uniform(0.0, 1.0, (NODES - hubs, 2))])
        links = [(u, v) for u in range(hubs) for v in range(u + 1, hubs)]
        for spoke in range(hubs, NODES):
            nearest = np.argsort(np.linalg.norm(positions[:hubs] - positions[spoke], axis=1))
            links.append((int(nearest[0]), spoke))
            if rng.random() < second_hub_share:
                links.append((int(nearest[1]), spoke))
        if len(links) >= min_links:
            capacities = np.where([v < hubs for _, v in links], 20, rng.integers(5, 16, len(links)))  # project choice
            return positions, links, capacities


GENERATORS = {"random_geometric": random_geometric_network, "ring_with_chords": ring_with_chords_network,
              "hub_and_spoke": hub_and_spoke_network}


def damage_disk(rng, positions, links, count):
    center = rng.uniform(0.0, 1.0, 2)
    midpoints = np.array([(positions[u] + positions[v]) / 2 for u, v in links])
    distance = np.linalg.norm(midpoints - center, axis=1)
    nearest = np.argsort(distance)[:count]
    return sorted(int(e) for e in nearest), center, float(distance[nearest[-1]])


def random_demands(rng):
    demands = []
    for k in range(int(rng.integers(6, 9))):  # plan section 2.4: 6 to 8 commodities
        source, sink = (int(a) for a in rng.choice(NODES, 2, replace=False))
        tier = k % len(TIER_WEIGHTS)
        demands.append({"source": source, "sink": sink, "demand": float(rng.integers(2, 9)),  # project choice
                        "tier": tier, "weight": TIER_WEIGHTS[tier]})
    return demands


def generate_instance(family, damaged, seed):
    rng = np.random.default_rng([20261010, FAMILIES.index(family), damaged, seed])  # repo-wide SEED, split per instance
    positions, links, capacities = GENERATORS[family](rng, int(np.ceil(1.5 * damaged)))  # project choice: 1.5x spare
    while True:
        elements, center, radius = damage_disk(rng, positions, links, damaged)
        instance = {"name": f"{family}_{damaged}_{seed}", "family": family, "seed": seed, "synthetic": True,
                    "network": {"nodes": positions.round(6).tolist(),
                                "links": [[int(u), int(v), float(c)] for (u, v), c in zip(links, capacities)]},
                    "elements": elements, "demands": random_demands(rng),
                    "budget": [damaged // 3, damaged // 2],  # links restored; project choice, 4 and 6 of 12 as plan 9.2
                    "damage": {"center": center.round(6).tolist(), "radius": round(radius, 6)}}
        program = FlowProgram(instance)
        if program.value(np.zeros(damaged)) < program.value(damaged=False) - 1e-6:
            return instance


def handmade_instance():
    return {"name": "handmade", "family": "handmade", "seed": None, "synthetic": True,
            "network": {"nodes": [[0, 0], [1, 1], [2, 0], [1, -1], [3, 0]],
                        "links": [[0, 1, 4.0], [1, 2, 3.0], [0, 3, 2.0], [3, 2, 2.0], [2, 4, 5.0]]},
            "elements": [1, 3],
            "demands": [{"source": 0, "sink": 2, "demand": 4.0, "tier": 0, "weight": 2.0},
                        {"source": 0, "sink": 4, "demand": 3.0, "tier": 1, "weight": 1.0}],
            "budget": [1], "damage": None}


def write_atomically(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(temporary, "wb") as handle:
        write(handle)
    os.replace(temporary, path)  # swaps in the whole file; atomic on POSIX (Python os.replace docs)


def save_instance(instance, path):
    write_atomically(path, lambda handle: handle.write(json.dumps(instance, indent=1).encode()))


def load_instance(path):
    return json.loads(Path(path).read_text())


def instance_paths():
    return [DATA_DIR / f"{family}_{damaged}_{seed}.json"
            for damaged in DAMAGED_COUNTS for family in FAMILIES for seed in SEEDS]


class FlowProgram:
    def __init__(self, instance):
        nodes = len(instance["network"]["nodes"])
        links = instance["network"]["links"]
        demands = instance["demands"]
        arcs, commodities = 2 * len(links), len(demands)
        self.flow_count = commodities * arcs
        self.capacity = np.array([c for _, _, c in links])
        self.elements = np.array(instance["elements"], dtype=int)
        rows, cols, vals = [], [], []
        for k in range(commodities):
            for e, (u, v, _) in enumerate(links):
                for d, (tail, head) in enumerate(((u, v), (v, u))):
                    column = k * arcs + 2 * e + d
                    rows += [k * nodes + tail, k * nodes + head]
                    cols += [column, column]
                    vals += [1.0, -1.0]
            served = self.flow_count + k
            rows += [k * nodes + demands[k]["source"], k * nodes + demands[k]["sink"]]
            cols += [served, served]
            vals += [-1.0, 1.0]
        size = self.flow_count + commodities
        self.a_eq = coo_matrix((vals, (rows, cols)), shape=(commodities * nodes, size)).tocsr()
        self.b_eq = np.zeros(commodities * nodes)
        link_of = np.tile(np.repeat(np.arange(len(links)), 2), commodities)
        self.a_ub = coo_matrix((np.ones(self.flow_count), (link_of, np.arange(self.flow_count))),
                               shape=(len(links), size)).tocsr()
        self.cost = np.concatenate([np.zeros(self.flow_count), -np.array([d["weight"] for d in demands])])
        self.bounds = [(0, None)] * self.flow_count + [(0, d["demand"]) for d in demands]

    def value(self, plan=None, damaged=True):
        available = np.ones(len(self.capacity))
        if damaged:
            available[self.elements] = 0.0 if plan is None else np.asarray(plan, dtype=float)
        result = linprog(self.cost, A_ub=self.a_ub, b_ub=self.capacity * available, A_eq=self.a_eq,
                         b_eq=self.b_eq, bounds=self.bounds, method="highs")
        if result.status != 0:
            raise RuntimeError(result.message)
        return -result.fun + 0.0


def undamaged_value(instance):
    return FlowProgram(instance).value(damaged=False)


def _table_chunk(arguments):
    instance, plans = arguments
    program = FlowProgram(instance)
    return [program.value(plan) for plan in plans]


def value_table(instance, chunks=256):
    plans = all_plans(len(instance["elements"]))
    processes = int(os.environ.get("FLOW_PROCESSES", os.cpu_count()))
    if processes == 1 or len(plans) <= 64:  # project choice: below this the pool costs more than it saves
        return np.array(_table_chunk((instance, plans)))
    pieces = np.array_split(plans, min(chunks, len(plans)))
    with Pool(processes) as pool:
        return np.concatenate(pool.map(_table_chunk, [(instance, piece) for piece in pieces]))


def stored_instance(name):
    if name == "handmade":
        instance = handmade_instance()
    else:
        family, damaged, seed = name.rsplit("_", 2)
        instance = generate_instance(family, int(damaged), int(seed))
    path = DATA_DIR / f"{name}.json"
    if not path.exists() or path.read_text() != json.dumps(instance, indent=1):
        save_instance(instance, path)
    return instance


def instance_names():
    return [path.stem for path in instance_paths()]


def fingerprint(instance):
    return hashlib.sha256(json.dumps(instance, sort_keys=True).encode()).hexdigest()


def stored_value_table(instance):
    path = RESULTS_DIR / "tables" / f"{instance['name']}.npz"
    if path.exists():
        with np.load(path) as stored:
            if str(stored["fingerprint"]) == fingerprint(instance):
                return stored["table"], float(stored["seconds"])
    start = time.perf_counter()
    table = value_table(instance)
    seconds = time.perf_counter() - start
    write_atomically(path, lambda handle: np.savez(handle, table=table, fingerprint=fingerprint(instance),
                                                   seconds=seconds))
    return table, seconds
