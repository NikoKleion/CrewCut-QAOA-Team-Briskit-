import numpy as np
from qiskit import transpile

from briskit.collective import collective_layer, level_blocks
from briskit.feeder import sample_instance

SEED = 20261010


def leveled_repairs(rng, n, levels):
    hours = rng.integers(1, 11, n).astype(float)
    ratios = np.sort(rng.uniform(0.2, 3.0, levels))[::-1]
    level = rng.integers(0, levels, n)
    level[:levels] = np.arange(levels)
    return ratios[level] * hours, hours


def main(n=12):
    rng = np.random.default_rng(SEED)
    cases = (("distinct urgency ratios", sample_instance(rng, n)), ("3 urgency levels", leveled_repairs(rng, n, 3)))
    print(f"one cost layer, {n} repairs, {n * (n - 1) // 2} pairs")
    for label, (weights, hours) in cases:
        strengths, amplitudes = level_blocks(weights, hours)
        layer = collective_layer(n, strengths, amplitudes, 0.37, 10.0)
        written = layer.decompose().count_ops().get("rzz", 0)
        compiled = transpile(layer, basis_gates=["rzz", "rx", "ry", "rz"], optimization_level=3, seed_transpiler=1)
        print(f"   {label:24s} {len(strengths):2d} blocks, {written:3d} RZZ gates as written,"
              f" {compiled.count_ops().get('rzz', 0):3d} after compiling")


if __name__ == "__main__":
    main()
