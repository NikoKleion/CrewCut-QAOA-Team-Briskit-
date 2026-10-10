import numpy as np

from briskit.ion_model import collective_drive, overlap, target_operator

HOURS = np.array([3.0, 5.0, 8.0])
STRENGTH = 0.02 * HOURS


def best_match(strength, periods):
    block, residual = collective_drive(strength, periods)
    matches = {sign: overlap(block, target_operator(strength, periods, sign=sign)) for sign in (1, -1)}
    sign = max(matches, key=matches.get)
    return matches[sign], sign, residual


def main():
    print("three ions and one vibrational mode, laser strength proportional to repair hours 3, 5, 8")
    signs = {}
    for periods in ((5, 3, 2), (4, 4, 4), (5, 2.5, 2)):
        match, signs[periods], residual = best_match(STRENGTH, periods)
        label = "on for " + ", ".join(str(x) for x in periods) + " periods"
        print(f"   {label:26s} match to the target {match:.6f}   left in the mode {residual:.2e}", flush=True)
    periods = (5, 3, 2)
    sign = signs[periods]
    ratios = [float(HOURS[i] * HOURS[j] * min(periods[i], periods[j])) for i, j in ((0, 1), (0, 2), (1, 2))]
    print(f"   target pair phases for 5, 3, 2 periods are in the ratio {ratios}")

    flipped = STRENGTH * np.array([1, -1, 1])
    block, _ = collective_drive(flipped, periods)
    print("sign controls, on for 5, 3, 2 periods")
    print(f"   phase of the second laser flipped: match to the target with that strength negative"
          f" {overlap(block, target_operator(flipped, periods, sign=sign)):.6f},"
          f" to the unflipped target {overlap(block, target_operator(STRENGTH, periods, sign=sign)):.6f}")
    block, _ = collective_drive(STRENGTH, periods, detuning=-1.0)
    print(f"   drive on the other side of the mode frequency: match to the target with the detuning negative"
          f" {overlap(block, target_operator(STRENGTH, periods, detuning=-1.0, sign=sign)):.6f}")


if __name__ == "__main__":
    main()
