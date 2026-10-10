import csv
import time

import numpy as np

from briskit.feeder import all_plans
from briskit.flow import RESULTS_DIR, FlowProgram, instance_names, stored_instance

SEED = 20261010


def main(pairs=200, timed=50):
    rng = np.random.default_rng(SEED)
    handmade = FlowProgram(stored_instance("handmade"))
    values = [handmade.value(plan) for plan in all_plans(2)]
    assert np.allclose(values, [0.0, 6.0, 4.0, 9.0]), values  # worked on paper, syllabus.md
    print(f"hand-made instance: {values} (expected [0, 6, 4, 9])")
    print(f"{'instance':24s} undamaged  all restored  none restored  monotone pairs")
    timings = []
    for name in instance_names():
        instance = stored_instance(name)
        program = FlowProgram(instance)
        m = len(instance["elements"])
        full = program.value(damaged=False)
        restored = program.value(np.ones(m))
        assert abs(full - restored) < 1e-6, (name, full, restored)
        for _ in range(pairs):
            lower = rng.integers(0, 2, m)
            upper = lower | rng.integers(0, 2, m)
            assert program.value(lower) <= program.value(upper) + 1e-9, name  # tolerance: HiGHS round-off
        plans = rng.integers(0, 2, (timed, m))
        start = time.perf_counter()
        for plan in plans:
            program.value(plan)
        timings.append({"instance": name, "programs": timed,
                        "milliseconds_per_program": round(1000 * (time.perf_counter() - start) / timed, 3)})
        print(f"{name:24s} {full:9.2f}  {restored:12.2f}  {program.value(np.zeros(m)):13.2f}  {pairs:14d}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "program_time.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timings[0]))
        writer.writeheader()
        writer.writerows(timings)
    print("all checks hold; time of one program in results/flow/program_time.csv")


if __name__ == "__main__":
    main()
