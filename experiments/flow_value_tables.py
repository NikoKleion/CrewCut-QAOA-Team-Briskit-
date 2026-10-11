import csv

import numpy as np

from briskit.flow import RESULTS_DIR, instance_names, stored_instance, stored_value_table, undamaged_value


def main():
    rows = []
    print(f"{'instance':24s} {'plans':>6s} {'none':>7s} {'all':>7s} {'undamaged':>9s} {'idle links':>10s}")
    for name in ["handmade"] + instance_names():
        instance = stored_instance(name)
        table, seconds = stored_value_table(instance)
        full = undamaged_value(instance)
        assert abs(table[-1] - full) < 1e-6, name
        index = np.arange(table.size)
        idle = 0
        for bit in range(len(instance["elements"])):
            gain = table[index | (1 << bit)] - table
            assert np.all(gain >= -1e-9), name  # tolerance: HiGHS round-off
            idle += bool(np.all(gain < 1e-9))
        rows.append({"instance": name, "family": instance["family"], "damaged": len(instance["elements"]),
                     "plans": table.size, "seconds": round(seconds, 2), "none_restored": table[0],
                     "all_restored": table[-1], "undamaged": full, "idle_links": idle})
        print(f"{name:24s} {table.size:6d} {table[0]:7.2f} {table[-1]:7.2f} {full:9.2f} {idle:10d}", flush=True)
    with open(RESULTS_DIR / "value_tables.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("monotone in every link on every table; idle links never change the value;"
          " time per table in results/flow/value_tables.csv")


if __name__ == "__main__":
    main()
