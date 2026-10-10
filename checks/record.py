import csv

COLUMNS = ["check", "version", "expected", "got", "verdict", "required", "as_required"]


def run_checks(cases, path):
    rows = []
    for check, real, broken, description in cases:
        for version, target, required in (("real", real, "PASS"), (f"broken: {description}", broken, "FAIL")):
            expected, got, passed = check(target)
            verdict = "PASS" if passed else "FAIL"
            rows.append({"check": check.__name__, "version": version, "expected": expected, "got": got,
                         "verdict": verdict, "required": required,
                         "as_required": "yes" if verdict == required else "NO"})
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['as_required']:3s}  {row['verdict']}  {row['check']}  [{row['version']}]  expected {row['expected']}  got {row['got']}")
    return all(row["as_required"] == "yes" for row in rows)
