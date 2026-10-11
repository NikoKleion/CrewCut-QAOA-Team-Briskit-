from main import ANGLES, record_name, run_with_plans


def run_page(device, shots, depth, keep):
    shots, depth, keep = int(shots), int(depth), int(keep)
    if keep not in ANGLES or depth not in ANGLES[keep]:
        raise ValueError(f"keep must be one of {sorted(ANGLES)} and depth one of {sorted(ANGLES[100])}")
    lines, plans = run_with_plans(device, shots, depth, keep)
    record = ["project file: qollab/page_run.py, called from qollab/main.js"] + lines
    return {"lines": lines, "plans": plans, "name": record_name(lines, depth, keep),
            "record": "\n".join(record) + "\n", "one_job": "ionq" in lines[0].lower()}
