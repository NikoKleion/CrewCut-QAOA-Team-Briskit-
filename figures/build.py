import json
import re
import sys
from pathlib import Path

from briskit import summary
from briskit.benchmark import read_rows
from experiments.budget_experiment import FINAL, SAMPLES

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DATA_FILE = ROOT / "qollab" / "figure_data.json"
ORDERS = ["order 1", "order 2", "order 3", "order 4", "order 5+"]
SUMMARY_TOTALS = (50, 100, 150, 200)

BUDGET_PANELS = [
    {"key": "restoration", "title": "Restore 6 of 12 lines",
     "file": "feeder/budget_experiment_restoration.csv",
     "methods": ["random plans", "fitted cost, best first", "fitted cost, thermal beta 2", "fitted cost, thermal beta 5",
                 "fitted cost, QAOA depth 1", "fitted cost, QAOA depth 2", "fitted cost, QAOA depth 3",
                 "greedy", "largest weights"]},
    {"key": "crews", "title": "12 repairs on two crews",
     "file": "feeder/budget_experiment_crews.csv",
     "methods": ["random plans", "fitted cost, best first", "fitted cost, thermal beta 2", "fitted cost, thermal beta 5",
                 "fitted cost, QAOA depth 1", "fitted cost, QAOA depth 2", "fitted cost, QAOA depth 3",
                 "conversion algorithm"]},
]


def output_and_code(name):
    header, output = (RESULTS / name).read_text(encoding="utf-8").split("\n\n", 1)
    fields = dict(line.split(": ", 1) for line in header.splitlines())
    return output, fields["code"]


def spectrum_data():
    output, code = output_and_code("interaction_spectrum.txt")
    title = output.splitlines()[0]
    instances = int(re.search(r"mean over (\d+) instances", title).group(1))
    rows = []
    for line in output.splitlines():
        found = re.match(r"\s+(.+?)\s+((?:\d+\.\d%\s*){5})$", line)
        if found:
            shares = [float(x) / 100 for x in re.findall(r"(\d+\.\d)%", found.group(2))]
            rows.append({"decision": found.group(1), "shares": shares})
    if len(rows) != 4:
        raise ValueError(f"expected 4 decisions in interaction_spectrum.txt, found {len(rows)}")
    return {"source": "results/interaction_spectrum.txt", "code": code, "instances": instances,
            "orders": ORDERS, "rows": rows}


def gate_data():
    output, code = output_and_code("gate_counts.txt")
    pattern = (r"(\d+) repairs: all pairs\s+(\d+) gates, depth\s+(\d+), shots without a two-qubit error (\d+)%"
               r" \| heavy hex\s+(\d+) gates, depth\s+(\d+) \| ratio ([\d.]+)")
    rows = [{"repairs": int(a), "all_pairs": int(b), "all_pairs_depth": int(c), "unaffected": int(d) / 100,
             "heavy_hex": int(e), "heavy_hex_depth": int(f), "ratio": float(g)}
            for a, b, c, d, e, f, g in re.findall(pattern, output)]
    if not rows:
        raise ValueError("no rows found in gate_counts.txt")
    return {"source": "results/gate_counts.txt", "code": code, "rows": rows}


def budget_data():
    _, code = output_and_code("budget_experiment.txt")
    total = SAMPLES + FINAL
    panels = []
    for panel in BUDGET_PANELS:
        runs = summary.by_run(read_rows(RESULTS / panel["file"]))
        instances = len({instance for instance, _, _ in runs})
        methods = []
        for method in panel["methods"]:
            means, table = [], {}
            for t in range(1, total + 1):
                s = summary.regret_summary(runs, method, t)
                means.append(None if s is None else s["mean"])
                if t in SUMMARY_TOTALS:
                    table[str(t)] = s
            methods.append({"name": method, "mean": means, "summary": table})
        panels.append({"key": panel["key"], "title": panel["title"], "source": "results/" + panel["file"],
                       "instances": instances, "methods": methods})
    return {"code": code, "fit": SAMPLES, "final": FINAL, "total": total, "panels": panels}


def main():
    data = {"spectrum": spectrum_data(), "gates": gate_data(), "budget": budget_data()}
    text = json.dumps(data, indent=1)
    if not DATA_FILE.exists() or DATA_FILE.read_text(encoding="utf-8") != text:
        DATA_FILE.write_text(text, encoding="utf-8")
    sys.path.insert(0, str(ROOT / "qollab"))
    import matplotlib
    # https://matplotlib.org/stable/users/explain/figure/backends.html: Agg writes files only
    matplotlib.use("Agg")
    from project_figures import draw_all, figure_notes
    FIGURES.mkdir(exist_ok=True)
    figures = draw_all(data, str(FIGURES) + "/", show_text=False)
    (FIGURES / "captions.md").write_text(figure_notes(figures), encoding="utf-8")
    print(f"wrote {DATA_FILE.relative_to(ROOT).as_posix()} ({len(text) // 1024} kB)")
    for figure in figures:
        print(f"wrote figures/{figure['name']}.png")
    print("wrote figures/captions.md")


if __name__ == "__main__":
    main()
