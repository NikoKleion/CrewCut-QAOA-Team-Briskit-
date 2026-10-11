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


def ladder_data():
    output, code = output_and_code("flow_ladder.txt")
    blocks = []
    for block in output.split("\n\nflow networks"):
        head = re.search(r"restore (\d+) of (\d+) damaged links, (\d+) instances, (\d+) seeds, fit on (\d+) plans", block)
        if not head:
            continue
        lines = block.splitlines()
        totals = re.findall(r"\d+", next(line for line in lines if line.strip().startswith("method") and
                                         re.search(r"\d", line)))
        regret, reached, gates, part = {}, {}, {}, None
        for line in lines:
            if line.startswith("regret at a total"):
                part = "regret"
            elif line.startswith("runs that reach"):
                part = "targets"
            elif "pair gates per cost layer" in line:
                part = "gates"
            elif part == "regret":
                found = re.match(r"\s+(.+?)\s{2,}((?:[\d.]+ \(\s*[\d.]+ to\s+[\d.]+\)\s+[\d.]+\s*)+)$", line)
                if found:
                    means = re.findall(r"\)\s+([\d.]+)", found.group(2))
                    regret[found.group(1)] = dict(zip(totals, (float(m) / 100 for m in means)))
            elif part == "targets":
                found = re.match(r"\s+(.+?)\s{2,}.*?(\d+) of (\d+),\s+(\d+)$", line)
                if found:
                    reached[found.group(1)] = {"runs": int(found.group(2)), "of": int(found.group(3)),
                                               "evaluations": int(found.group(4))}
            elif part == "gates":
                found = re.match(r"\s+(.+?)\s{2,}(\d+)\s+(\d+)$", line)
                if found:
                    gates[found.group(1)] = {"with_penalty": int(found.group(2)), "without": int(found.group(3))}
        if not regret or set(regret) != set(reached):
            raise ValueError(f"unexpected method tables in flow_ladder.txt for restore {head.group(1)}")
        blocks.append({"restore": int(head.group(1)), "damaged": int(head.group(2)),
                       "instances": int(head.group(3)), "seeds": int(head.group(4)), "fit": int(head.group(5)),
                       "total": totals[-1], "methods": [{"name": name, "regret": regret[name],
                                                         "optimum": reached[name]} for name in regret],
                       "gates": gates})
    if len(blocks) != 2:
        raise ValueError(f"expected 2 restore counts in flow_ladder.txt, found {len(blocks)}")
    return {"source": "results/flow_ladder.txt", "code": code, "blocks": blocks}


def main():
    data = {"spectrum": spectrum_data(), "gates": gate_data(), "budget": budget_data(), "ladder": ladder_data()}
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
