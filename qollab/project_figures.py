import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
STYLE = "tableau-colorblind10"
# Okabe and Ito, Color Universal Design: colour plus shape, line style and hatching, https://jfly.uni-koeln.de/color/
HATCHES = ["", "//", "..", "xx", "\\\\"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
LINES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
BUDGET_PLOTTED = ["random plans", "fitted cost, best first", "fitted cost, thermal beta 2", "fitted cost, QAOA depth 3"]
RULES = {"restoration": "greedy", "crews": "conversion algorithm"}


def palette():
    return plt.style.library[STYLE]["axes.prop_cycle"].by_key()["color"]


def percent(x):
    return f"{100 * x:.2f}%"


def pipeline_figure(data):
    budget = data["budget"]
    stages = [
        ("Scoring model", "value of any plan:\nfeeder rule or\nflow linear program"),
        ("Exact value table", "every plan of a\nsmall instance:\nground truth"),
        ("Pair cost", f"one qubit per element,\nfitted on {budget['fit']}\nrandom plans"),
        ("QAOA", "angles trained locally,\nQiskit circuit,\none job to the backend"),
        ("Benchmark", f"every method gets\n{budget['total']} evaluations,\nfit counted"),
    ]
    colors = palette()
    figure, axes = plt.subplots(figsize=(14, 3.2))
    width, gap = 2.7, 0.5
    for k, (title, body) in enumerate(stages):
        left = k * (width + gap)
        axes.add_patch(FancyBboxPatch((left, 0), width, 2, boxstyle="round,pad=0.05", linewidth=2,
                                      edgecolor=colors[0], facecolor="white"))
        axes.text(left + width / 2, 1.62, title, ha="center", va="center", fontsize=13, fontweight="bold")
        axes.text(left + width / 2, 0.8, body, ha="center", va="center", fontsize=10)
        if k < len(stages) - 1:
            axes.annotate("", xy=(left + width + gap - 0.05, 1), xytext=(left + width + 0.1, 1),
                          arrowprops={"arrowstyle": "-|>", "linewidth": 2, "color": colors[3]})
    axes.set_xlim(-0.2, len(stages) * (width + gap) - gap + 0.2)
    axes.set_ylim(-0.2, 2.2)
    axes.axis("off")
    caption = (f"Every method, classical or quantum, is scored on the true model with the same budget of "
               f"{budget['total']} evaluations, and the {budget['fit']} used to fit the pair cost count against it.")
    alt = ("Flow diagram of five stages from left to right: scoring model, exact value table, pair cost, QAOA, "
           "benchmark. The text below the figure describes each stage.")
    description = "\n".join(f"{k + 1}. {title}: {body.replace(chr(10), ' ')}" for k, (title, body) in enumerate(stages))
    return {"name": "pipeline", "figure": figure, "alt": alt, "caption": caption, "description": description}


def spectrum_figure(data):
    spectrum = data["spectrum"]
    rows = spectrum["rows"]
    colors = palette()
    figure, axes = plt.subplots(figsize=(11, 4.2))
    for row_index, row in enumerate(rows):
        start = 0.0
        for order_index, share in enumerate(row["shares"]):
            axes.barh(row_index, 100 * share, left=100 * start, color=colors[order_index], edgecolor="black",
                      hatch=HATCHES[order_index], label=spectrum["orders"][order_index] if row_index == 0 else None)
            if share >= 0.06:
                axes.text(100 * (start + share / 2), row_index, f"{100 * share:.1f}%", ha="center", va="center",
                          fontsize=10.5, fontweight="bold",
                          bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none"})
            start += share
    axes.set_yticks(range(len(rows)), [row["decision"] for row in rows], fontsize=11)
    axes.invert_yaxis()
    axes.set_xlim(0, 100)
    axes.set_xlabel("share of the variation in the value table (%)", fontsize=11)
    axes.axvline(0, color="black")
    axes.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=10.5)
    axes.set_title(f"Interaction order of each decision, 12 lines, mean over {spectrum['instances']} instances",
                   fontsize=12)
    share = {row["decision"]: row["shares"] for row in rows}
    crew = share["crew assignment on the feeder"]
    restoration = share["budgeted restoration on the feeder"]
    protection = share["protection, independent lines"]
    caption = (f"A pair cost captures orders 1 and 2 only. Those orders hold {100 * (crew[0] + crew[1]):.1f}% of the "
               f"variation for crew assignment on the feeder and {100 * (protection[0] + protection[1]):.1f}% for "
               f"protection, but {100 * (restoration[0] + restoration[1]):.1f}% for budgeted restoration, where "
               f"{100 * sum(restoration[2:]):.1f}% sits at order 3 and above.")
    alt = ("Stacked horizontal bar chart: for four decisions, the share of variation at interaction orders 1 to 5. "
           "The table below the figure gives every value.")
    header = f"{'decision':40s}" + "".join(f"{order:>10s}" for order in spectrum["orders"])
    lines = [header] + [f"{row['decision']:40s}" + "".join(f"{100 * x:9.1f}%" for x in row["shares"]) for row in rows]
    return {"name": "interaction_spectrum", "figure": figure, "alt": alt, "caption": caption,
            "description": "\n".join(lines)}


def budget_figure(data):
    budget = data["budget"]
    colors = palette()
    figure, all_axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for axes, panel in zip(all_axes, budget["panels"]):
        methods = {method["name"]: method for method in panel["methods"]}
        plotted = BUDGET_PLOTTED + [RULES[panel["key"]]]
        tallest = 0.0
        for k, name in enumerate(plotted):
            means = methods[name]["mean"]
            points = [(t + 1, 100 * m) for t, m in enumerate(means) if m is not None]
            x, y = zip(*points)
            tallest = max([tallest] + [value for t, value in points if t >= 25])
            axes.plot(x, y, color=colors[k], linestyle=LINES[k], linewidth=2.2, marker=MARKERS[k],
                      markevery=max(1, len(x) // 8), markersize=7, label=name)
        axes.axvline(budget["fit"], color="black", linewidth=1, linestyle=":")
        axes.text(budget["fit"] - 2, 0.03, "fit done", transform=axes.get_xaxis_transform(), ha="right",
                  va="bottom", fontsize=10)
        axes.set_xlim(1, budget["total"])
        axes.set_ylim(0, 1.1 * tallest)
        axes.set_xlabel("evaluations of the true model, fit included", fontsize=11)
        axes.set_ylabel("mean regret (%)", fontsize=11)
        axes.set_title(f"{panel['title']}, {panel['instances']} feeder instances", fontsize=12)
        axes.grid(alpha=0.3)
        axes.legend(fontsize=9.5, loc="upper right")
    figure.tight_layout()
    total = str(budget["total"])
    final = {panel["key"]: {m["name"]: m["summary"][total]["mean"] for m in panel["methods"]}
             for panel in budget["panels"]}
    restoration, crews = final["restoration"], final["crews"]
    caption = (f"After {total} evaluations, the {budget['fit']} for the fit included, the fitted pair cost with "
               f"best-first selection has mean regret {percent(restoration['fitted cost, best first'])} on "
               f"restoration against {percent(restoration['random plans'])} for random plans. QAOA at depth 3 and a "
               f"classical thermal sampler on the same cost end close together: "
               f"{percent(restoration['fitted cost, QAOA depth 3'])} and "
               f"{percent(restoration['fitted cost, thermal beta 2'])} on restoration, "
               f"{percent(crews['fitted cost, QAOA depth 3'])} and {percent(crews['fitted cost, thermal beta 2'])} "
               f"on crew assignment.")
    alt = ("Two line charts of mean regret against evaluations for restoration and crew assignment on the feeder. "
           "The table below the figure gives the values.")
    lines = []
    for panel in budget["panels"]:
        lines.append(f"{panel['title']}: regret in percent, median (lower to upper quartile) and mean")
        totals = list(panel["methods"][0]["summary"])
        lines.append(f"{'method':30s}" + "".join(f"{t:>30s}" for t in totals))
        for method in panel["methods"]:
            cells = []
            for t in totals:
                s = method["summary"][t]
                cells.append(f"{'none':>30s}" if s is None else
                             f"{100 * s['median']:7.2f} ({100 * s['lower']:5.2f} to {100 * s['upper']:6.2f}) "
                             f"{100 * s['mean']:6.2f}")
            lines.append(f"{method['name']:30s}" + "".join(cells))
        lines.append("")
    return {"name": "budget_curves", "figure": figure, "alt": alt, "caption": caption,
            "description": "\n".join(lines).rstrip()}


def gate_figure(data):
    rows = data["gates"]["rows"]
    colors = palette()
    figure, axes = plt.subplots(figsize=(9, 4.6))
    width = 0.38
    tallest = max(row["heavy_hex"] for row in rows)
    for k, (key, label) in enumerate([("all_pairs", "every pair interacts (trapped ions)"),
                                      ("heavy_hex", "heavy-hex layout")]):
        positions = [i + (k - 0.5) * width for i in range(len(rows))]
        values = [row[key] for row in rows]
        bars = axes.bar(positions, values, width, color=colors[k], edgecolor="black", hatch=HATCHES[k * 2],
                        label=label)
        for bar, value in zip(bars, values):
            axes.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom", fontsize=10)
    for i, row in enumerate(rows):
        axes.text(i + width / 2, row["heavy_hex"] + 0.08 * tallest, f"x{row['ratio']:.2f}", ha="center",
                  fontsize=11, fontweight="bold")
    axes.set_xticks(range(len(rows)), [f"{row['repairs']} repairs" for row in rows], fontsize=11)
    axes.set_ylabel("two-qubit gates in one cost layer", fontsize=11)
    axes.set_ylim(0, tallest * 1.25)
    axes.legend(fontsize=10.5, loc="upper left")
    axes.set_title("One QAOA cost layer with a term on every pair", fontsize=12)
    ratios = [row["ratio"] for row in rows]
    first = rows[0]
    caption = (f"A cost layer with a term on every pair needs {min(ratios):.2f} to {max(ratios):.2f} times as many "
               f"two-qubit gates on a heavy-hex layout as on a device where every pair of qubits interacts: "
               f"{first['heavy_hex']} against {first['all_pairs']} at {first['repairs']} repairs. At IonQ Forte's "
               f"two-qubit error rate, {100 * first['unaffected']:.0f}% of shots pass that layer without a "
               f"two-qubit error.")
    alt = ("Grouped bar chart of two-qubit gate counts for 12, 22 and 29 repairs, all-pairs device against heavy-hex "
           "layout. The table below the figure gives the values.")
    lines = [f"{'repairs':>8s}{'all pairs':>11s}{'depth':>7s}{'heavy hex':>11s}{'depth':>7s}{'ratio':>7s}"
             f"{'shots without error':>21s}"]
    lines += [f"{row['repairs']:8d}{row['all_pairs']:11d}{row['all_pairs_depth']:7d}{row['heavy_hex']:11d}"
              f"{row['heavy_hex_depth']:7d}{row['ratio']:7.2f}{100 * row['unaffected']:20.0f}%" for row in rows]
    return {"name": "gate_counts", "figure": figure, "alt": alt, "caption": caption, "description": "\n".join(lines)}


# W3C WAI, complex images: short alt text plus a long description in the page, https://www.w3.org/WAI/tutorials/images/complex/
def draw_all(data, folder="", show_text=True):
    figures = []
    with plt.style.context(STYLE):
        for build in (pipeline_figure, spectrum_figure, budget_figure, gate_figure):
            figure = build(data)
            # Qollab shows a saved figure in the console and offers the PNG, https://qollab.xyz/learn/docs/runtime-environment
            figure["figure"].savefig(folder + figure["name"] + ".png", dpi=150, bbox_inches="tight",
                                     facecolor="white")
            plt.close(figure["figure"])
            del figure["figure"]
            figures.append(figure)
            if show_text:
                print()
                print(f"figure {len(figures)}: {figure['name']}")
                print(f"caption: {figure['caption']}")
                print(f"alt text: {figure['alt']}")
                print(figure["description"])
    return figures


def figure_notes(figures):
    parts = ["# Figures", ""]
    for k, figure in enumerate(figures, 1):
        parts += [f"## Figure {k}: {figure['name']}", "", f"![{figure['alt']}]({figure['name']}.png)", "",
                  f"**Caption.** {figure['caption']}", "", "**Description.**", "", "```", figure["description"],
                  "```", ""]
    return "\n".join(parts)
