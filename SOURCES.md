Sources for the flow networks (area B). Each source was opened and the part used was read on 2026-10-10 unless stated otherwise. Constants with no source are listed at the end.

SciPy linprog, https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html. The form minimise c @ x subject to A_ub x <= b_ub, A_eq x = b_eq and bounds; method "highs"; status 0 means optimal. Used in briskit/flow.py.

Multi-commodity flow problem, Wikipedia, https://en.wikipedia.org/wiki/Multi-commodity_flow_problem. Commodities given by source, sink and demand; the capacity constraint summed over commodities; flow conservation. The page has no weighted variant, so the weighted objective with served demand capped at the demand is this project's own. Used in briskit/flow.py.

NetworkX random_geometric_graph, https://networkx.org/documentation/stable/reference/generated/networkx.generators.geometric.random_geometric_graph.html. Nodes uniform in the unit cube, linked when their distance is at most the radius. Definition only; NetworkX is not used. Used in briskit/flow.py.

NetworkX newman_watts_strogatz_graph, https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.newman_watts_strogatz_graph.html. A ring with shortcuts added and none removed; the page cites Newman and Watts, Physics Letters A 263, 341 (1999). Definition only. Used in briskit/flow.py.

Newman and Watts, Scaling and percolation in the small-world network model, https://arxiv.org/abs/cond-mat/9904419. Abstract only; it does not itself describe the construction. Not cited in the code.

Udekwe et al., Q-RESTORE, https://arxiv.org/abs/2501.11197. Abstract only. Choosing which damaged links to restore within a budget, on road networks. Used in experiments/flow_fitted_pair_cost.py.

Python multiprocessing, https://docs.python.org/3/library/multiprocessing.html. Pool.map returns results in input order; the spawn start method on Windows needs the __main__ guard. Used in briskit/flow.py.

Python os.replace, https://docs.python.org/3/library/os.html#os.replace. Replaces the destination file silently; the rename is atomic on POSIX. The page does not promise atomicity on Windows. Used in briskit/flow.py to swap in finished files.

Simulated annealing, Wikipedia, https://en.wikipedia.org/wiki/Simulated_annealing. Read 2026-10-10. Kirkpatrick et al.'s acceptance rule: accept if the new state is better, else with probability exp(-(e' - e)/T); T is lowered over the run, with the schedule left to the user; neighbours alter the current state conservatively, for example by a swap. The page does not give a geometric schedule. Used in experiments/flow_budget_experiment.py.

Lucas, Ising formulations of many NP problems, https://arxiv.org/abs/1302.5843. Read 2026-10-10, section 2.3 (Cliques). Equation 13 writes "exactly K variables are 1" as the penalty A(K - sum of x)^2 added to the objective with weight B; equation 14 and the line after it show the lowest states keep the count only when A is large enough against B (A > KB there). Used for the count penalty at the bound in briskit/solvers.py, experiments/flow_budget_experiment.py, experiments/flow_ladder.py and experiments/flow_trained_circuit.py.

RestoreBench-Q Plan Addendum, sections 5.2 and 6.2. Team document. Gives the ladder of rungs 0 to 5, tasks B9 and B10, and the change to B7 (the count penalty in the standard form). Used in experiments/flow_ladder.py and experiments/flow_trained_circuit.py.

Ridge fit, from briskit/pair_cost.py; its sources are in syllabus.md. Used in experiments/flow_fitted_pair_cost.py.

RestoreBench-Q Project Plan, sections 2.4 and 9.2 and experiments E1 to E4. Team document. Gives 12 and 16 damaged links, three seeds, the three graph families, a regional damage disk, 6 to 8 commodities in three priority tiers, and restoring 4 and 6 of 12. Used in all area B files.

Project choices with no source. 16 nodes per network. Tier weights 4, 2 and 1. Random geometric radius 0.35. Link capacities whole numbers from 5 to 15, and 20 between hubs. Three hubs placed in [0.35, 0.65] squared, with 75% of spokes also linked to the second-nearest hub. Demands whole numbers from 2 to 8. At least 1.5 times as many links as damaged links. The damage disk is the smallest disk around a uniform random centre that holds the nearest link midpoints, redrawn with the demands until the damage lowers the served value. Restoring 5 and 8 of 16, stored as the instance budget. Value tables with 64 plans or fewer are computed without a process pool. Fit sizes 10 to 320, doubling, with 10 draws each. Budget experiment (E8): 30 seeds; fits on 150 plans for 12 damaged links (as in experiments/budget_experiment.py) and 400 for 16 (from Jithesh's fit sizes); swap annealing temperature falling geometrically from 0.1 to 0.005 of the current value (Jithesh's end points, which were a share of the optimum). The hand-made instance of 5 nodes with values 0, 6, 4 and 9 worked on paper. At most 6 worker processes for the QAOA experiments, as in experiments/standard_qaoa.py. The count penalty at the bound itself, with no margin above it, as Niko's penalty_bound in briskit/one_hot.py. Ladder (B9): 12 damaged links only, depths 1 and 2, pair terms kept 100%, 50% and 25% (the keep setting of qollab/main.py), weak terms chosen by the magnitude of the fitted coefficient, 30 seeds. Trained flow circuit (B10): random_geometric_12_0, restore 6, seed 0, depths 1 and 2. All in briskit/flow.py, briskit/solvers.py, experiments/flow_fitted_pair_cost.py, experiments/flow_budget_experiment.py, experiments/flow_ladder.py and experiments/flow_trained_circuit.py.

## Area C: Qollab project and figures

Each source was opened and the part used was read on 2026-10-10.

Qollab, Python Qiskit projects, https://qollab.xyz/learn/docs/python-qiskit-projects. Code goes in main.py. backend exists when the code starts and is chosen in the Select QPU dialog: BasicSimulator for built-in runs, a Fake backend for IBM noise models, IonQProvider's backend for IonQ. A shots: int parameter of main() appears under Run settings. The console shows what the code prints. Used in qollab/main.py and qollab/run_local.py.

Qollab, Runtime environment, https://qollab.xyz/learn/docs/runtime-environment. Up to 100 files of at most 2 MB each. NumPy, SciPy and Matplotlib load on import. A data file loads only when its path is a plain string. main() settings must be annotated int, float, str or bool, with the signature on one line, and the runtime calls main() itself. IonQ backends allow one job per run. plt.savefig shows the figure and offers a PNG. Used in qollab/main.py and qollab/project_figures.py.

Qollab, Compute backends, https://qollab.xyz/learn/docs/compute-backends. Built-in simulator, 24 qubits. IBM noise-model simulators run in the browser. IonQ remote simulators (Aria 1, Aria 2, Forte 1, Forte Enterprise 1) are free noise models of IonQ machines, one job per run. Aria hardware is retired. Used for the runs of task C1.

Matplotlib style sheets reference, https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html. Lists the tableau-colorblind10 style but does not describe it; its colours were read from Matplotlib 3.11.2 itself. Used in qollab/project_figures.py.

Okabe and Ito, Color Universal Design, https://jfly.uni-koeln.de/color/. Do not convey information in colour only; show differences in colour and in shape (solid and dotted lines, symbols, hatching); use thicker lines and bigger symbols. The palette's RGB values are only in an image that was not read, so the Okabe-Ito colours themselves are not used. Used in qollab/project_figures.py.

W3C WAI, Complex images, https://www.w3.org/WAI/tutorials/images/complex/. A chart needs a short alt text that identifies it and says where the long description is, and a long description with the values and trends that everyone can reach. Used in qollab/project_figures.py and figures/captions.md.

RestoreBench-Q Project Plan, section 13 and section 14 step 8. Team document. Quick path of 10 minutes or less; figures with alt text, a caption that states the finding and a palette that does not rely on colour alone; tables as text. Used in qollab/main.py, qollab/project_figures.py and figures/build.py.

Matplotlib 3.11.2 was installed in the project environment for the figures. It is not in requirements.txt yet.

Project choices with no source (area C). Summary totals of 50, 100, 150 and 200 evaluations in the figure data, a subset of the totals results/budget_experiment.txt reports. Methods drawn in the budget curves: random plans, fitted cost with best first, thermal beta 2 and QAOA depth 3, plus greedy or the conversion algorithm; the rest appear in the table only. The y range of each budget panel is 1.1 times the largest drawn mean regret from 25 evaluations on. A percentage is written inside a spectrum bar when the share is 6% or more. Markers every eighth of a curve. Figures saved at 150 dots per inch on white. The figures setting of main() defaults to on. The fit size of 150 and the 50 further evaluations are read from experiments/budget_experiment.py, not set here. All in qollab/project_figures.py and figures/build.py.

Python runpy, https://docs.python.org/3/library/runpy.html. run_path's init_globals pre-populates the module's globals before the code runs, and run_path returns the module's globals. Used in qollab/run_local.py to supply backend and the figure folder as Qollab supplies backend.

Matplotlib backends, https://matplotlib.org/stable/users/explain/figure/backends.html. The MPLBACKEND environment variable selects the backend and matplotlib.use() overrides it; Agg is non-interactive and writes files only. Used in qollab/run_local.py and figures/build.py.

Project choices with no source (task C1). Runs at 100% and 50% of the pair terms, as task C1 of the division of work asks; 2000 shots and depth 2, the defaults of qollab/main.py. Record file name qollab_run_<backend>_keep<keep>_depth<depth>.txt. Local check records in results/qollab/local/, hosted records in results/qollab/. In qollab/main.py and qollab/run_local.py.

Qollab, JS Qiskit projects, https://qollab.xyz/learn/docs/js-qiskit-projects. Read 2026-10-11. A Python module is imported in main.js by dotted name with import * as (a default import gives undefined); call its functions through .callPromising, as a plain call fails with a ConnectionError on IonQ backends; return values are proxies converted with .toJs(); backend is supplied, not created; the code runs in an async context, so top-level await works. Used in qollab/main.js.

Qollab, Runtime environment, https://qollab.xyz/learn/docs/runtime-environment. Read again 2026-10-11 for pages. A project runs as a page when index.html and main.js are at the top level; index.html holds body content only and its script tags do not run; main.css is the only stylesheet loaded; IonQ backends take one job per run, and a second job stops the run. The page does not mention the template's filter of outcomes under 5%. Used in qollab/index.html, qollab/main.js and qollab/main.css.

Pyodide JavaScript API, https://pyodide.org/en/stable/usage/api/js-api.html. Read 2026-10-11. PyProxy.toJs converts deeply, and turns a dict into a Map unless dict_converter is given (Object.fromEntries gives a plain object); destroy releases the proxy; callPromising is marked experimental and needs JS Promise integration. Used in qollab/main.js.

Project choices with no source (task C2 and the ladder table). Page settings default to 2000 shots, depth 2 and 100% of the pair terms, as main.py; shots limited to 1 to 10000 in the form. Bands of cost above the best plan: best plan, up to 1%, 1% to 5%, 5% to 10%, more than 10%. A plan counts as a best plan within a relative gap of 1e-12, as in main.py. The table of plans scrolls after 28rem. The page stops after one run when the backend line names IonQ. Ladder table: the mean regret at the last total of results/flow_ladder.txt (200 evaluations) and the runs that reach regret 0; lowest regret per column in bold; methods matched to their gate counts by dropping ", best first" or ", QAOA depth N". In qollab/main.js, qollab/page_run.py, qollab/main.css, qollab/project_figures.py and figures/build.py.
