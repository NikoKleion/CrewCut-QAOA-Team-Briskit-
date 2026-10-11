# Syllabus

What each file does, how, and the source it rests on.

Layout: `briskit/` is the library. `experiments/` holds one script per result, each run as `python -m experiments.<name>` from the repository root. `run_experiments.py` runs them and saves what they print in `results/`. `checks/` holds the checks of the library, each run on the real code and on a broken version. `qollab/` is the project file for Qollab with its trained circuits. `data/flow/` holds the flow instances.

Entries are in the order of that layout, and by name within a folder.

## briskit/baselines.py

Classical methods used for comparison: descent by single bit flips, the end point of steepest descent from every plan at once, Karmarkar-Karp differencing for splitting numbers into two equal sums, longest job to the less loaded crew, a thermal sampler that draws a plan with probability proportional to exp(-beta * cost), and greedy selection by marginal gain.

Differencing: Karmarkar and Karp, The Differencing Method of Set Partitioning, UC Berkeley report CSD-83-113, https://www2.eecs.berkeley.edu/Pubs/TechRpts/1983/6353.html

## briskit/benchmark.py

The benchmark harness. A solver is a function `solve(instance, budget, rng)` that returns at most `budget` plans; it gets the instance with a counter on its scoring model and never the value table. The harness refuses plans with the wrong count and returns one row per run and budget: experiment, instance, method, budget, seed, value, regret, returned, evaluations, seconds, script. `value` is the best over every plan the run scored, the solver's own scoring included, `returned` the best of the returned plans alone, `evaluations` the number of distinct plans scored, and regret the distance from `value` to the optimum of the value table divided by that optimum. Each run's random generator is seeded with the seed, the instance name and the method name, so a row can be made again from its own columns. With `trace` the harness also writes a row each time the best value improves during a solver's own scoring, `changes_only` keeps the rows where a run's value changes, and `write_rows` leaves a saved file untouched when only the times differ. The count of distinct plans is right only for a scoring model without noise.

Evaluations as the cost of a run, and time reported apart from it: Hansen et al., COCO: The Experimental Procedure, https://arxiv.org/abs/1603.08776

Value of a run as the best plan scored so far: Hansen et al., COCO: Performance Assessment, section 2.1, https://arxiv.org/abs/1605.03560

Seeding with a list of integers: NumPy `default_rng`, https://numpy.org/doc/stable/reference/random/generator.html

Names to integers: Python `zlib.crc32`, https://docs.python.org/3/library/zlib.html

## briskit/collective.py

Builds the QAOA cost layer from collective blocks. A block with amplitudes a and strength c is exp(-i c (a . Z)^2), written as one named gate that holds an RZZ rotation of angle 4 c a_i a_j on each pair. Blocks come from the urgency levels of independent repairs (strength a quarter of the gap between levels, amplitudes the repair hours of the lines at that level or above) or from the eigenvalues and eigenvectors of any pair coefficient matrix. `collective_qaoa_circuit` differs from `qaoa_circuit` of `briskit/qaoa.py` in two ways: its cost layer is the sequence of blocks in place of one rotation per pair, and it has an XX form in which the pair rotations are RXX, the mixer is a Z rotation and a Hadamard on each qubit comes before measurement. Both give the same state. In Qiskit a block is still a set of pair gates: at 12 repairs the block layer holds 286 RZZ gates as written against 66 for the pair by pair layer, and the transpiler leaves 285, so on a public gate set the pair by pair layer is the one to send.

RZZ: https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.library.RZZGate

RXX: https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.library.RXXGate

One collective interaction of ions for number partitioning: Graß et al., https://arxiv.org/abs/1507.07863

## briskit/cost_circuit.py

Turns a fitted cost into a QAOA circuit. The cost has a constant, a term on each element and a term on each pair, on yes or no variables, in the order `fitted_merit` of `briskit/solvers.py` uses, and may carry a penalty on the number of elements chosen. `ising_terms` rewrites it with x = (1 - Z) / 2 as one coefficient per qubit and one per pair, `cost_scale` gives its standard deviation over all plans as the root of the sum of the squared coefficients, so no table of 2^n values is needed, and `cost_circuit` applies per layer RZ(2 gamma h_i / scale) on each qubit, RZZ(2 gamma J_ij / scale) on each pair and RX(2 beta) on each qubit. It differs from `qaoa_circuit` of `briskit/qaoa.py`, which takes pair terms only and a scale given from outside. The circuit reproduces the distribution the QAOA selector of `briskit/solvers.py` samples; it has been checked up to 12 qubits on Qiskit's state vector and has not been run on Qollab.

RZ: https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.library.RZGate

RZZ: https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.library.RZZGate

QAOA: Farhi, Goldstone and Gutmann, https://arxiv.org/abs/1411.4028

## briskit/dispatch.py

Computes the rho factor of each line, the largest ratio of summed weight to summed repair time over the subtrees rooted at that line, by listing every subtree, which is practical for feeders of a few dozen lines. Dispatches crews by a priority and returns when each repair is completed and when each line is live. The candidate lines are either those whose upstream line has been started, or those whose upstream node is live or whose upstream line is under repair. Also gives the share of weight restored at a given time.

Tan et al., section 5.4, equation 51, Algorithms 4 and 5, https://arxiv.org/abs/1702.08382

## briskit/feeder.py

Holds the 12 lines of the IEEE 13 node test feeder and the line upstream of each. Draws random instances (weights uniform between 0 and 1 with one weight of 5, repair times whole hours from 1 to 10), lists every plan as a row of bits, and scores a restoration plan by the total weight of the lines that are restored together with every line upstream of them.

Line list: OpenDSS model of the feeder, https://raw.githubusercontent.com/dss-extensions/electricdss-tst/master/Version8/Distrib/IEEETestCases/13Bus/IEEE13Nodeckt.dss

Instance generator and the rule that a line is live only when everything upstream is repaired: Tan et al., Scheduling Post-Disaster Repairs in Electricity Distribution Networks, sections 2 and 6, https://arxiv.org/abs/1702.08382

## briskit/flow.py

Flow networks. Stores an instance as JSON (network with node positions and link capacities, the damaged links as `elements`, commodities with source, sink, demand, priority tier and weight, `budget` as the numbers of links to restore, damage disk). Generates three families on 16 nodes (random geometric, ring with chords, hub and spoke), damages the links whose midpoints lie nearest a random centre, and draws 6 to 8 commodities in three priority tiers. Scores a plan with a multi-commodity flow linear program: each link carries flow in both directions up to its capacity, or none if it is damaged and not restored, and the program maximises the priority-weighted demand served. The value table runs that program on every plan, in the plan order of `briskit/feeder.py`, in parallel; the environment variable `FLOW_PROCESSES` caps the number of processes. Every flow script regenerates the instances it needs and rewrites a stored file that differs, and reuses a stored value table only when it carries the fingerprint of the current instance, so `run_experiments.py` can run them in any order. Files are written to a temporary name and then swapped in whole, so a script running at the same time never reads half a file. `restoration` wraps a value table as an instance of `briskit/instances.py` for the harness: restore a given number of the damaged links, maximise the value, scored by looking the plan up in the table.

Hand-made instance: links 0-1 (capacity 4), 1-2 (3), 0-3 (2), 3-2 (2), 2-4 (5); links 1-2 and 3-2 damaged; commodity 0 to 2 with demand 4 and weight 2, commodity 0 to 4 with demand 3 and weight 1. Both commodities must enter node 2 through a damaged link. Nothing restored: 0. Only 1-2: 3 units into node 2, all to the heavier commodity, 3 x 2 = 6. Only 3-2: 2 units, 2 x 2 = 4. Both: 5 units, 4 to the first commodity and 1 to the second, 4 x 2 + 1 x 1 = 9.

SciPy `linprog` with HiGHS, https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html

Multi-commodity flow, https://en.wikipedia.org/wiki/Multi-commodity_flow_problem

NetworkX random geometric graph (definition only; NetworkX is not used), https://networkx.org/documentation/stable/reference/generated/networkx.generators.geometric.random_geometric_graph.html

NetworkX Newman-Watts-Strogatz graph (definition only), https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.newman_watts_strogatz_graph.html

Python `multiprocessing`, https://docs.python.org/3/library/multiprocessing.html

Python `os.replace` (atomic on POSIX), https://docs.python.org/3/library/os.html#os.replace

Every constant without a source is listed in `SOURCES.md`.

## briskit/hardware.py

Builds one cost layer with a rotation on every pair of qubits, counts the two-qubit gates and the two-qubit depth of a compiled circuit, and gives the fraction of shots with no two-qubit error as (1 - r) to the power of the gate count, with r = 0.00495, the two-qubit rate in IonQ's Forte 1 noise model.

Error rate: IonQ, Simulation with Noise Models, https://docs.ionq.com/guides/simulation-with-noise-models

Compiling: Qiskit transpiler, https://quantum.cloud.ibm.com/docs/api/qiskit/transpiler

## briskit/instances.py

An instance is one decision on a network: a name, the number of yes or no elements, whether a plan's score is a value to raise or a cost to lower, how many elements a plan must choose, what a planner knows (weights, hours, network), and a scoring model. The scoring model is one function, `score(plans)`, that takes rows of 0 and 1 and returns one number per row; `table()` scores every plan, with element q as bit q of the row number. The three feeder decisions are built here: restoring a fixed number of lines, assigning repairs to two crews, and protecting a fixed number of lines.

The scoring rules are those of `briskit/feeder.py`, `briskit/scheduling.py` and `briskit/pair_cost.py`, with the sources listed for them above.

## briskit/ion_model.py

Simulates ions driven through one shared vibrational mode. The Hamiltonian is the sum over ions of g_i(t) X_i (a exp(-i delta t) + a^dagger exp(i delta t)), stepped in time with the mode cut off at 30 levels. Returns the operator on the ions when the mode starts and ends in its ground state, the population left in the mode, and the target operator with pair phase (2 g_i g_j / delta) min(T_i, T_j).

Interaction: Sørensen and Mølmer, Quantum computation with ions in thermal motion, https://arxiv.org/abs/quant-ph/9810039

Many ions at once, and the mode returning to its start: Lotshaw et al., Modelling noise in global Mølmer-Sørensen interactions applied to quantum approximate optimization, section II A, https://arxiv.org/abs/2211.00133

Matrix exponential: SciPy `expm`, https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html

## briskit/one_hot.py

Crew assignment in the standard form: one binary variable for each repair and crew, the schedule cost as single and pair terms, and a penalty that makes each repair sit on exactly one crew. `one_hot_terms` returns the constant, the single terms and the pair terms; `penalty_bound` is a penalty above which every lowest state is an assignment; `cost_table` gives the cost of every state; `assignments` says which states are assignments and which crew each repair is on; `one_hot_circuit` is the QAOA circuit of the form. For n repairs and two crews it takes 2n qubits and n squared pair terms, where one qubit per repair takes n qubits and n(n-1)/2 pair terms.

The program with one variable per job and machine: Skutella, Convex quadratic and semidefinite programming relaxations in scheduling, J. ACM 48(2), 2001, section 2.1; the same cost as a cut with one label per job, which for two crews is one qubit per repair: section 6 of the same paper, https://page.math.tu-berlin.de/~combi/coga-old/publications/techreports/2000/Report-683-2000.html

The penalty for "each job on exactly one machine": Lucas, Ising formulations of many NP problems, section 6.3, https://arxiv.org/abs/1302.5843

## briskit/pair_cost.py

Builds the pair coefficients of crew assignment for independent repairs: two repairs on one crew cost the weight of the later one times the hours of the earlier one. Splits those coefficients into one sum of squares per urgency level, fits a pair cost to scored plans by least squares or ridge regression, turns a pair matrix into eigenvalue blocks, expands a value table in the Walsh-Hadamard basis and reports the share of variance at each order, and gives the single and pair terms of the expected cost when lines fail independently.

The pair coefficients are checked against brute force in `experiments/crew_assignment.py`. The cost of an assignment to identical crews as a cut on a complete graph, with weight w_j p_k on the edge between repairs j and k when k comes first: Skutella, Convex quadratic and semidefinite programming relaxations in scheduling, section 6 (page 30 of the report), https://page.math.tu-berlin.de/~combi/coga-old/publications/techreports/2000/Report-683-2000.html

Walsh-Hadamard expansion and weight by degree: O'Donnell, Analysis of Boolean Functions, chapter 1, https://arxiv.org/abs/2105.10386

Least squares: NumPy `lstsq`, https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html

## briskit/qaoa.py

Computes the output probabilities of QAOA exactly from a table of the cost of every plan: each layer multiplies the state by a phase set by the cost, then rotates every qubit about X. Trains the angles with BFGS from several starting points, gives the expected best cost among a number of shots, and builds the same circuit in Qiskit with one RZZ rotation per pair.

Algorithm: Farhi, Goldstone and Gutmann, A Quantum Approximate Optimization Algorithm, https://arxiv.org/abs/1411.4028

Optimiser: SciPy `minimize` with BFGS, https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html

Circuit: Qiskit `QuantumCircuit`, https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit

## briskit/scheduling.py

Orders repairs by weight per hour, finds the optimal order for one crew on a tree by repeatedly merging the line with the largest weight per hour into the line upstream of it, and computes the cost of every two crew assignment as the sum of weight times the hour each line is live. Converts a one crew order to two crews by giving each repair to the crew that is free first, dispatches by a priority rule, solves crew assignment one urgency level at a time, and finds the proven optimum with an integer program that has one variable per line and hour.

One crew order (Algorithm 2), conversion to several crews (Algorithm 3), dispatch by priority (section 5.4, Algorithm 5) with the FirstEnergy and Edison Electric Institute priorities (section 5.5), and the integer program (section 3, used here without interrupted repairs): Tan et al., https://arxiv.org/abs/1702.08382

Integer program solver: SciPy `milp`, https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html

## briskit/solvers.py

Solvers for the harness. `random_plans` draws plans with the right count uniformly, `largest_weights` takes the largest weights, `conversion` is the conversion algorithm of Tan et al. for two crews, and `greedy` adds one element at a time, the one whose addition scores best on the true model. `fitted(samples, select)` scores a random sample of plans, fits single and pair terms to the scores by ridge regression, and passes the fitted value of every plan to a selector: `best_first` returns the best fitted plans by exact search, `thermal(beta)` draws plans with probability proportional to the exponential of beta times the standardised fitted value, and `qaoa(depth)` trains QAOA on the fitted cost with a penalty on the count and draws plans of the right count from its distribution. The penalty is the standard form A(K - number chosen)^2 with A the standard deviation of the fitted value over plans of the right count; `qaoa(depth, at_bound=True)` raises A to the bound when it is below it (`cost_at_bound`). The bound is the largest gain of a plan of another count over the best fitted plan of the right count, divided by the squared difference in count, so that at the bound the best plan of the right count is a lowest state of the cost.

The count as a squared penalty, and the condition that the penalty be large enough for the lowest states to have the right count: Lucas, Ising formulations of many NP problems, section 2.3, equations 13 and 14, https://arxiv.org/abs/1302.5843

QAOA: Farhi, Goldstone and Gutmann, https://arxiv.org/abs/1411.4028

Linear solve behind the ridge fit: NumPy `linalg.solve`, https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html

## briskit/summary.py

Reads result rows as one step curve per run. `regret_at` gives a run's regret at a total number of evaluations, `regret_summary` the median, quartiles and mean over runs, and `runtime_to_target` the number of runs that reach a regret target and the average runtime: the evaluations of the runs that missed plus the evaluations to the target of the runs that reached it, divided by the number that reached it.

Average runtime: Hansen et al., COCO: Performance Assessment, section 4, https://arxiv.org/abs/1605.03560

Median and quartiles in place of the mean alone: Doerr et al., IOHprofiler, https://arxiv.org/abs/1810.05281

## briskit/trained.py

Reads and writes the file of trained circuits, `qollab/trained_circuits.json`, and rebuilds from one record the QAOA circuit and the true value of measured plans. A record holds what the circuit needs (size, sense, count, coefficients of the fitted cost, penalty strength, angles), what scoring needs (weights and the line upstream of each line), and where it came from (experiment, instance, method, seed, number of fitting plans, depth, commit of the code). It imports NumPy, Qiskit and `json` only, so a run on Qollab trains nothing and loads no SciPy, and `within_limits` tests the platform's limits of 2 MB for a file and 24 qubits for the built-in simulator. A flow record holds the value of every plan (`values`, in the plan order of `briskit/feeder.py`) in place of the weights and the tree, and `record_value` scores from it, since the flow program needs SciPy. No record has been run on Qollab.

Files named by a plain string, and the 2 MB limit: Qollab, Runtime environment, https://qollab.xyz/learn/docs/runtime-environment

24 qubits: Qollab, Compute backends, https://qollab.xyz/learn/docs/compute-backends

JSON: Python `json`, https://docs.python.org/3/library/json.html

## briskit/tree_knapsack.py

The exact method for budgeted restoration on a tree. Restoring a fixed number of lines, where a line counts only with every line upstream of it, is the precedence constrained tree knapsack problem with the line weights as profits, cost 1 for each line and the number of lines as the capacity. `best_closed_set` fills a table over the lines in depth-first order: a line left out skips its whole subtree, a line taken moves on into it; the table has (lines) times (capacity + 1) entries and the chosen set is read back from it. Costs other than 1, such as repair hours with a budget in hours, are handled the same way. `tree_optimum` is the same method as a solver for the harness: it reads the weights and the tree from the instance and returns one plan. It applies to trees only, and to weights that are not negative.

Kumabe, Maehara and Sin'ya, Linear Pseudo-Polynomial Factor Algorithm for Automaton Constrained Tree Knapsack Problem, Example 2 (the precedence constrained problem; the paper credits the linear-time method to Johnson and Niemi and to Cho and Shaw), https://arxiv.org/abs/1807.04942

## experiments/budget_experiment.py

Runs every solver through the harness on 30 instances of two feeder decisions, restoring 6 of 12 lines and assigning 12 repairs to two crews. Methods that fit a pair cost score 150 random plans and then return up to 50 more; random plans run to the same total of 200. It writes the rows to `results/feeder/budget_experiment_restoration.csv` and `results/feeder/budget_experiment_crews.csv`, keeping a row only where a run's value changes, and prints regret at fixed totals and the average runtime to regret targets of 5%, 1% and 0.

Comparing methods only up to the smallest budget any of them was given, and instances as repetitions: Hansen et al., COCO: The Experimental Procedure, https://arxiv.org/abs/1603.08776

Instance generator: Tan et al., section 6, https://arxiv.org/abs/1702.08382

## experiments/budgeted_restoration.py

Restore 8 of 18 lines on random trees. A pair cost is fitted on 600 random sets of the right size and a penalty on the count is added. QAOA on that cost, with shots of the wrong count discarded, is compared with a thermal sampler at the same mean value, random sets, greedy selection, and exact search of the fitted cost.

The decision of which damaged links to restore within a budget: Udekwe et al., Q-RESTORE, https://arxiv.org/abs/2501.11197

Farhi, Goldstone and Gutmann, https://arxiv.org/abs/1411.4028

## experiments/collective_layer_counts.py

Counts, for one cost layer of 12 repairs, the collective blocks, the RZZ gates they hold as written, and the RZZ gates left after Qiskit compiles the layer.

Qiskit transpiler, https://quantum.cloud.ibm.com/docs/api/qiskit/transpiler

## experiments/crew_assignment.py

Checks the one crew order and the pair cost against brute force. On the IEEE 13 node feeder with two crews it compares the conversion algorithm and the best plan of the closed-form pair cost with the best of all 4,096 assignments, solves 12 instances to proven optimality, and runs QAOA on the closed-form pair cost.

Tan et al., sections 3, 5 and 6.1, https://arxiv.org/abs/1702.08382

## experiments/dispatch_game_instance.py

Fixes the instance of the dispatch game in `qollab/game/` and trains its two circuits. The instance is the first of the 30 instances of 8 repairs of `experiments/standard_qaoa.py`; the script counts on how many of the 30 the simple dispatch rule is already a best plan. It trains one qubit per repair and the standard form at four penalties with that experiment's trainer and streams, at two layers, picks the standard form's penalty by the share of best plan shots, which uses the answer and favours the standard form, prints the exact shares of both, and writes `qollab/game/game_instance.py`.

Farhi, Goldstone and Gutmann, https://arxiv.org/abs/1411.4028 ; Skutella, sections 2.1 and 6; Lucas, section 6.3, https://arxiv.org/abs/1302.5843

## experiments/dispatch_readings.py

Dispatches 300 feeder instances by the rho factor, the FirstEnergy rule and the Edison Electric Institute rule under both readings of the candidate set of Algorithm 5, and reports how often the schedules differ, the customer-hours above the best assignment, and the share of weight restored halfway through, the quantity plotted in Figure 4 of the paper.

Tan et al., Algorithm 5 and Figure 4, https://arxiv.org/abs/1702.08382

## experiments/dispatch_rules.py

Customer-hours without power on 300 feeder instances under the FirstEnergy rule (largest weight first), the Edison Electric Institute rule (largest weight per hour first) and the conversion algorithm, each against the best two crew assignment.

Tan et al., sections 5.4, 5.5 and 6.1, https://arxiv.org/abs/1702.08382

## experiments/feeder_fitted_models.py

Restoring a fixed number of the 12 feeder lines: largest weights first, greedy selection, and fitted costs with and without pair terms, each against the optimum found by enumeration. Then the fitted pair cost of crew assignment is cut down to its largest eigenvalue blocks and the best plan of each cut is scored.

The decision of which damaged links to restore within a budget: Udekwe et al., Q-RESTORE, https://arxiv.org/abs/2501.11197

## experiments/fit_size.py

Regret against the number of plans used for the fit, for restoring 4 and 6 of 12 feeder lines on 30 instances. For fits on 20 to 300 random plans and on every plan of the allowed size, with single and pair terms and with single terms only, it prints the regret of the fitted cost's own best plan, and the regret of the run after the fit and 10 more plans beside random plans given the same total. Rows go to `results/feeder/fit_size_restore_4.csv` and `results/feeder/fit_size_restore_6.csv`.

Linear solve behind the ridge fit: NumPy `linalg.solve`, https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html

Second route used to check it: NumPy `lstsq`, https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html

## experiments/fitted_pair_cost.py

Fits a pair cost to the true cost on a tree from randomly chosen plans. Compares the best plan of the fit, and QAOA shots drawn from it, with random plans, bit-flip descent, the conversion algorithm and the plans used for the fit, on the feeder with 12 repairs and on random trees with 18.

NumPy `lstsq`, https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html

Farhi, Goldstone and Gutmann, https://arxiv.org/abs/1411.4028

## experiments/flow_budget_experiment.py

The budget experiment of `experiments/budget_experiment.py` on the flow networks (E8): every instance, each number of links in its `budget`, 30 seeds. Methods that fit a pair cost score 150 random plans of the right size (12 damaged links) or 400 (16), then return up to 50 more by best first, thermal sampling at beta 2 and 5, or QAOA at depth 1 and 2. Random plans and swap annealing on the true values run to the same total; greedy runs once. Swap annealing starts from a random plan of the right size, swaps one restored link for one unrestored link each step, and accepts a worse plan with probability exp(-loss / (T x current value)), T falling geometrically from 0.1 to 0.005 over the run; every plan it scores is returned. Rows go to `results/flow/budget_experiment.csv`, keeping a row only where a run's value changes, and it prints regret at fixed totals and the average runtime to regret targets of 5%, 1% and 0 for each number of damaged links and number restored. The QAOA methods use the count penalty in the standard form with the penalty raised to the bound (`qaoa(depth, at_bound=True)` in `briskit/solvers.py`): with the feeder's penalty, the standard deviation of the fitted value, under 1% of the probability fell on plans of the right count on some flow instances, because restoring more links always serves more demand. A mixer that keeps the count would remove the penalty and would be a rung of its own. Jobs run in parallel, one per instance, number restored and seed, on at most 6 processes and at most `FLOW_PROCESSES`, since each loads Qiskit.

The count penalty: Lucas, section 2.3, https://arxiv.org/abs/1302.5843

Comparing methods only up to the smallest budget any of them was given: Hansen et al., COCO: The Experimental Procedure, https://arxiv.org/abs/1603.08776

Simulated annealing (the acceptance rule exp(-(e' - e)/T) and lowering T over the run; the geometric schedule and its end points are project choices), https://en.wikipedia.org/wiki/Simulated_annealing

## experiments/flow_fitted_pair_cost.py

On the flow tables, restore each number of damaged links in the instance's `budget`. Each instance and number restored has its own generator seeded from `SEED` and its name, so adding or removing an instance leaves the samples of the others unchanged. Fits a cost with single-link terms only (the control) and with single and pair terms, on all plans of the right size (E3) and on 10 to 320 random plans (E4); both fits use the same sample in each draw. Each fit gives two rows in `results/flow/fitted_pair_cost.csv`: budget 0, the fitting sample alone, and budget 1, the sample plus the plan the fit returns. `value` is the best plan scored in the run, `returned` the plan the fit returns, `evaluations` the distinct plans scored, `seconds` the time of the fit. The instance column carries the number restored, for example `ring_with_chords_12_0_restore4`, and `seed` is the draw. With all plans in the fit (E3) every plan is scored, so only `returned` is informative there. A number restored for which no plan serves any demand is skipped and named in the output, since regret divides by the optimum.

Udekwe et al., Q-RESTORE, https://arxiv.org/abs/2501.11197

Ridge fit: `briskit/pair_cost.py`.

## experiments/flow_instances.py

Writes the hand-made instance and the 18 generated instances (three families, 12 and 16 damaged links, seeds 0 to 2) to `data/flow/`.

## experiments/flow_ladder.py

Rungs 2 and 4 of the ladder of the plan addendum (section 5.2) on the flow networks (B9), through the harness. On the nine instances with 12 damaged links, each number of links in their `budget`, 30 seeds, the cost is fitted on 150 plans of the right size, as in the budget experiment, and QAOA at depths 1 and 2 runs on: single-link terms only (the control), single and pair terms (rung 2), and single and pair terms with only the 50% or 25% of pair terms of largest magnitude kept (rung 4, as in the `keep` setting of `qollab/main.py`). Exact search on each fit (best first) and random plans give the classical references. The count penalty is at the bound, as in `experiments/flow_budget_experiment.py`; plans are always scored on the true value table. Rows go to `results/flow/ladder.csv`; it prints regret at fixed totals and runtime to regret targets, and the pair gates per cost layer of each fit with and without the count penalty, counted from the circuit's terms on a fit of each instance. Instances with 16 links are left out: QAOA at depth 2 on 16 qubits takes about 12 seconds per run, against well under one at 12. Rungs 0, 1 and 3 do not apply here: no standard formulation with penalty terms exists for the flow networks (addendum, section 6.2), and connectivity is compared in area A.

Comparing methods only up to the smallest budget any of them was given: Hansen et al., COCO: The Experimental Procedure, https://arxiv.org/abs/1603.08776

The count penalty: Lucas, section 2.3, https://arxiv.org/abs/1302.5843

## experiments/flow_model_checks.py

Checks the flow program: the hand-made instance gives 0, 6, 4 and 9; restoring every damaged link gives the value of the undamaged network; for 200 random pairs of plans, adding links never lowers the value. Writes the time of one program to `results/flow/program_time.csv`, so the printed output is the same on every run.

## experiments/flow_trained_circuit.py

One flow instance in the file of trained circuits (B10), in the format of `experiments/trained_circuits.py`: restore 6 of the 12 damaged links of `random_geometric_12_0`, pair cost fitted on 150 plans, seed 0, QAOA at depths 1 and 2 with the count penalty at the bound. It replays the random stream of the budget experiment's QAOA runs, so the fit and the angles are the ones behind the saved E8 rows for that instance and seed. The record carries the value of every plan in place of the weights and the tree, so a run on Qollab can score measured plans without the flow program. It replaces only its own records (experiment `E8`) in `qollab/trained_circuits.json`. `trained_circuits.py` and this script both rewrite that file, so run them one after the other, not in the same `run_experiments.py --jobs` batch.

Farhi, Goldstone and Gutmann, https://arxiv.org/abs/1411.4028

## experiments/flow_value_tables.py

Exact value table of every flow instance (E1), stored in `results/flow/tables/<instance>.npz` with the instance fingerprint and the time it took, in the plan order of `briskit/feeder.py`; the times are copied to `results/flow/value_tables.csv`. Checks on each table that the last plan equals the undamaged value and that restoring any one more link never lowers the value, over every plan, and counts the idle links, damaged links whose restoration never changes the value.

## experiments/gate_counts.py

Compiles one cost layer with a term on every pair, once for a device where any two qubits interact with RZZ as a basis gate and once for a heavy-hex coupling map with CZ, and reports two-qubit gate count and depth.

Qiskit transpiler, https://quantum.cloud.ibm.com/docs/api/qiskit/transpiler

Qiskit `CouplingMap`, https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.transpiler.CouplingMap

IonQ native gates, https://docs.ionq.com/guides/getting-started-with-native-gates

## experiments/independent_repairs.py

Repairs with no network between them, where the pair cost is exact. Compares QAOA at depths 1 to 3 with random plans given the same number of shots, and with the rule that takes repairs in order of weight per hour and gives each to the crew that is free first, at 12 and 18 repairs.

Farhi, Goldstone and Gutmann, https://arxiv.org/abs/1411.4028

Instance generator: Tan et al., section 6, https://arxiv.org/abs/1702.08382

## experiments/interaction_spectrum.py

Expands the full value table of each decision on the feeder in the Walsh-Hadamard basis and reports the share of the non-constant variance at orders 1 to 4 and above.

O'Donnell, Analysis of Boolean Functions, chapter 1, https://arxiv.org/abs/2105.10386

## experiments/ion_collective_interaction.py

Three ions and one mode. Compares the simulated operator with the target pair phases for switch-off times that are whole and non-whole numbers of periods, with the phase of one laser flipped, and with the drive on the other side of the mode frequency.

Sørensen and Mølmer, https://arxiv.org/abs/quant-ph/9810039

Lotshaw et al., section II A, https://arxiv.org/abs/2211.00133

## experiments/level_merging.py

Rounds each urgency ratio up to a geometric grid, which leaves fewer levels, and measures the rounded cost over the true cost for every plan and the loss of the plan the rounded problem selects.

The cost of `briskit/pair_cost.py` does not decrease when a ratio is raised and scales with the ratios, which gives the bound. Its check is the enumeration in this file.

## experiments/priority_ablation.py

Priority weights on and off, on the flow instances (E10). For each instance and number of links restored it scores every plan of that size twice, with the commodity weights of the instance and with every weight set to 1, and compares the two sets of optimal plans: how many there are, whether they share a plan, and how much weighted value is lost by planning without weights, for the best and the worst plan among the unweighted optima. It covers every number of links from 1 to one less than the damaged count, reads the weighted tables from `briskit/flow.py` and stores the unweighted ones beside them as `<instance>_unweighted`. Each of the three plans (weights on, and the best and worst unweighted optimum under the real weights) is a harness row in `results/flow/priority_ablation.csv`, with the instance seed as `seed`; the tie counts and losses go to `results/flow/priority_ablation_ties.csv`.

Flow program: `briskit/flow.py`, with the sources listed there.

## experiments/protection_and_levels.py

Checks the closed form for the expected cost of leaving lines unprotected against the expectation over every set of failed lines. Then solves crew assignment one urgency level at a time, carrying the difference in hours between the crews to the next level, and compares with the optimum and the classical rule.

The closed form is the pair cost of `briskit/pair_cost.py` averaged over independent failures. Its check is the enumeration in this file.

## experiments/qaoa_limits.py

Four measurements on independent repairs: one urgency level against Karmarkar-Karp differencing; QAOA shots as starting points for descent; pair terms dropped from the cost layer under a noise model in which a shot is unaffected with probability (1 - r) to the power of the gate count and random otherwise; and the chance of the best plan against the number of repairs. The optimiser stops after 40 iterations per start.

Karmarkar and Karp, https://www2.eecs.berkeley.edu/Pubs/TechRpts/1983/6353.html

IonQ, Simulation with Noise Models, https://docs.ionq.com/guides/simulation-with-noise-models

## experiments/samplers_and_protection_rule.py

On the fitted pair cost of the feeder, compares QAOA shots with a thermal sampler at the same mean cost, the thermal sampler at its best temperature, and the exact list of lowest-cost plans. Then counts how often choosing lines to protect by their single-line terms differs from the optimum of the expected cost.

Farhi, Goldstone and Gutmann, https://arxiv.org/abs/1411.4028

## experiments/simulator_agreement.py

Compares the probabilities from the table-based simulator in `briskit/qaoa.py` with those of the Qiskit circuit, by total variation distance, at 10 qubits and depths 1 to 3.

Qiskit `Statevector`, https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.Statevector

## experiments/standard_qaoa.py

Compares the two forms of crew assignment for two crews under QAOA with the same trainer, depths and number of starts: one qubit per repair, and one qubit per repair and crew with a penalty. The penalty is run as one number for every repair at 1, 1/2, 1/4 and 1/8 of the bound and at 2 and 4 times the bound, and as a number for each repair at its own bound and at half of it; at 8 repairs only the bound, 2 and 4 times the bound and each repair's own bound are run. For 6 and 8 repairs and 30 instances each it reports qubits, pair gates per layer, the share of shots that are an assignment, the share that are a best plan, the shots needed for a 95% chance of a best plan, and the share of the gain over random plans kept among the valid shots. The best penalty is picked for each instance by the share of best plan shots, which uses the answer and favours the standard form. Instances are spread over the worker processes the runner allows, at most 6; the output does not depend on their number.

Farhi, Goldstone and Gutmann, https://arxiv.org/abs/1411.4028 ; Skutella, sections 2.1 and 6; Lucas, section 6.3, https://arxiv.org/abs/1302.5843

## experiments/standard_qaoa_10.py

The comparison of `experiments/standard_qaoa.py` at 10 repairs, where the standard form takes 20 qubits, on 10 instances with the penalties run at 8 repairs. Ten instances and not 30 because one circuit evaluation takes 48 times as long as at 8 repairs.

Farhi, Goldstone and Gutmann, https://arxiv.org/abs/1411.4028 ; Skutella, sections 2.1 and 6; Lucas, section 6.3, https://arxiv.org/abs/1302.5843

## experiments/trained_circuits.py

Replays the random stream of the budget experiment's QAOA runs on its first instance (restore 6 of 12 lines, pair cost fitted on 150 plans, depths 1 to 3), so the fit and the angles are the ones behind the saved benchmark rows, and writes them to `qollab/trained_circuits.json`. Prints for each depth the share of probability on plans with the right count, the share of those on the optimum, and their mean regret. The file is rewritten only when one of its records changes, and records of other experiments in the file are kept.

Farhi, Goldstone and Gutmann, https://arxiv.org/abs/1411.4028

## experiments/tree_optimum.py

Runs the exact tree method through the harness on the 30 feeder instances of the budget experiment, for 4 and for 6 lines restored, and reports on how many it returns the optimum and how many plans it scored. Then solves a random tree of 2,500 lines, for which no value table can be built, for two numbers of lines and for a budget in repair hours.

Kumabe, Maehara and Sin'ya, https://arxiv.org/abs/1807.04942

## experiments/urgency_levels.py

Checks by enumeration that the pair matrix of crew assignment is a sum of one outer product per urgency level, that its rank is the number of levels, and that the cost is the matching sum of squares. Then compares, by state fidelity, a cost layer made of one diagonal operation per level with the layer made of one RZZ rotation per pair.

Number partitioning as one collective interaction of ions: Graß et al., Quantum annealing for the number partitioning problem using a tunable spin glass of ions, https://arxiv.org/abs/1507.07863

Qiskit `DiagonalGate`, https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.library.DiagonalGate

Qiskit `Statevector`, https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.Statevector

## run_experiments.py

Runs every script in `experiments/` in its own process with a fixed number of numerical threads and writes what each prints to `results/<name>.txt`. With `--check` it runs them again and reports whether each output equals the saved one. `--jobs` sets how many run at once, `--threads` the thread count, and names after the command select scripts.

    python run_experiments.py
    python run_experiments.py --check
    python run_experiments.py --check dispatch_rules level_merging

Every script seeds NumPy's generator with the constant `SEED` at the top of the file, so a rerun draws the same instances: https://numpy.org/doc/stable/reference/random/generator.html

## checks/

`benchmark_checks.py`, `budget_checks.py`, `collective_checks.py`, `cost_circuit_checks.py`, `dispatch_checks.py`, `dispatch_game_checks.py`, `fit_checks.py`, `flow_trained_checks.py`, `one_hot_checks.py`, `solver_checks.py`, `standard_qaoa_checks.py`, `summary_checks.py`, `syllabus_checks.py`, `trained_checks.py` and `tree_knapsack_checks.py` hold one named function per check, each taking the piece under test and returning what it expected, what it got and whether they agree. `record.py` runs every check twice, on the real code and on a broken version written beside it, and writes both outcomes to a `.csv` file next to the checks: a check counts only if it passes on the real code and fails on the broken one.

`budget_checks.py` audits the saved rows of the budget experiment against the exact expectation for the best of a number of distinct random plans. `fit_checks.py` compares the fit with a second way of solving the same least squares problem and with the saved result of `experiments/feeder_fitted_models.py`. `dispatch_checks.py` compares the rho factors with a second listing of subtrees by bit masks, the dispatch with `priority_dispatch`, and both with Theorem 5 of Tan et al.: with rho factors as priorities the dispatch must equal the conversion algorithm on every one of 300 feeder instances. `collective_checks.py` compares a block with the matrix exponential of its definition, the block layers with the pair by pair layer by `Operator.equiv`, the whole circuit with the table simulator, and the XX form with the ZZ form. `cost_circuit_checks.py` compares the rewritten terms with the cost table of the QAOA selector, the scale with the standard deviation of that table, and the circuit's probabilities with `qaoa_probabilities` at 8 and 12 qubits and depths 1 to 3, and with the distribution `train_qaoa` returns for a fit on the feeder. `trained_checks.py` checks that a record read back from the file equals the one written, that its circuit gives the trained distribution, that the plans drawn after replaying the stream are the plans the harness's solver returns for the same seed, instance, method and budget, that scoring from the record equals the feeder model, and that the size limits are tested. `flow_trained_checks.py` does the same for the flow record of `experiments/flow_trained_circuit.py`: its circuit gives the trained distribution, the plans drawn after replaying the stream are the plans of the budget experiment's QAOA solver with the penalty at the bound, and scoring from the stored values equals the flow table. `tree_knapsack_checks.py` compares the exact tree method with enumeration on 60 feeder cases and 10 random trees of 16 lines, and with brute force over closed sets for a budget in repair hours, and checks that the plan it returns has the right number of lines when weights are zero. `one_hot_checks.py` compares the standard form on every valid state with the schedule run crew by crew for two and three crews, checks that the penalty moves only the states that are not assignments, that the lowest state at the bound is an assignment with the optimum, that the form agrees plan by plan with one qubit per repair, that its circuit gives the distribution of the table simulator, and that qubits and pair terms follow the closed form. `standard_qaoa_checks.py` audits the saved result of `experiments/standard_qaoa.py` at 6 repairs by a second route: the same trained angles, but the distribution from the Qiskit state vector of the circuit, validity and crews read from each bit string, and the cost of a plan from the schedule of each crew; it also checks on the saved results at 6, 8 and 10 repairs that the standard form at 4 times the bound returns best plans at the rate of random valid plans. `dispatch_game_checks.py` compares the game's Python in `qollab/game/` with the library: the cost of all 256 plans, both circuits against the table simulator at the saved angles, the decoder on all 65,536 states, the saved exact shares, and the count of instances on which the simple rule is best. `syllabus_checks.py` checks that every code file in the repository has an entry in this file, and that no paragraph and no command of it is there twice.

    python -m checks.benchmark_checks
    python -m checks.budget_checks
    python -m checks.collective_checks
    python -m checks.cost_circuit_checks
    python -m checks.dispatch_checks
    python -m checks.dispatch_game_checks
    python -m checks.fit_checks
    python -m checks.flow_trained_checks
    python -m checks.one_hot_checks
    python -m checks.solver_checks
    python -m checks.standard_qaoa_checks
    python -m checks.summary_checks
    python -m checks.syllabus_checks
    python -m checks.trained_checks
    python -m checks.tree_knapsack_checks

Theorem 5: Tan et al., sections 5.3 to 5.5, https://arxiv.org/abs/1702.08382

`Operator.equiv`: https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.Operator

Success probability and repetitions to a target: Rønnow et al., Defining and detecting quantum speedup, https://arxiv.org/abs/1401.2910

## qollab/main.py

The file Qollab runs. Builds the QAOA circuit for one 12 repair instance with angles trained beforehand, compiles it for the `backend` Qollab supplies, sends one job, and scores the measured plans with the classical cost table. The `keep` setting (100, 50 or 25) is the percentage of pair terms left in the cost layer, largest first, with angles trained for each; scoring always uses the full cost. `run_circuit` does one run and returns its report as lines of text: backend (with the IonQ noise model when the backend names one), UTC date, shots, depth, share of pair terms kept, gate counts before and after compiling, the plan quality figures and the seconds taken, so the same function can serve the console and a page. Every run is saved as a record, `qollab_run_<backend>_keep<keep>_depth<depth>.txt`, headed by the project file name. On Qollab `sendFile` places the record in the console to download; these downloaded records are the outputs of task C1, kept in `results/qollab/`. Locally the record is written to the folder `run_local.py` supplies. With the `figures` setting on (the default) it then reads `figure_data.json` and draws the four figures of `qollab/project_figures.py` with their captions, alt text and tables. That is the quick path of section 13 of the plan: cached results, the final circuit on the chosen backend, the figures regenerated. It prints the time spent on the circuit and scoring, on the figures, and in total. The settings are annotated `int` and `bool` because Qollab builds its settings form only from `int`, `float`, `str` and `bool` and does not call `main()` otherwise. The data file is opened by a plain string path because Qollab loads only files named that way. The files to upload to the Qollab project are `main.py`, `project_figures.py` and `figure_data.json`. `run_with_plans` is `run_circuit` plus every measured plan, as crew letters, shots, cost and the share above the best plan, most frequent first; the page of task C2 uses it through `qollab/page_run.py`.

Qollab, Python Qiskit projects, https://qollab.xyz/learn/docs/python-qiskit-projects

Qollab, Runtime environment (settings form, plain string paths, one job per run on IonQ, figures shown in the console, `sendFile`), https://qollab.xyz/learn/docs/runtime-environment

Qollab, Compute backends, https://qollab.xyz/learn/docs/compute-backends

## qollab/project_figures.py

Draws the four static figures of section 13 of the plan from the cached data in `figure_data.json`: the pipeline, the interaction spectrum, the budget curves and the gate counts. When the data holds the ladder it adds a fifth, the ladder on the flow networks (task B9) as a table drawn as a figure: for each method the mean regret and the runs that reach the optimum at each restore count, and the pair gates per cost layer with and without the count penalty. The lowest regret in each column is in bold, and the caption's comparison of the best QAOA rung with best-first selection is computed, ties included. Each figure comes with a caption whose numbers are computed from the cached data, a short alt text, and a long description as a text table, as the W3C guidance for complex images asks. Colours come from Matplotlib's `tableau-colorblind10` style, and every series also differs by hatching, marker or line style, so no figure depends on colour alone. The budget curves show the mean regret at every total from 1 to 200 evaluations for random plans, the fitted cost with best-first selection, the thermal sampler at beta 2, QAOA at depth 3 and the rule of each decision. The table gives every method, including largest weights, which is off the chart at 64% mean regret. On Qollab `savefig` shows each figure in the console and offers the PNG; locally the figures are written to a folder. It imports Matplotlib only, which Qollab loads on import without a `requirements.txt` entry.

Matplotlib style sheets reference (`tableau-colorblind10`), https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html

Okabe and Ito, Color Universal Design: show differences in colour and in shape (line style, symbols, hatching), https://jfly.uni-koeln.de/color/

W3C WAI, Complex images: short alt text and a long description, https://www.w3.org/WAI/tutorials/images/complex/

Qollab, Runtime environment (Matplotlib loads on import; `plt.savefig` shows the figure and offers a PNG), https://qollab.xyz/learn/docs/runtime-environment

## qollab/page_run.py

The Python behind the page of task C2. `run_page(device, shots, depth, keep)` checks the settings, does one run with `run_with_plans` of `main.py`, and returns a dict with the report lines, every measured plan, the record name, the record text headed by the project file names, and whether the backend is IonQ's, which takes one job per run.

## qollab/main.js

The page of task C2, with `qollab/index.html` (the body: settings form, report, tables) and `qollab/main.css`. Qollab runs a project as a page when `index.html` and `main.js` are at its top level, and then does not run `main.py` by itself. The page waits for the Run button, reads shots, depth and the share of pair terms kept, and calls `run_page` through `.callPromising` with the `backend` Qollab supplies, the call the docs require on IonQ backends. It shows the report, a link to download the record, the shots in five bands of cost above the best plan, and every measured plan with no outcome left out: the template's filter of outcomes under 5% is not used. On IonQ it stops after one run; on other backends it waits for the next press. The record is also printed to the console. Tested in Node with a stand-in page and a real simulator result, not yet on Qollab.

Qollab, JS Qiskit projects (`import * as` a Python module by dotted name, `.callPromising`, top-level `await`), https://qollab.xyz/learn/docs/js-qiskit-projects

Qollab, Runtime environment (`index.html` and `main.js` at the top level make a page; body only; `main.css`; one job per run on IonQ), https://qollab.xyz/learn/docs/runtime-environment

Pyodide JavaScript API (`toJs` turns a dict into a `Map` unless `dict_converter` is given; `destroy`), https://pyodide.org/en/stable/usage/api/js-api.html

## qollab/run_local.py

Runs `main.py` on a laptop the way Qollab does. It supplies Qiskit's `BasicSimulator`, or an IBM fake backend, as `backend`, works from the `qollab/` folder so `figure_data.json` is found by the same plain path, and writes the figures to `figures/quick_path/`, which git ignores. Options: `--shots` (2000), `--depth` (2), `--keep` with one or more shares of pair terms (100), `--devices` with `basic` and any IBM fake backend names such as `fez` (`basic`), `--no-figures`, and `--save`, which writes each run's record to `results/qollab/local/` (ignored by git). These local records check the code before the hosted runs and are not the outputs of task C1. The figures are drawn on the first run only. It prints the wall time of each run including backend setup.

    python qollab/run_local.py --devices basic fez --keep 100 50 --no-figures --save

Python `runpy.run_path` and `init_globals`, https://docs.python.org/3/library/runpy.html

Qollab, Python Qiskit projects, https://qollab.xyz/learn/docs/python-qiskit-projects

Qiskit `BasicSimulator`, https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.providers.basic_provider.BasicSimulator

## qollab/figure_data.json

The cached data behind the figures, written by `figures/build.py`. It holds the interaction spectrum and the gate counts as printed in `results/interaction_spectrum.txt` and `results/gate_counts.txt`. For each method of the budget experiment it holds the mean regret at every total from 1 to 200 evaluations, and the median, quartiles and mean at 50, 100, 150 and 200, computed with `briskit/summary.py` from the row files in `results/feeder/`. From `results/flow_ladder.txt` it holds, for each restore count, every method's mean regret at each total, the runs that reach the optimum, and the pair gates per cost layer. Each part records its source file and the commit of the code that made it.

## figures/build.py

The full path for the figures, run after `run_experiments.py` as `python -m figures.build`. Reads the result files, writes `qollab/figure_data.json` only when it changes, and draws the figures into `figures/` (the four of section 13 and the ladder table, `flow_ladder.png`) with `figures/captions.md`, which holds each figure's alt text, caption and description for the project page. It fails if a result file no longer has the expected lines.

## qollab/game/

A second Qollab project, complete in this folder: a game in which the player splits 8 repairs between two crews and the page then shows brute force, standard QAOA and one qubit per repair on the same instance. `game_instance.py` holds the instance, the trained angles and the exact shares, written by `experiments/dispatch_game_instance.py`. `dispatch_game.py` is the Python the page calls and imports nothing from `briskit/`, because Qollab has only the project's files: the cost of every plan in the pair form, both circuits, the decoder of the standard form's 16 bits, `setup` and `run_circuits`. On a local simulator `run_circuits` sends two jobs; on an IonQ backend it sends the circuit with one qubit per repair only, because Qollab allows one job in a run there. `index.html` is the page body, `main.css` its style, `main.js` the game: it scores all 256 plans itself by running each crew's schedule, compares them with the Python costs and shows the outcome of that comparison, draws the crews' timelines, and after the player's plan is locked shows the best plan, the player's and the simple rule's place, and the circuits' shots. Compiling and backend name as in `qollab/main.py`.

Qollab, JS / Qiskit projects, https://qollab.xyz/learn/docs/js-qiskit-projects ; one job in a run on IonQ backends: Qollab, the runtime environment, https://qollab.xyz/learn/docs/runtime-environment ; times of brute force on the page: `future/timing/search_and_training_times.txt`

## data/flow/

The flow instances as JSON, one file per instance: the hand-made one and 18 generated ones (three families, 12 and 16 damaged links, seeds 0 to 2). `experiments/flow_instances.py` writes them and `briskit/flow.py` gives the format.

## future/

Simulations run outside `run_experiments.py`, one folder for each task, each script named for what it measures. A script with (WIP) in its first line has run on a few instances only. `saved_run.py` runs a script and saves what it prints beside it as a `.txt` with the code version, also when the script fails partway (`saved_run_checks.py`); `crew_pairs.py` holds the pair cost of crew assignment the scripts share. `A8_noise_study/` applies the depolarisation channels of IonQ's noise model after every gate of the circuit as written, by sampling error patterns and averaging the outcome distribution (`pauli_trajectories.py`); `trajectory_checks.py` compares it with the table simulator at zero error and with the channel's Kraus operators applied to the density matrix of 3 qubits; `sparse_layer_under_noise.py` then scores the sparse cost layer at the Aria 1 rates and at the Forte 1 rates of the model IonQ replaced in September 2025. Noise model and rates: IonQ, Simulation with Noise Models, https://docs.ionq.com/guides/simulation-with-noise-models. `A11_ladder/` scores the sparse cost layer without errors over 30 instances and counts the two-qubit gates of both forms of crew assignment on a heavy hex layout. `timing/` measures wall clock time on one machine for exhaustive search, one dispatch plan and the training of the circuit's angles at 12 repairs, and for exhaustive search at 12 to 20 repairs; its numbers differ from run to run. Run as `python future/<folder>/<script>.py` from any folder and commit the `.txt` it saves.

## results/

`results/<name>.txt` is what `experiments/<name>.py` printed, one file per script. The lines before the first blank line record how the output was made: the command, the commit of the code and whether the code had uncommitted changes, the Python and package versions, the platform, the thread count, the start time in UTC, the run time in seconds and the exit code. The rest is the output. Row files are grouped by network: `results/feeder/` for the feeder experiments and `results/flow/` for the flow networks.

Package versions are pinned in `requirements.txt`. The saved results were made with one thread. The depth 3 figures of `qaoa_limits.py` depend on the thread count, because its optimiser stops after 40 iterations and the last bits of a sum depend on how it is split: with 14 repairs the chance of the best plan is 7.2 times random at one thread and 7.5 at two (growth per added repair x1.105 and x1.111), and the best of 100 shots at one urgency level is 0.0856 hours at one and two threads and 0.0854 at the machine's default.

## SOURCES.md

Sources for the flow networks, with what was read in each and the constants that have no source.

## docs/crewcut_vs_standard_qaoa.md

The write-up comparing one qubit per repair with standard QAOA, the dispatch rules and the classical methods; its Section 8 maps each claim to the result file it is read from. `crewcut_vs_standard_qaoa.pdf` is the same text as a PDF.

## docs/CrewCut-QAOA-pamphlet.pdf

The pamphlet: the project in 17 pages for a reader who has ten minutes. The problem, what a percent is worth, the classical baseline, QAOA in one page, how a plan becomes qubits, what CrewCut does differently, the results with the setup behind each number, what could matter beyond the hackathon, what a quantum advantage would take, and the limits. Its numbers are those of the result files named in `docs/crewcut_vs_standard_qaoa.md`; the dollar figures are those of `docs/money/`.

## docs/money/

The business figures: what weather outages cost and what one percent of that is worth. `inputs.csv` holds each number taken from a source or from a result file, with its unit and where it comes from. `value.py` does the arithmetic, run as `python docs/money/value.py`, and writes `value.csv`: name, low, high, unit and formula. `sources.md` lists the six sources with the sentence used from each and the assumptions under which percentage points of the test feeder are read as dollars. The dollar figures show scale and are not a forecast.

## Demo_PublicFacing/

The public demo. `crewcut_lab.html` is the page; `sim.py` is the exact state-vector simulation of one seeded 6-repair instance behind its probability maps, run as `python Demo_PublicFacing/sim.py`. It uses NumPy and SciPy only and is not the research code: its numbers are illustrative, and the page marks which figures come from the paper.
