# Syllabus

What each file does, how, and the source it rests on.

## briskit/feeder.py

Holds the 12 lines of the IEEE 13 node test feeder and the line upstream of each. Draws random instances (weights uniform between 0 and 1 with one weight of 5, repair times whole hours from 1 to 10), lists every plan as a row of bits, and scores a restoration plan by the total weight of the lines that are restored together with every line upstream of them.

Line list: OpenDSS model of the feeder, https://raw.githubusercontent.com/dss-extensions/electricdss-tst/master/Version8/Distrib/IEEETestCases/13Bus/IEEE13Nodeckt.dss

Instance generator and the rule that a line is live only when everything upstream is repaired: Tan et al., Scheduling Post-Disaster Repairs in Electricity Distribution Networks, sections 2 and 6, https://arxiv.org/abs/1702.08382

## briskit/scheduling.py

Orders repairs by weight per hour, finds the optimal order for one crew on a tree by repeatedly merging the line with the largest weight per hour into the line upstream of it, and computes the cost of every two crew assignment as the sum of weight times the hour each line is live. Converts a one crew order to two crews by giving each repair to the crew that is free first, dispatches by a priority rule, solves crew assignment one urgency level at a time, and finds the proven optimum with an integer program that has one variable per line and hour.

One crew order (Algorithm 2), conversion to several crews (Algorithm 3), dispatch rules (section 5.4 and Algorithm 5) and the integer program (section 3, used here without interrupted repairs): Tan et al., https://arxiv.org/abs/1702.08382

Integer program solver: SciPy `milp`, https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html

## briskit/pair_cost.py

Builds the pair coefficients of crew assignment for independent repairs: two repairs on one crew cost the weight of the later one times the hours of the earlier one. Splits those coefficients into one sum of squares per urgency level, fits a pair cost to scored plans by least squares or ridge regression, turns a pair matrix into eigenvalue blocks, expands a value table in the Walsh-Hadamard basis and reports the share of variance at each order, and gives the single and pair terms of the expected cost when lines fail independently.

The pair coefficients are checked against brute force in `experiments/crew_assignment.py`. Quadratic programs for weighted completion time on parallel machines: Skutella, Convex quadratic and semidefinite programming relaxations in scheduling, J. ACM 48(2), 2001 (abstract read), https://page.math.tu-berlin.de/~combi/coga-old/publications/techreports/2000/Report-683-2000.html

Walsh-Hadamard expansion and weight by degree: O'Donnell, Analysis of Boolean Functions, chapter 1, https://arxiv.org/abs/2105.10386

Least squares: NumPy `lstsq`, https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html

## briskit/qaoa.py

Computes the output probabilities of QAOA exactly from a table of the cost of every plan: each layer multiplies the state by a phase set by the cost, then rotates every qubit about X. Trains the angles with BFGS from several starting points, gives the expected best cost among a number of shots, and builds the same circuit in Qiskit with one RZZ rotation per pair.

Algorithm: Farhi, Goldstone and Gutmann, A Quantum Approximate Optimization Algorithm, https://arxiv.org/abs/1411.4028

Optimiser: SciPy `minimize` with BFGS, https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html

Circuit: Qiskit `QuantumCircuit`, https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit

## briskit/baselines.py

Classical methods used for comparison: descent by single bit flips, the end point of steepest descent from every plan at once, Karmarkar-Karp differencing for splitting numbers into two equal sums, longest job to the less loaded crew, a thermal sampler that draws a plan with probability proportional to exp(-beta * cost), and greedy selection by marginal gain.

Differencing: Karmarkar and Karp, The Differencing Method of Set Partitioning, UC Berkeley report CSD-83-113, https://www2.eecs.berkeley.edu/Pubs/TechRpts/1983/6353.html

## briskit/hardware.py

Builds one cost layer with a rotation on every pair of qubits, counts the two-qubit gates and the two-qubit depth of a compiled circuit, and gives the fraction of shots with no two-qubit error as (1 - r) to the power of the gate count, with r = 0.00495, the two-qubit rate in IonQ's Forte 1 noise model.

Error rate: IonQ, Simulation with Noise Models, https://docs.ionq.com/guides/simulation-with-noise-models

Compiling: Qiskit transpiler, https://quantum.cloud.ibm.com/docs/api/qiskit/transpiler

## briskit/ion_model.py

Simulates ions driven through one shared vibrational mode. The Hamiltonian is the sum over ions of g_i(t) X_i (a exp(-i delta t) + a^dagger exp(i delta t)), stepped in time with the mode cut off at 30 levels. Returns the operator on the ions when the mode starts and ends in its ground state, the population left in the mode, and the target operator with pair phase (2 g_i g_j / delta) min(T_i, T_j).

Interaction: Sørensen and Mølmer, Quantum computation with ions in thermal motion, https://arxiv.org/abs/quant-ph/9810039

Many ions at once, and the mode returning to its start: Lotshaw et al., Modelling noise in global Mølmer-Sørensen interactions applied to quantum approximate optimization, section II A, https://arxiv.org/abs/2211.00133

Matrix exponential: SciPy `expm`, https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html

## experiments/crew_assignment.py

Checks the one crew order and the pair cost against brute force. On the IEEE 13 node feeder with two crews it compares the conversion algorithm and the best plan of the closed-form pair cost with the best of all 4,096 assignments, solves 12 instances to proven optimality, and runs QAOA on the closed-form pair cost.

Tan et al., sections 3, 5 and 6.1, https://arxiv.org/abs/1702.08382

## experiments/fitted_pair_cost.py

Fits a pair cost to the true cost on a tree from randomly chosen plans. Compares the best plan of the fit, and QAOA shots drawn from it, with random plans, bit-flip descent, the conversion algorithm and the plans used for the fit, on the feeder with 12 repairs and on random trees with 18.

NumPy `lstsq`, https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html

Farhi, Goldstone and Gutmann, https://arxiv.org/abs/1411.4028

## experiments/urgency_levels.py

Checks by enumeration that the pair matrix of crew assignment is a sum of one outer product per urgency level, that its rank is the number of levels, and that the cost is the matching sum of squares. Then compares, by state fidelity, a cost layer made of one diagonal operation per level with the layer made of one RZZ rotation per pair.

Number partitioning as one collective interaction of ions: Graß et al., Quantum annealing for the number partitioning problem using a tunable spin glass of ions, https://arxiv.org/abs/1507.07863

Qiskit `DiagonalGate`, https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.library.DiagonalGate

Qiskit `Statevector`, https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.Statevector

## experiments/independent_repairs.py

Repairs with no network between them, where the pair cost is exact. Compares QAOA at depths 1 to 3 with random plans given the same number of shots, and with the rule that takes repairs in order of weight per hour and gives each to the crew that is free first, at 12 and 18 repairs.

Farhi, Goldstone and Gutmann, https://arxiv.org/abs/1411.4028

Instance generator: Tan et al., section 6, https://arxiv.org/abs/1702.08382

## experiments/dispatch_rules.py

Customer-hours without power on 300 feeder instances under the FirstEnergy rule (largest weight first), the Edison Electric Institute rule (largest weight per hour first) and the conversion algorithm, each against the best two crew assignment.

Tan et al., sections 5.4 and 6.1, https://arxiv.org/abs/1702.08382

## experiments/protection_and_levels.py

Checks the closed form for the expected cost of leaving lines unprotected against the expectation over every set of failed lines. Then solves crew assignment one urgency level at a time, carrying the difference in hours between the crews to the next level, and compares with the optimum and the classical rule.

The closed form is the pair cost of `briskit/pair_cost.py` averaged over independent failures. Its check is the enumeration in this file.

## experiments/simulator_agreement.py

Compares the probabilities from the table-based simulator in `briskit/qaoa.py` with those of the Qiskit circuit, by total variation distance, at 10 qubits and depths 1 to 3.

Qiskit `Statevector`, https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.Statevector

## experiments/samplers_and_protection_rule.py

On the fitted pair cost of the feeder, compares QAOA shots with a thermal sampler at the same mean cost, the thermal sampler at its best temperature, and the exact list of lowest-cost plans. Then counts how often choosing lines to protect by their single-line terms differs from the optimum of the expected cost.

Farhi, Goldstone and Gutmann, https://arxiv.org/abs/1411.4028

## experiments/qaoa_limits.py

Four measurements on independent repairs: one urgency level against Karmarkar-Karp differencing; QAOA shots as starting points for descent; pair terms dropped from the cost layer under a noise model in which a shot is unaffected with probability (1 - r) to the power of the gate count and random otherwise; and the chance of the best plan against the number of repairs. The optimiser stops after 40 iterations per start.

Karmarkar and Karp, https://www2.eecs.berkeley.edu/Pubs/TechRpts/1983/6353.html

IonQ, Simulation with Noise Models, https://docs.ionq.com/guides/simulation-with-noise-models

## experiments/feeder_fitted_models.py

Restoring a fixed number of the 12 feeder lines: largest weights first, greedy selection, and fitted costs with and without pair terms, each against the optimum found by enumeration. Then the fitted pair cost of crew assignment is cut down to its largest eigenvalue blocks and the best plan of each cut is scored.

The decision of which damaged links to restore within a budget: Udekwe et al., Q-RESTORE, https://arxiv.org/abs/2501.11197

## experiments/budgeted_restoration.py

Restore 8 of 18 lines on random trees. A pair cost is fitted on 600 random sets of the right size and a penalty on the count is added. QAOA on that cost, with shots of the wrong count discarded, is compared with a thermal sampler at the same mean value, random sets, greedy selection, and exact search of the fitted cost.

The decision of which damaged links to restore within a budget: Udekwe et al., Q-RESTORE, https://arxiv.org/abs/2501.11197

Farhi, Goldstone and Gutmann, https://arxiv.org/abs/1411.4028

## experiments/interaction_spectrum.py

Expands the full value table of each decision on the feeder in the Walsh-Hadamard basis and reports the share of the non-constant variance at orders 1 to 4 and above.

O'Donnell, Analysis of Boolean Functions, chapter 1, https://arxiv.org/abs/2105.10386

## experiments/level_merging.py

Rounds each urgency ratio up to a geometric grid, which leaves fewer levels, and measures the rounded cost over the true cost for every plan and the loss of the plan the rounded problem selects.

The cost of `briskit/pair_cost.py` does not decrease when a ratio is raised and scales with the ratios, which gives the bound. Its check is the enumeration in this file.

## experiments/gate_counts.py

Compiles one cost layer with a term on every pair, once for a device where any two qubits interact with RZZ as a basis gate and once for a heavy-hex coupling map with CZ, and reports two-qubit gate count and depth.

Qiskit transpiler, https://quantum.cloud.ibm.com/docs/api/qiskit/transpiler

Qiskit `CouplingMap`, https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.transpiler.CouplingMap

IonQ native gates, https://docs.ionq.com/guides/getting-started-with-native-gates

## experiments/ion_collective_interaction.py

Three ions and one mode. Compares the simulated operator with the target pair phases for switch-off times that are whole and non-whole numbers of periods, with the phase of one laser flipped, and with the drive on the other side of the mode frequency.

Sørensen and Mølmer, https://arxiv.org/abs/quant-ph/9810039

Lotshaw et al., section II A, https://arxiv.org/abs/2211.00133

## run_experiments.py

Runs every script in `experiments/` in its own process with a fixed number of numerical threads and writes what each prints to `results/<name>.txt`. With `--check` it runs them again and reports whether each output equals the saved one. `--jobs` sets how many run at once, `--threads` the thread count, and names after the command select scripts.

    python run_experiments.py
    python run_experiments.py --check
    python run_experiments.py --check dispatch_rules level_merging

Every script seeds NumPy's generator with the constant `SEED` at the top of the file, so a rerun draws the same instances: https://numpy.org/doc/stable/reference/random/generator.html

## results/

One file per script in `experiments/`. The lines before the first blank line record how the output was made: the command, the commit of the code and whether the code had uncommitted changes, the Python and package versions, the platform, the thread count, the start time in UTC, the run time in seconds and the exit code. The rest is the output.

Package versions are pinned in `requirements.txt`. The saved results were made with one thread. The depth 3 figures of `qaoa_limits.py` depend on the thread count, because its optimiser stops after 40 iterations and the last bits of a sum depend on how it is split: with 14 repairs the chance of the best plan is 7.2 times random at one thread and 7.5 at two (growth per added repair x1.105 and x1.111), and the best of 100 shots at one urgency level is 0.0856 hours at one and two threads and 0.0854 at the machine's default.

## qollab/main.py

The file Qollab runs. Builds the QAOA circuit for one 12 repair instance with angles trained beforehand, compiles it for the `backend` Qollab supplies, sends one job, and scores the measured plans with the classical cost table.

Qollab, Python Qiskit projects, https://qollab.xyz/learn/docs/python-qiskit-projects

Qollab, Runtime environment, https://qollab.xyz/learn/docs/runtime-environment

## qollab/run_local.py

Runs `main.py` on a laptop by supplying Qiskit's `BasicSimulator`, or an IBM fake backend, as `backend`, which is what Qollab does for a built-in run.

Qollab, Python Qiskit projects, https://qollab.xyz/learn/docs/python-qiskit-projects

Qiskit `BasicSimulator`, https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.providers.basic_provider.BasicSimulator

## briskit/flow.py

Flow networks. Stores an instance as JSON (network with node positions and link capacities, the damaged links as `elements`, commodities with source, sink, demand, priority tier and weight, `budget` as the numbers of links to restore, damage disk). Generates three families on 16 nodes (random geometric, ring with chords, hub and spoke), damages the links whose midpoints lie nearest a random centre, and draws 6 to 8 commodities in three priority tiers. Scores a plan with a multi-commodity flow linear program: each link carries flow in both directions up to its capacity, or none if it is damaged and not restored, and the program maximises the priority-weighted demand served. The value table runs that program on every plan, in the plan order of `briskit/feeder.py`, in parallel; the environment variable `FLOW_PROCESSES` caps the number of processes. Every flow script regenerates the instances it needs and rewrites a stored file that differs, and reuses a stored value table only when it carries the fingerprint of the current instance, so `run_experiments.py` can run them in any order. Files are written to a temporary name and then swapped in whole, so a script running at the same time never reads half a file.

Hand-made instance: links 0-1 (capacity 4), 1-2 (3), 0-3 (2), 3-2 (2), 2-4 (5); links 1-2 and 3-2 damaged; commodity 0 to 2 with demand 4 and weight 2, commodity 0 to 4 with demand 3 and weight 1. Both commodities must enter node 2 through a damaged link. Nothing restored: 0. Only 1-2: 3 units into node 2, all to the heavier commodity, 3 x 2 = 6. Only 3-2: 2 units, 2 x 2 = 4. Both: 5 units, 4 to the first commodity and 1 to the second, 4 x 2 + 1 x 1 = 9.

SciPy `linprog` with HiGHS, https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html

Multi-commodity flow, https://en.wikipedia.org/wiki/Multi-commodity_flow_problem

NetworkX random geometric graph (definition only; NetworkX is not used), https://networkx.org/documentation/stable/reference/generated/networkx.generators.geometric.random_geometric_graph.html

NetworkX Newman-Watts-Strogatz graph (definition only), https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.newman_watts_strogatz_graph.html

Python `multiprocessing`, https://docs.python.org/3/library/multiprocessing.html

Python `os.replace` (atomic on POSIX), https://docs.python.org/3/library/os.html#os.replace

Every constant without a source is listed in `SOURCES.md`.

## experiments/flow_instances.py

Writes the hand-made instance and the 18 generated instances (three families, 12 and 16 damaged links, seeds 0 to 2) to `data/flow/`.

## experiments/flow_model_checks.py

Checks the flow program: the hand-made instance gives 0, 6, 4 and 9; restoring every damaged link gives the value of the undamaged network; for 200 random pairs of plans, adding links never lowers the value. Writes the time of one program to `results/flow/program_time.csv`, so the printed output is the same on every run.

## experiments/flow_value_tables.py

Exact value table of every flow instance (E1), stored in `results/flow/tables/<instance>.npz` with the instance fingerprint and the time it took, in the plan order of `briskit/feeder.py`; the times are copied to `results/flow/value_tables.csv`. Checks on each table that the last plan equals the undamaged value and that restoring any one more link never lowers the value, over every plan, and counts the idle links, damaged links whose restoration never changes the value.

## experiments/flow_fitted_pair_cost.py

On the flow tables, restore each number of damaged links in the instance's `budget`. Each instance and number restored has its own generator seeded from `SEED` and its name, so adding or removing an instance leaves the samples of the others unchanged. Fits a cost with single-link terms only (the control) and with single and pair terms, on all plans of the right size (E3) and on 10 to 320 random plans (E4); both fits use the same sample in each draw. Each fit gives two rows in `results/flow/fitted_pair_cost.csv`: budget 0, the fitting sample alone, and budget 1, the sample plus the plan the fit returns. `value` is the best plan scored in the run, `returned` the plan the fit returns, `evaluations` the distinct plans scored, `seconds` the time of the fit. The instance column carries the number restored, for example `ring_with_chords_12_0_restore4`, and `seed` is the draw. With all plans in the fit (E3) every plan is scored, so only `returned` is informative there. A number restored for which no plan serves any demand is skipped and named in the output, since regret divides by the optimum.

Udekwe et al., Q-RESTORE, https://arxiv.org/abs/2501.11197

Ridge fit: `briskit/pair_cost.py`.
