Sources for the flow networks (area B). Each source was opened and the part used was read on 2026-10-10 unless stated otherwise. Constants with no source are listed at the end.

SciPy linprog, https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html. The form minimise c @ x subject to A_ub x <= b_ub, A_eq x = b_eq and bounds; method "highs"; status 0 means optimal. Used in briskit/flow.py.

Multi-commodity flow problem, Wikipedia, https://en.wikipedia.org/wiki/Multi-commodity_flow_problem. Commodities given by source, sink and demand; the capacity constraint summed over commodities; flow conservation. The page has no weighted variant, so the weighted objective with served demand capped at the demand is this project's own. Used in briskit/flow.py.

NetworkX random_geometric_graph, https://networkx.org/documentation/stable/reference/generated/networkx.generators.geometric.random_geometric_graph.html. Nodes uniform in the unit cube, linked when their distance is at most the radius. Definition only; NetworkX is not used. Used in briskit/flow.py.

NetworkX newman_watts_strogatz_graph, https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.newman_watts_strogatz_graph.html. A ring with shortcuts added and none removed; the page cites Newman and Watts, Physics Letters A 263, 341 (1999). Definition only. Used in briskit/flow.py.

Newman and Watts, Scaling and percolation in the small-world network model, https://arxiv.org/abs/cond-mat/9904419. Abstract only; it does not itself describe the construction. Not cited in the code.

Udekwe et al., Q-RESTORE, https://arxiv.org/abs/2501.11197. Abstract only. Choosing which damaged links to restore within a budget, on road networks. Used in experiments/flow_fitted_pair_cost.py.

Python multiprocessing, https://docs.python.org/3/library/multiprocessing.html. Pool.map returns results in input order; the spawn start method on Windows needs the __main__ guard. Used in briskit/flow.py.

Python os.replace, https://docs.python.org/3/library/os.html#os.replace. Replaces the destination file silently; the rename is atomic on POSIX. The page does not promise atomicity on Windows. Used in briskit/flow.py to swap in finished files.

Ridge fit, from briskit/pair_cost.py; its sources are in syllabus.md. Used in experiments/flow_fitted_pair_cost.py.

RestoreBench-Q Project Plan, sections 2.4 and 9.2 and experiments E1 to E4. Team document. Gives 12 and 16 damaged links, three seeds, the three graph families, a regional damage disk, 6 to 8 commodities in three priority tiers, and restoring 4 and 6 of 12. Used in all area B files.

Project choices with no source. 16 nodes per network. Tier weights 4, 2 and 1. Random geometric radius 0.35. Link capacities whole numbers from 5 to 15, and 20 between hubs. Three hubs placed in [0.35, 0.65] squared, with 75% of spokes also linked to the second-nearest hub. Demands whole numbers from 2 to 8. At least 1.5 times as many links as damaged links. The damage disk is the smallest disk around a uniform random centre that holds the nearest link midpoints, redrawn with the demands until the damage lowers the served value. Restoring 5 and 8 of 16, stored as the instance budget. Value tables with 64 plans or fewer are computed without a process pool. Fit sizes 10 to 320, doubling, with 10 draws each. The hand-made instance of 5 nodes with values 0, 6, 4 and 9 worked on paper. All in briskit/flow.py and experiments/flow_fitted_pair_cost.py.
