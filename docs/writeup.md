# CrewCut-QAOA

Team Briskit. Qollab x IonQ hackathon, October 2026. This effort is supported by Qollab & IonQ.

## What it is

CrewCut-QAOA is a quantum circuit that assigns storm repairs to two repair crews. It uses one qubit for each repair, and the value of that qubit says which crew takes the repair.

We tested it on the IEEE 13 node test feeder with 12 damaged lines and two crews. With a hundred shots the circuit lands 0.32% from the best plan. A hundred random plans land 0.62% from it, and a published classical algorithm lands 1.47% from it. The best plan is the cheapest way to split the repairs between the two crews, and we find it by scoring every split.

The project page is a small game with eight repairs. You assign the repairs to the two crews and send the crews out. The page then scores all 256 possible plans and runs CrewCut and standard QAOA on the same eight repairs.

All circuits ran on simulators. We did not run anything on a quantum computer.

## Why it matters and who it is for

After a storm a utility has more broken lines than crews. The crew that each repair goes to decides how long customers stay without power. Weather outages cost the United States $18 to $33 billion a year, so one percent of that cost is $180 to $330 million. On our test feeder, two utility dispatch rules end 19.4% and 14.2% from the best plan.

Researchers can use our comparison with standard QAOA. Standard QAOA needs a penalty term. Its valid shots get 0.1% to 7.8% of the way from random plans to the best plan. The shots of CrewCut get 56% to 73% of the way.

Teachers and students can use the project because the problem needs no quantum background. It has two crews, repair hours and priorities, and it runs with 6 to 20 qubits on a laptop.

Engineers who build quantum compilers and machines can use it as a small test job, because the best answer is known for every instance.

Classical methods still do better than the circuit at this size, so we do not suggest using QAOA for real power restoration today.

## How the quantum part works

Each repair gets one qubit. A measured 0 gives the repair to crew A and a measured 1 gives it to crew B. Each crew does its repairs in a fixed order, with the highest priority per hour first, so the order needs no extra qubits.

When two repairs are on the same crew, one of them has to wait for the other. The cost of a plan is a fixed amount plus a waiting cost for every pair of repairs that share a crew. The cheapest plan puts the pairs with the largest waiting costs on different crews. This is the weighted MaxCut problem.

The circuit is QAOA. It starts with a Hadamard gate on every qubit. Each layer then applies one RZZ gate to every pair of qubits, with an angle proportional to the waiting cost of that pair, followed by one RX gate on every qubit. Each layer has two angles, and we tune them in simulation before the run. Measuring the circuit gives one plan. We measure many times, score each plan with the full cost model and keep the best one.

On a feeder, a repaired line carries power only after the lines between it and the substation are repaired, so the cost is no longer an exact sum over pairs. For that case we fit the pair costs to 150 scored plans.

Standard QAOA uses two qubits for each repair, one for each crew, and adds a penalty so that every repair ends up on exactly one crew. For 12 repairs it needs 24 qubits and 144 two qubit gates in each layer. CrewCut needs 12 qubits and 66 gates, and every measurement is a valid plan.

The one qubit form comes from scheduling theory (Skutella, 2000). We added its use for power restoration on a feeder, the fitted pair costs, a cost layer with the weakest half of the gates removed, and the tests against standard QAOA and classical search.

| Result | Setup |
|---|---|
| 0.32% from the best plan with a hundred shots | one circuit layer, 20 feeder instances, mean |
| 0.09% from the best plan with 200 scored plans. Classical best first search reaches 0.00% with the same 200. | two layers, 30 feeder instances, median |
| 3 to 5 times fewer shots than standard QAOA to return a best plan | 6 to 10 independent repairs, two layers, median, ahead on every instance |
| 52% of the gain kept with half of the two qubit gates removed, and 45% with all of them | noise model with 1.33% error per two qubit gate, 12 independent repairs, two layers, 30 instances, median |

## How to run it

On Qollab, open the Code tab and click Run Project. Choose the built-in simulator in the Select QPU window and press Run. Every simulator in that window is free. The entries marked "Credits apply" are real quantum computers.

When the page has loaded, click each site to give it to crew A or crew B, then press Send the crews. Press Search every plan to score all 256 plans. Press Run the circuits to sample plans with both circuits. The last section of the page runs a 12 repair circuit and gives you the record of the run as a file.

On a laptop you need Python 3.13. Clone the repository at github.com/NikoKleion/CrewCut-QAOA-Team-Briskit- and run these commands from its root folder.

    pip install -r requirements.txt
    python qollab/run_local.py --no-figures
    python run_experiments.py dispatch_rules --check
    python -m checks.dispatch_game_checks

The second command runs the 12 repair circuit on the Qiskit simulator and prints its report. The third reruns one experiment and reports whether the output is the same as the saved result. The fourth checks the Python code of the page against the library, once on the real code and once on a broken copy that has to fail.

The file syllabus.md lists every experiment with the source behind it. Any of them can be rerun with `python run_experiments.py` and the name of the experiment. The largest one takes three hours.

## Limits

The method handles two crews. It leaves out travel time between sites. All results come from simulators and noise models. At 12 repairs a classical search of every plan takes about a millisecond, and classical best first search is 0.09 points ahead of CrewCut when both score 200 plans. We do not claim a quantum advantage.

## Links

The repository is at github.com/NikoKleion/CrewCut-QAOA-Team-Briskit- under the MIT licence. The file docs/crewcut_vs_standard_qaoa.md is the full writeup and maps each claim to its result file. The file docs/CrewCut-QAOA-pamphlet.pdf is the pamphlet. The folder docs/money holds the dollar figures and their sources.
