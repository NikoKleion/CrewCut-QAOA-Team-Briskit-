import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROCESSES = min(5, os.cpu_count() or 1)
OVERNIGHT = "overnight" in sys.argv[1:]
NAMES = ["standard_qaoa"] + (["standard_qaoa_10"] if OVERNIGHT else [])

print("Compares standard QAOA (one qubit per repair and crew, with a penalty) with one qubit per repair,")
print("at 6 and 8 repairs, 30 instances each. Nothing else needs to be run.")
print("Needs Python 3.13 and: pip install -r requirements.txt")
print(f"Uses {PROCESSES} worker processes at one thread each. Plug the machine in first.")
print("Measured in one process on a laptop on battery: 6 minutes for 6 repairs, 3 minutes per instance at 8 repairs.")
print("With 5 processes that is about 25 minutes in all. Nothing is printed until a run ends.")
if OVERNIGHT:
    print("Then the same at 10 repairs, 10 instances. Not timed as a whole: one circuit evaluation takes 48 times")
    print("as long as at 8 repairs, which puts it at several hours with 5 processes.")
else:
    print("For the run at 10 repairs as well, which takes several hours: python RUNTHIS.py overnight")
print("", flush=True)

code = 0
for name in NAMES:
    done = subprocess.run([sys.executable, "run_experiments.py", name, "--processes", str(PROCESSES)], cwd=ROOT)
    code = code or done.returncode

results = [f"results/{name}.txt" for name in NAMES if (ROOT / "results" / f"{name}.txt").exists()]
print("")
if code == 0 and len(results) == len(NAMES):
    print("Done. Send the result back with:")
    print(f"    git add {' '.join(results)}")
    print('    git commit -m "Add standard QAOA result"')
    print("    git push")
else:
    print("It did not finish. Send what is printed above to Niko.")
sys.exit(code)
