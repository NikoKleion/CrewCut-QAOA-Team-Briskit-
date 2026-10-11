import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROCESSES = min(5, os.cpu_count() or 1)
OVERNIGHT = "overnight" in sys.argv[1:]
NAME = "standard_qaoa_10" if OVERNIGHT else "standard_qaoa"

print("Compares standard QAOA (one qubit per repair and crew, with a penalty) with one qubit per repair.")
print("python RUNTHIS.py            6 and 8 repairs, 30 instances each; took 2021 s with 5 processes when it was run")
print("python RUNTHIS.py overnight  10 repairs, 10 instances; took 10966 s and 11361 s on two machines")
print("Needs Python 3.13 and: pip install -r requirements.txt")
print(f"This run: {NAME}, {PROCESSES} worker processes at one thread each. Plug the machine in first.")
print(f"It replaces results/{NAME}.txt; the lines below its header must come out as they were.")
print(f"To compare without replacing the file: python run_experiments.py {NAME} --check")
print("Nothing is printed until the run ends.")
print("", flush=True)

done = subprocess.run([sys.executable, "run_experiments.py", NAME, "--processes", str(PROCESSES)], cwd=ROOT)

print("")
if done.returncode == 0:
    print("Done. Send the result back with:")
    print(f"    git add results/{NAME}.txt")
    print('    git commit -m "Add standard QAOA result"')
    print("    git push")
else:
    print(f"{NAME} did not finish. Send what is printed above to Niko.")
sys.exit(done.returncode)
