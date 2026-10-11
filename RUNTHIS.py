import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROCESSES = min(5, os.cpu_count() or 1)

print("Compares standard QAOA (one qubit per repair and crew, with a penalty) with one qubit per repair,")
print("at 6 and 8 repairs, 30 instances each. Nothing else needs to be run.")
print("Needs Python 3.13 and: pip install -r requirements.txt")
print(f"Uses {PROCESSES} worker processes at one thread each. Plug the machine in first.")
print("Measured in one process on a laptop on battery: 6 minutes for 6 repairs, 3 minutes per instance at 8 repairs.")
print("With 5 processes that is about 25 minutes in all. Nothing is printed until it ends.")
print("", flush=True)

done = subprocess.run([sys.executable, "run_experiments.py", "standard_qaoa", "--processes", str(PROCESSES)], cwd=ROOT)

print("")
if done.returncode == 0 and (ROOT / "results" / "standard_qaoa.txt").exists():
    print("Done. The result is results/standard_qaoa.txt. Send it back with:")
    print("    git add results/standard_qaoa.txt")
    print('    git commit -m "Add standard QAOA result"')
    print("    git push")
else:
    print("It did not finish. Send what is printed above to Niko.")
sys.exit(done.returncode)
