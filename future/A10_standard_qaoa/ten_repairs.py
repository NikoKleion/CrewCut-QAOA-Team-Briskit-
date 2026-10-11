# Standard QAOA against one qubit per repair at 10 repairs, 10 instances: runs experiments/standard_qaoa_10.py and saves results/standard_qaoa_10.txt (ready to run, several hours, not yet run)
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSES = min(5, os.cpu_count() or 1)

sys.exit(subprocess.run([sys.executable, "run_experiments.py", "standard_qaoa_10", "--processes", str(PROCESSES)],
                        cwd=ROOT).returncode)
