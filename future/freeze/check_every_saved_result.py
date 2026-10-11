# Reruns every experiment and compares what it prints with the file saved in results/ (ready to run, not yet run; the saved results record 4.9 hours of run time in all, 3 of them the comparison at 10 repairs)
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

sys.exit(subprocess.run([sys.executable, "run_experiments.py", "--check", "--jobs", "1", "--processes", "5"],
                        cwd=ROOT).returncode)
