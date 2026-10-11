# Reruns every experiment and compares what it prints with the file saved in results/ (ready to run, about an hour, to be done before results are frozen)
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

sys.exit(subprocess.run([sys.executable, "run_experiments.py", "--check", "--jobs", "1", "--processes", "5"],
                        cwd=ROOT).returncode)
