# Runs a script of this folder and saves what it prints beside it, with the code version and the environment
import contextlib
import io
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run_experiments import environment_lines


def folder_state():
    done = subprocess.run(["git", "status", "--porcelain", "--", "future"], cwd=ROOT, capture_output=True, text=True)
    return "unknown" if done.returncode else ("modified" if done.stdout.strip() else "clean")


def saved_run(script, main):
    script = Path(script).resolve()
    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        main()
    header = [f"command: python {script.relative_to(ROOT).as_posix()}",
              *environment_lines(os.environ["OMP_NUM_THREADS"]),
              f"future folder: {folder_state()}",
              f"started: {started.strftime('%Y-%m-%dT%H:%M:%SZ')}",
              f"seconds: {time.perf_counter() - clock:.0f}"]
    saved = script.with_suffix(".txt")
    saved.write_text("\n".join(header) + "\n\n" + printed.getvalue(), encoding="utf-8", newline="\n")
    print(printed.getvalue(), end="")
    print(f"saved: {saved.relative_to(ROOT).as_posix()}")
