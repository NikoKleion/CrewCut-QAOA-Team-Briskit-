# Runs a script of this folder and saves what it prints beside it, with the code version and the environment; a run that fails keeps what it printed
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


class Both(io.TextIOBase):
    def __init__(self, stream):
        self.stream = stream
        self.kept = io.StringIO()

    def write(self, text):
        self.stream.write(text)
        self.kept.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()


def scripts_state():
    done = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", "future"], cwd=ROOT,
                          capture_output=True, text=True)
    changed = [line for line in done.stdout.splitlines() if line.strip().endswith(".py")]
    return "unknown" if done.returncode else ("modified" if changed else "clean")


def saved_run(script, main):
    script = Path(script).resolve()
    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    printed = Both(sys.stdout)
    outcome, failure = "0", None
    try:
        with contextlib.redirect_stdout(printed):
            main()
    except BaseException as error:
        outcome, failure = f"failed: {type(error).__name__}: {error}", error
    header = [f"command: python {script.relative_to(ROOT).as_posix()}",
              *environment_lines(os.environ["OMP_NUM_THREADS"]),
              f"future scripts: {scripts_state()}",
              f"started: {started.strftime('%Y-%m-%dT%H:%M:%SZ')}",
              f"seconds: {time.perf_counter() - clock:.0f}",
              f"exit: {outcome}"]
    saved = script.with_suffix(".txt")
    saved.write_text("\n".join(header) + "\n\n" + printed.kept.getvalue(), encoding="utf-8", newline="\n")
    print(f"saved: {saved.relative_to(ROOT).as_posix()}")
    if failure is not None:
        raise failure
