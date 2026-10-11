import argparse
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
TRACKED = ["briskit", "experiments", "run_experiments.py", "requirements.txt"]
THREAD_VARIABLES = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]


def experiment_names():
    return sorted(path.stem for path in (ROOT / "experiments").glob("*.py") if path.stem != "__init__")


def git(*arguments):
    done = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else None


def code_version():
    commit = git("rev-parse", "--short", "HEAD")
    if commit is None:
        return "unknown"
    changed = git("status", "--porcelain", "--", *TRACKED)
    return f"{commit} ({'modified' if changed else 'clean'})"


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def environment_lines(threads):
    packages = "  ".join(f"{name} {package_version(name)}" for name in ("numpy", "scipy", "qiskit"))
    return [f"code: {code_version()}",
            f"python: {platform.python_version()}",
            f"packages: {packages}",
            f"platform: {platform.platform()}, {platform.machine()}",
            f"threads: {threads}"]


def run(name, threads, processes):
    command = [sys.executable, "-m", f"experiments.{name}"]
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", FLOW_PROCESSES=str(processes),
                       **{key: str(threads) for key in THREAD_VARIABLES})
    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    done = subprocess.run(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8")
    header = [f"command: python -m experiments.{name}",
              *environment_lines(threads),
              f"started: {started.strftime('%Y-%m-%dT%H:%M:%SZ')}",
              f"seconds: {time.perf_counter() - clock:.0f}",
              f"exit: {done.returncode}"]
    return header, done.stdout.replace("\r\n", "\n")


def saved_output(name):
    path = RESULTS / f"{name}.txt"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").split("\n\n", 1)[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=experiment_names())
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--processes", type=int, default=None)
    options = parser.parse_args()
    processes = options.processes or max(1, (os.cpu_count() or 1) // options.jobs)

    RESULTS.mkdir(exist_ok=True)
    different = []
    with ThreadPoolExecutor(max_workers=options.jobs) as pool:
        runs = {name: pool.submit(run, name, options.threads, processes) for name in options.names}
        for name, pending in runs.items():
            header, output = pending.result()
            if options.check:
                same = output == saved_output(name)
                different += [] if same else [name]
                print(f"{name}: {'same as saved' if same else 'DIFFERENT from saved'}", flush=True)
            else:
                (RESULTS / f"{name}.txt").write_text("\n".join(header) + "\n\n" + output, encoding="utf-8", newline="\n")
                print(f"{name}: {header[-2]}, {header[-1]}", flush=True)
    sys.exit(1 if different else 0)


if __name__ == "__main__":
    main()
