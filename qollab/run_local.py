import argparse
import os
import runpy
import sys
import time
from pathlib import Path

project = Path(__file__).resolve().parent
records = project.parent / "results" / "qollab" / "local"
figure_folder = project.parent / "figures" / "quick_path"


def make_device(device_name):
    if device_name == "basic":
        from qiskit.providers.basic_provider import BasicSimulator
        return BasicSimulator()
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, "Fake" + device_name.capitalize())()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--keep", type=int, nargs="+", default=[100])
    parser.add_argument("--devices", nargs="+", default=["basic"])
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--save", action="store_true")
    arguments = parser.parse_args()

    # https://matplotlib.org/stable/users/explain/figure/backends.html: MPLBACKEND picks the backend, Agg writes files only
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(project)
    sys.path.insert(0, str(project))
    figure_folder.mkdir(parents=True, exist_ok=True)
    if arguments.save:
        records.mkdir(parents=True, exist_ok=True)

    figures = not arguments.no_figures
    for device_name in arguments.devices:
        device = make_device(device_name.lower())
        for keep in arguments.keep:
            started = time.perf_counter()
            supplied = {"backend": device, "figure_folder": str(figure_folder) + os.sep}
            if arguments.save:
                supplied["record_folder"] = str(records) + os.sep
            # https://docs.python.org/3/library/runpy.html: init_globals pre-populates the globals, as Qollab supplies backend
            namespace = runpy.run_path(str(project / "main.py"), init_globals=supplied)
            namespace["main"](shots=arguments.shots, depth=arguments.depth, keep=keep, figures=figures)
            figures = False
            print(f"wall time including backend setup: {time.perf_counter() - started:.1f} s")
            print()


if __name__ == "__main__":
    main()
