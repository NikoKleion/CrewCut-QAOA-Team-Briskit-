# Checks of saved_run.py, each run on the real code and on a broken version
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import saved_run as module

from checks.record import run_checks

PROBE = module.ROOT / "future" / "probe_of_saved_run.py"


def holding_everything_until_the_end(script, main):
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        main()
    Path(script).with_suffix(".txt").write_text("command: probe\ncode: probe\nexit: 0\n\n" + printed.getvalue(), encoding="utf-8")


def without_a_header(script, main):
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        main()
    Path(script).with_suffix(".txt").write_text(printed.getvalue(), encoding="utf-8")


def saved_text(run, main):
    saved = PROBE.with_suffix(".txt")
    saved.unlink(missing_ok=True)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            run(PROBE, main)
    except ValueError:
        pass
    text = saved.read_text(encoding="utf-8") if saved.exists() else ""
    saved.unlink(missing_ok=True)
    return text


def a_failure_keeps_what_was_printed(run):
    def main():
        print("first line")
        print("second line")
        raise ValueError("stopped on purpose")

    text = saved_text(run, main)
    found = [part for part in ("first line", "second line", "exit: failed: ValueError: stopped on purpose") if part in text]
    return "both lines and the failure in the saved file", f"{len(found)} of 3 found", len(found) == 3


def a_clean_run_saves_header_and_prints(run):
    def main():
        print("first line")
        print("second line")

    text = saved_text(run, main)
    head, _, body = text.partition("\n\n")
    found = [part for part in ("command: python future/probe_of_saved_run.py", "code: ", "threads: ", "exit: 0") if part in head]
    return "4 header lines and the 2 printed lines", f"{len(found)} header lines, body {body.splitlines()}", len(found) == 4 and body.splitlines() == ["first line", "second line"]


CASES = [
    (a_failure_keeps_what_was_printed, module.saved_run, holding_everything_until_the_end,
     "prints held in memory until the script has finished"),
    (a_clean_run_saves_header_and_prints, module.saved_run, without_a_header, "no header written"),
]


if __name__ == "__main__":
    sys.exit(0 if run_checks(CASES, Path(__file__).with_suffix(".csv")) else 1)
