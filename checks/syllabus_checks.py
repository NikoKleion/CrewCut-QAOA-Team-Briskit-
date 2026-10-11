import re
import sys
from pathlib import Path

from checks.record import run_checks

ROOT = Path(__file__).resolve().parent.parent


def code_files():
    files = [f"{folder}/{path.name}" for folder in ("briskit", "experiments", "qollab")
             for path in sorted((ROOT / folder).glob("*.py")) if path.name != "__init__.py"]
    return files + ["run_experiments.py"]


def check_files():
    return [path.name for path in sorted((ROOT / "checks").glob("*.py")) if path.name != "__init__.py"]


def every_code_file_has_an_entry(text):
    titles = set(re.findall(r"(?m)^## (\S+)\s*$", text))
    missing = [name for name in code_files() if name not in titles]
    return "no file without an entry", missing or "no file without an entry", not missing


def every_check_file_is_named(text):
    entry = text.split("## checks/", 1)[1].split("\n## ", 1)[0] if "## checks/" in text else ""
    missing = [name for name in check_files() if f"`{name}`" not in entry]
    return "no check file left out", missing or "no check file left out", not missing


def every_entry_names_a_file_that_exists(text):
    titles = re.findall(r"(?m)^## (\S+)\s*$", text)
    missing = [title for title in titles if not (ROOT / title).exists()]
    return "no entry without a file", missing or "no entry without a file", not missing


def nothing_is_there_twice(text):
    starts = [block.strip()[:80] for block in text.split("\n\n") if len(block.strip()) > 200]
    commands = [line.strip() for line in re.findall(r"(?m)^    python \S.*$", text)]
    twice = sorted({start for start in starts if starts.count(start) > 1} | {line for line in commands if commands.count(line) > 1})
    return "nothing twice", twice or "nothing twice", not twice


def syllabus():
    return (ROOT / "syllabus.md").read_text(encoding="utf-8")


def without_one_entry():
    return re.sub(r"(?ms)^## briskit/feeder\.py\n.*?(?=^## )", "", syllabus(), count=1)


def without_one_check_name():
    return syllabus().replace("`record.py`", "the recorder")


def with_a_paragraph_twice():
    text = syllabus()
    longest = max(text.split("\n\n"), key=len)
    return text.replace(longest, longest + "\n\n" + longest[:300] + " and more.", 1)


def with_an_entry_for_no_file():
    return syllabus() + "\n## briskit/not_there.py\n\nNothing.\n"


CASES = [
    (every_code_file_has_an_entry, syllabus(), without_one_entry(), "the entry of one library file removed"),
    (every_check_file_is_named, syllabus(), without_one_check_name(), "one check file not named"),
    (every_entry_names_a_file_that_exists, syllabus(), with_an_entry_for_no_file(), "an entry for a file that is not there"),
    (nothing_is_there_twice, syllabus(), with_a_paragraph_twice(), "one paragraph there twice with a changed end"),
]


if __name__ == "__main__":
    sys.exit(0 if run_checks(CASES, Path(__file__).with_suffix(".csv")) else 1)
