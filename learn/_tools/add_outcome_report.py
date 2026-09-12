"""Append a cell that prints every question's grading outcome.

A COMPLETE run proves the notebook executed, not that the answers were right:
learntools prints "Incorrect" rather than raising, and the rendered Correct /
Incorrect output is display data that the kernel log does not carry.

But each question object keeps `_last_outcome` after `check()` runs, and that
is plain Python. Printing it to stdout puts the grade in the log, where the
API can read it. Measured on a probe: a deliberately wrong answer reported
OutcomeType.FAIL while the kernel still reported COMPLETE.

    python kaggle/_tools/add_outcome_report.py <notebook> [<notebook> ...]
    python kaggle/_tools/add_outcome_report.py --strip <notebook> [...]
"""

import json
import sys
from pathlib import Path

MARKER = "OUTCOME REPORT"

CELL = '''# --- OUTCOME REPORT (added automatically; prints each question's grade) ------
import re as _re

_qnames = sorted(n for n in list(globals()) if _re.fullmatch(r"(q|step)_\\d+", n))
for _n in _qnames:
    _obj = globals()[_n]
    _targets = [(_n, _obj)]
    for _sub in ("a", "b", "c", "d"):
        _child = getattr(_obj, _sub, None)
        if _child is not None and hasattr(_child, "_last_outcome"):
            _targets.append((f"{_n}.{_sub}", _child))
    for _label, _p in _targets:
        if _targets and len(_targets) > 1 and _label == _n and not hasattr(_p, "_last_outcome"):
            continue
        print("OUTCOME", _label, getattr(_p, "_last_outcome", "NOT_CHECKED"))
print("OUTCOME REPORT DONE")
'''


def strip(path):
    nb = json.loads(Path(path).read_text())
    before = len(nb["cells"])
    nb["cells"] = [c for c in nb["cells"] if MARKER not in "".join(c.get("source", []))]
    if len(nb["cells"]) != before:
        Path(path).write_text(json.dumps(nb, indent=1) + "\n")
        print(f"  stripped  {path}")


def add(path):
    nb = json.loads(Path(path).read_text())
    if any(MARKER in "".join(c.get("source", [])) for c in nb["cells"]):
        print(f"  already has report  {path}")
        return
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": CELL.splitlines(keepends=True),
    })
    Path(path).write_text(json.dumps(nb, indent=1) + "\n")
    print(f"  added  {path}")


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    if args[0] == "--strip":
        for p in args[1:]:
            strip(p)
    else:
        for p in args:
            add(p)


if __name__ == "__main__":
    main()
