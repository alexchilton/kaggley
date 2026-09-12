"""Report every graded question that is still unanswered.

Three notebooks shipped with an unanswered question because the stub detector
was built from one example. This course family marks a blank at least four
ways -- `____`, an empty `""" """` query, a complete query under "# Amend the
query below", and a bare assignment like `trend = food_sales` -- and the last
two run without error, so nothing fails until a human reads the grader.

It reports two things. First, any cell anywhere that still contains `____` --
in several courses the answer cell is separate from the cell that calls
`check()`, so looking only at check cells misses them entirely. Second, and not
relying on any pattern, it compares each built notebook against the untouched
copy in `sources/` and lists every `q_N.check()` cell that is byte-identical,
i.e. every question left exactly as the exercise shipped it.
That is only correct when the question needs no answer, which is what the
`--thought-experiments` argument records: those question numbers are
`ThoughtExperiment` in the learntools grader and are credited by running the
cell.

    python kaggle/_tools/audit.py <course> [--thought-experiments ex2:1,3 ex4:2]

Exits non-zero if anything is unanswered, so it can gate a push.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Courses do not agree on what the checker object is called: most use q_1, the
# Intermediate ML course uses step_1, and some have sub-parts like step_1.a.
# Matching only `q_` made this tool report a whole course clean that it could
# not actually see -- the exact false green light it exists to prevent.
CHECK_CALL = re.compile(r"\b(q_|step_)(\d+)(?:\.[a-c])?\.check\(")


def is_bare_check(text):
    """True if the cell holds nothing but check/hint/solution calls and comments.

    Several courses put the answer in one cell and the `check()` in the next.
    That trailing cell is meant to stay exactly as shipped, so an unchanged
    one is correct rather than suspicious -- but only when it contains no code
    of its own to fill in.
    """
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if re.fullmatch(r"(?:q_|step_)\d+(\.[a-c])?\.(check|hint|solution)\([^)]*\)", line):
            continue
        return False
    return True


def parse_allowed(args):
    allowed = {}
    for item in args:
        ex, _, nums = item.partition(":")
        allowed[ex] = {int(n) for n in nums.split(",") if n}
    return allowed


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    course = sys.argv[1]
    allowed = parse_allowed(sys.argv[3:]) if "--thought-experiments" in sys.argv else {}

    problems = 0

    # An empty ex/ directory is not a pass. Reporting "clean" for a course
    # whose notebooks have not been built yet is exactly the kind of green
    # light this tool exists to prevent.
    for exdir in sorted((ROOT / course).glob("ex*/")):
        if not list(exdir.glob("*.ipynb")):
            print(f"  NOT BUILT    {course}/{exdir.name}  has metadata but no notebook")
            problems += 1

    for built in sorted((ROOT / course).glob("ex*/*.ipynb")):
        ex = built.parent.name
        src = ROOT / course / "sources" / built.name
        if not src.exists():
            print(f"  ?? {built}: no source to compare against")
            problems += 1
            continue
        b = json.loads(built.read_text())["cells"]
        s = json.loads(src.read_text())["cells"]

        # A literal ____ left anywhere is unanswered, wherever check() lives.
        for i, cb in enumerate(b):
            if cb["cell_type"] == "code" and "____" in "".join(cb["source"]):
                print(f"  STUB LEFT    {course}/{ex}  cell {i:>3}  contains ____")
                problems += 1

        for i, (cb, cs) in enumerate(zip(b, s)):
            if cb["cell_type"] != "code":
                continue
            tb = "".join(cb["source"])
            m = CHECK_CALL.search(tb)
            if not m or tb != "".join(cs["source"]):
                continue
            q = int(m.group(2))
            if q in allowed.get(ex, set()):
                continue
            if is_bare_check(tb):
                continue
            print(f"  UNANSWERED  {course}/{ex}  cell {i:>3}  q_{q}")
            problems += 1

    print("audit clean" if problems == 0 else f"{problems} unanswered")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
