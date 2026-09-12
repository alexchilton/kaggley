"""Read each kernel's log and report the grade of every question.

Pairs with add_outcome_report.py. A run reporting COMPLETE only means the
notebook executed; this reads the OUTCOME lines that report prints, so a wrong
answer is visible instead of silently passing as a green run.

    python kaggle/_tools/collect_outcomes.py <slug> [<slug> ...]

Prints one line per question and a summary. Exits non-zero if anything is not
PASS.
"""

import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

# The end marker is literally "OUTCOME REPORT DONE", which the naive pattern
# happily read as a question called REPORT with outcome DONE -- so every
# notebook came back flagged. Require a q_/step_ name.
LINE = re.compile(r"OUTCOME ((?:q|step)_\d+(?:\.[a-d])?) (?:OutcomeType\.)?(\w+)")


def outcomes_for(slug):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["kaggle", "kernels", "output", f"alexchilton/{slug}", "-p", tmp],
            capture_output=True, text=True,
        )
        logs = list(Path(tmp).glob("*.log"))
        if not logs:
            return None
        try:
            rows = json.loads(logs[0].read_text())
        except Exception:
            return None
        text = "\n".join(r.get("data", "") for r in rows if isinstance(r, dict))
        return LINE.findall(text), ("OUTCOME REPORT DONE" in text)


def main():
    slugs = sys.argv[1:]
    if not slugs:
        raise SystemExit(__doc__)

    tally = Counter()
    bad = []
    for slug in slugs:
        got = outcomes_for(slug)
        if got is None:
            print(f"{slug:<52} NO LOG")
            bad.append((slug, "no log"))
            continue
        pairs, finished = got
        if not pairs:
            print(f"{slug:<52} NO OUTCOME LINES{'' if finished else ' (report did not run)'}")
            bad.append((slug, "no outcome lines"))
            continue
        # `None` means check() never ran for that question. That is expected
        # for the ThoughtExperiment half of a two-part question, which is
        # credited by calling .solution(), so report it separately from a FAIL.
        fails = [f"{q}={o}" for q, o in pairs if o == "FAIL"]
        unchecked = [q for q, o in pairs if o == "None"]
        tally.update(o for _, o in pairs)
        note = ""
        if unchecked:
            note = f"  (not checked: {' '.join(unchecked)})"
        status = "all pass" if not fails else "FAIL " + " ".join(fails)
        print(f"{slug:<52} {len(pairs):>2} questions  {status}{note}")
        if fails:
            bad.append((slug, status))

    print("\ntotals:", dict(tally))
    if bad:
        print(f"\n{len(bad)} notebook(s) need attention:")
        for slug, why in bad:
            print(f"  {slug}: {why}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
