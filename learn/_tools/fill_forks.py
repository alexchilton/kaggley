"""Copy built notebooks into the hash-suffixed forks Kaggle creates.

Pushing a notebook to `alexchilton/<slug>` claims that slug. When the user then
opens the same exercise from the Kaggle Learn course page, Kaggle cannot reuse
the slug, so it forks a fresh EMPTY copy at `<slug>-<6 hex>` -- and that is the
one the course page links to. The answers sit in the clean slug and the user
opens a blank notebook.

This finds those forks and stages the built notebook for each, so both copies
carry the answers.

Matching strips the trailing hash and then looks for a built notebook whose
slug starts with what remains, because Kaggle also truncates long names:
`exercise-linear-regression-with-time-series` forks as
`exercise-linear-regression-with-time-serie-547c43`.

    python kaggle/_tools/fill_forks.py <staging-dir> <fork-slug> [<fork-slug> ...]
"""

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HASH = re.compile(r"-[0-9a-f]{6}$")


def built_notebooks():
    """slug -> (notebook path, metadata path) for everything already built."""
    out = {}
    for meta in ROOT.glob("*/ex*/kernel-metadata.json"):
        m = json.loads(meta.read_text())
        slug = m["id"].split("/", 1)[1]
        books = [p for p in meta.parent.glob("*.ipynb")]
        if books:
            out[slug] = (books[0], meta)
    return out


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    staging, forks = Path(sys.argv[1]), sys.argv[2:]
    staging.mkdir(parents=True, exist_ok=True)
    built = built_notebooks()

    for fork in forks:
        base = HASH.sub("", fork)
        matches = [s for s in built if s.startswith(base)]
        if len(matches) != 1:
            print(f"  SKIP  {fork}: {len(matches)} matches for {base!r}")
            continue
        notebook, meta_path = built[matches[0]]

        out = staging / fork
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(notebook, out / notebook.name)
        meta = json.loads(meta_path.read_text())
        meta["id"] = f"alexchilton/{fork}"
        meta.pop("id_no", None)
        (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"  staged  {fork:<52} <- {matches[0]}")


if __name__ == "__main__":
    main()
