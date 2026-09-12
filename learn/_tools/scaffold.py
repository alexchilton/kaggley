"""Lay out a pulled Kaggle Learn course as a pushable directory tree.

Takes the notebooks pulled into a scratch directory and writes
`kaggle/<course>/sources/<notebook>` plus `kaggle/<course>/ex<N>/` holding a
`kernel-metadata.json` retargeted at this account.

    python kaggle/_tools/scaffold.py <scratch-dir> <course-name> <slug> [<slug> ...]
"""

import json
import shutil
import sys
from pathlib import Path

ACCOUNT = "alexchilton"
ROOT = Path(__file__).resolve().parents[1]


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    scratch, course, slugs = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]

    (ROOT / course / "sources").mkdir(parents=True, exist_ok=True)
    for n, slug in enumerate(slugs, start=1):
        books = list((scratch / slug).glob("*.ipynb"))
        if not books:
            raise SystemExit(f"{slug}: no notebook in {scratch / slug}")
        src = books[0]
        shutil.copy(src, ROOT / course / "sources" / src.name)

        out = ROOT / course / f"ex{n}"
        out.mkdir(parents=True, exist_ok=True)
        meta = json.loads((scratch / slug / "kernel-metadata.json").read_text())
        meta["id"] = f"{ACCOUNT}/{slug}"
        meta.pop("id_no", None)
        meta["is_private"] = True
        (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"  ex{n:<2} {slug:<50} {src.name}")


if __name__ == "__main__":
    main()
