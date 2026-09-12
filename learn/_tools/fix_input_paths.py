"""Prepend a cell that makes `../input/<slug>/...` resolve on any Kaggle layout.

Kaggle mounts attached data in two different shapes, and nothing I could
isolate in kernel-metadata.json controls which you get -- same pinned image,
same source lists, different result:

    ../input/<slug>/...                       what every course notebook assumes
    /kaggle/input/datasets/<owner>/<slug>/    seen on API-pushed kernels
    /kaggle/input/competitions/<slug>/        likewise, for competition data

Rewriting the paths inside the notebook is not enough, and finding that out
cost a wasted push: the exercises fail at `from learntools.<course>.exN import
*`, because learntools loads the data itself, at import time, through its own
hardcoded `../input/...`. Nothing in the notebook can reach those strings.

So the fix is to make the path real rather than to avoid it. `/kaggle/input` is
read-only, but the working directory is not, so the prelude builds a directory
of symlinks under /kaggle/working and changes into a sibling of it. After that
`../input/<slug>` resolves for every reader -- the notebook, learntools, and
the course's own `os.symlink` of a competition file to a bare `../input/x.csv`,
which then works because this `../input` is writable.

The cost is that the working directory moves, so files the notebook writes land
in /kaggle/working/_nb/ instead of /kaggle/working/. They are still collected.

    python kaggle/_tools/fix_input_paths.py <notebook> [<notebook> ...]
"""

import json
import sys
from pathlib import Path

MARKER = "input layout shim"

PRELUDE = '''# --- input layout shim (added automatically) --------------------------------
# Kaggle mounts attached data either at ../input/<slug>/ or, on API-pushed
# kernels, at /kaggle/input/datasets/<owner>/<slug>/ and
# /kaggle/input/competitions/<slug>/. learntools reads ../input/... itself at
# import time, so the path has to be made real rather than rewritten.
# /kaggle/input is read-only; /kaggle/working is not.
import glob as _glob
import os as _os

# Collect every mounted source, under either layout. The container
# directories themselves are skipped.
_mounted = [p for p in _glob.glob("/kaggle/input/*")
            if _os.path.isdir(p) and _os.path.basename(p) not in ("datasets", "competitions")]
_mounted += _glob.glob("/kaggle/input/datasets/*/*")
_mounted += _glob.glob("/kaggle/input/competitions/*")

# Always relocate, even when ../input/<slug> already resolves: /kaggle/input is
# read-only under BOTH layouts, and some exercises symlink a competition file
# to a bare ../input/train.csv before reading it. That write needs ../input to
# be ours.
# Must be literally "input": ../input from _nb resolves to /kaggle/working/input.
_farm = "/kaggle/working/input"
_here = "/kaggle/working/_nb"
_os.makedirs(_farm, exist_ok=True)
_os.makedirs(_here, exist_ok=True)
for _src in _mounted:
    _dst = _os.path.join(_farm, _os.path.basename(_src))
    if not _os.path.exists(_dst):
        _os.symlink(_src, _dst)
    # Some courses read a bare ../input/<file>.csv, because a single attached
    # dataset used to be mounted with its files directly under ../input. Expose
    # each dataset's own entries at the farm root as well so both spellings work.
    for _child in _glob.glob(_os.path.join(_src, "*")):
        _cdst = _os.path.join(_farm, _os.path.basename(_child))
        if not _os.path.exists(_cdst):
            _os.symlink(_child, _cdst)
_os.chdir(_here)


def _safe_symlink(src, dst):
    """Stand in for os.symlink in the course setup cells.

    Several exercises do

        if not os.path.exists("../input/x.csv"):
            os.symlink("../input/<slug>/x.csv", "../input/x.csv")

    which breaks two ways once ../input is the farm. If the shim already
    exposed x.csv, os.symlink raises FileExistsError -- os.path.exists returns
    False for a dangling link, so the guard does not protect it. And if the
    dataset did not mount, the call happily creates a dangling link and the
    read fails later with a confusing FileNotFoundError.

    Link only when the source is real and the name is free, and never raise.
    """
    try:
        if _os.path.lexists(dst):
            return
        if not _os.path.exists(src):
            return
        _os.symlink(src, dst)
    except OSError:
        pass


print("input shim active:", sorted(_os.listdir(_farm)))
# --- end shim ---------------------------------------------------------------
'''


def fix(path):
    nb = json.loads(Path(path).read_text())
    cells = nb["cells"]
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        if "os.symlink(" in text:
            cell["source"] = text.replace("os.symlink(", "_safe_symlink(").splitlines(keepends=True)
    if cells and MARKER in "".join(cells[0].get("source", [])):
        print(f"  already shimmed  {path}")
        return
    cells.insert(0, {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": PRELUDE.splitlines(keepends=True),
    })
    Path(path).write_text(json.dumps(nb, indent=1) + "\n")
    print(f"  shimmed  {path}")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in sys.argv[1:]:
        fix(p)


if __name__ == "__main__":
    main()
