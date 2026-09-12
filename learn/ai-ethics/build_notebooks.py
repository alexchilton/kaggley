"""Fill the Intro to AI Ethics exercise notebooks.

There is nothing to answer: `learntools/ethics/` defines ex2 to ex5 and every
class in them is a `ThoughtExperiment` with an empty `_solution`. Credit comes
from running the cell, and no answer can be marked wrong.

The one edit this makes is a path fix. Two of the notebooks read a CSV with a
hardcoded `../input/<slug>/...`, which is where Kaggle mounts a dataset for a
notebook forked in the browser. A notebook created by `kaggle kernels push`
gets a different layout -- measured with a probe kernel that walked
`/kaggle/input`:

    /kaggle/input/datasets/alexisbcook/synthetic-credit-card-approval/...

so the hardcoded path raises FileNotFoundError and the run fails before the
first question. This rewrites the `read_csv` call to search for the file by
name instead, which works under either layout and leaves everything else in
the notebook untouched.

    python kaggle/ai-ethics/build_notebooks.py
"""

import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent

FINDER = '''# Kaggle mounts datasets at ../input/<slug>/ for a notebook forked in the
# browser, but at /kaggle/input/datasets/<owner>/<slug>/ for one created by
# `kaggle kernels push`. Find the file by name so either layout works.
import glob as _glob


def _find(filename):
    for pattern in ("../input/**/" + filename, "/kaggle/input/**/" + filename):
        hits = _glob.glob(pattern, recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(filename)


'''

# notebook -> (cell index, the read_csv call to replace, the filename to find)
PATCHES = {
    "exercise-identifying-bias-in-ai.ipynb": (
        2,
        'pd.read_csv("../input/jigsaw-snapshot/data.csv")',
        "data.csv",
    ),
    "exercise-ai-fairness.ipynb": (
        2,
        'pd.read_csv("../input/synthetic-credit-card-approval/synthetic_credit_card_approval.csv")',
        "synthetic_credit_card_approval.csv",
    ),
}

PLAIN = {
    "ex1": "exercise-human-centered-design-for-ai.ipynb",
    "ex2": "exercise-identifying-bias-in-ai.ipynb",
    "ex3": "exercise-ai-fairness.ipynb",
    "ex4": "exercise-model-cards.ipynb",
}


def main():
    for name, filename in PLAIN.items():
        src = HERE / "sources" / filename
        out = HERE / name / filename
        if filename not in PATCHES:
            shutil.copy(src, out)
            print(f"copied  {out.relative_to(HERE.parents[1])}  (no changes needed)")
            continue

        index, call, target = PATCHES[filename]
        nb = json.loads(src.read_text())
        cell = nb["cells"][index]
        text = "".join(cell["source"])
        if call not in text:
            raise SystemExit(f"{filename}: cell {index} no longer contains {call!r}")
        text = FINDER + text.replace(call, f'pd.read_csv(_find("{target}"))')
        cell["source"] = text.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None
        out.write_text(json.dumps(nb, indent=1) + "\n")
        print(f"patched {out.relative_to(HERE.parents[1])}  (dataset path)")


if __name__ == "__main__":
    main()
