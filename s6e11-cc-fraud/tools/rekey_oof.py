"""Fingerprints of every run() member of a notebook, computed from its source alone (tools/fp_core.py, standard library only).

    uv run --python 3.13 --no-project python tools/rekey_oof.py NOTEBOOK.ipynb [--mode fe] > mapping.json

Run it with the notebook's Python (Kaggle: 3.13), since ast.dump differs between minor versions. --mode is the value MODE took
when the notebook ran (the members called without a mode use it). Prints JSON: [{name, mode, fn, fp, legacy_key}], where
legacy_key is the file key versions before the fingerprint cache used (oof_<name>_<mode>.npy).
"""
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fp_core import fp_compute, fp_feature_cell


def main(path, mode_default):
    nb = json.loads(Path(path).read_text())
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    py = ["\n".join(l for l in c.splitlines() if not l.lstrip().startswith("%")) for c in cells]   # %pip lines are not Python
    funcs, settings, calls = {}, {}, []
    for src in py:
        tree = ast.parse(src)
        for node in tree.body:                            # top level, in notebook order (a later def replaces an earlier one)
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                funcs[node.name] = ast.get_source_segment(src, node)
            elif isinstance(node, ast.Assign):
                tg = node.targets[0]
                names = [tg.id] if isinstance(tg, ast.Name) else [e.id for e in tg.elts if isinstance(e, ast.Name)] if isinstance(tg, ast.Tuple) else []
                try:
                    val = ast.literal_eval(node.value)
                except Exception:
                    continue
                vals = [val] if len(names) == 1 else list(val) if isinstance(val, tuple) and len(val) == len(names) else []
                for n, v in zip(names, vals):
                    if type(v) in (int, float, str, bool, dict):
                        settings[n] = v
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run" and len(node.args) >= 2:
                name, fn = node.args[0].value, node.args[1].id
                mode = node.args[2].value if len(node.args) > 2 else mode_default
                calls.append((name, fn, mode))
    feature_src = fp_feature_cell(cells)
    out = [dict(name=n, mode=m, fn=f, fp=fp_compute(n, m, f, funcs, settings, feature_src),
                legacy_key=''.join(ch if ch.isalnum() else '_' for ch in n) + f"_{m}") for n, f, m in calls]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[a.index("--mode") + 1] if "--mode" in a else "fe")
