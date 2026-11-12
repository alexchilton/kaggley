"""Publish a finished notebook version's member predictions as the public dataset alexchilton/s6e11-oof-predictions.

    python3 tools/publish_oof.py v8                       # downloads the kernel's latest output, versions (or creates) the dataset
    python3 tools/publish_oof.py v6 --rekey mapping.json  # files already in out/kernel_output, saved before the fingerprint
                                                          # cache, renamed to the fingerprints tools/rekey_oof.py computed

Keeps oof_*.npy (out-of-fold, train row order), test_*.npy (test row order) and oof_manifest.json; writes the description
from the manifest so the dataset page lists every member with its OOF AUC.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNEL = "alexchilton/credit-card-fraud-lb-0-9530-shallow-tree-stack"
DATASET = "alexchilton/s6e11-oof-predictions"


def main(version, rekey=None):
    raw, out = ROOT / "out/kernel_output", ROOT / "out/oof_dataset"
    shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
    if rekey is None:
        shutil.rmtree(raw, ignore_errors=True); raw.mkdir(parents=True)
        subprocess.run(["kaggle", "kernels", "output", KERNEL, "-p", str(raw)], check=True)
    files = sorted(raw.glob("oof_*.npy")) + sorted(raw.glob("test_*.npy"))
    man_p = raw / "oof_manifest.json"
    if rekey is not None:                              # legacy files -> fingerprint names; AUC recomputed from the OOF file
        import numpy as np
        import pandas as pd
        from sklearn.metrics import roc_auc_score
        y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].values
        import re
        log = "".join(x.get("data", "") for x in json.loads(next(raw.glob("*.log")).read_text()))
        mins = {m.group(1): float(m.group(2)) for m in re.finditer(r"^(\S.*?): OOF AUC [0-9.]+  \(([0-9.]+) min", log, re.M)}
        man = {}
        for r in json.loads(Path(rekey).read_text()):
            src = raw / f"oof_{r['legacy_key']}.npy"
            if not src.exists():
                print("not in this output:", r["name"], r["mode"]); continue
            shutil.copy(src, out / f"oof_{r['fp']}.npy"); shutil.copy(raw / f"test_{r['legacy_key']}.npy", out / f"test_{r['fp']}.npy")
            man[r["fp"]] = dict(name=r["name"], inputs=r["mode"], oof_auc=round(float(roc_auc_score(y, np.load(src))), 5), minutes=mins.get(r["name"], 0.0), version=version)
        files = sorted(out.glob("*.npy"))
    else:
        for f in files + sorted(raw.glob("optuna_*.json")):   # the Optuna search's trials, so the next version loads them
            shutil.copy(f, out / f.name)
    if rekey is not None:
        pass                                           # manifest built above
    elif man_p.exists():
        man = json.loads(man_p.read_text())
    else:                                              # versions before the fingerprint cache: files named <member>_<inputs>
        import numpy as np
        import pandas as pd
        y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].values
        from sklearn.metrics import roc_auc_score
        man = {f.stem[4:]: dict(name=f.stem[4:].rsplit("_", 1)[0].replace("_", " "), inputs=f.stem.rsplit("_", 1)[1],
                                oof_auc=round(float(roc_auc_score(y, np.load(f))), 5), version=version, legacy=True)
               for f in sorted(raw.glob("oof_*.npy"))}
    (out / "oof_manifest.json").write_text(json.dumps(man, indent=1))
    rows = "\n".join(f"| {m['name']} | {m['inputs']} | {m['oof_auc']:.5f} | {m['version']} | `oof_{k}.npy`, `test_{k}.npy` |"
                     for k, m in sorted(man.items(), key=lambda kv: -kv[1]["oof_auc"]))
    desc = f"""Out-of-fold and test predictions of every model in the notebook
[Credit Card Fraud | Shallow-Tree Stack](https://www.kaggle.com/code/{KERNEL}) for Playground Series S6E11.

**Use them to blend without training.** Each `oof_*.npy` holds one model's predictions for the 700k training rows, in `train.csv` order,
each made by a model that never saw that row (5-fold StratifiedKFold, `random_state=42`). Each `test_*.npy` holds the same model's
test predictions (mean of the 5 fold models), in `test.csv` order. Blend on the OOF files, apply the weights to the test files.

```python
import json, numpy as np
from pathlib import Path
D = next(Path('/kaggle/input').rglob('oof_manifest.json')).parent
man = json.loads((D / 'oof_manifest.json').read_text())
oof = {{m['name']: np.load(D / f'oof_{{k}}.npy') for k, m in man.items()}}
test = {{m['name']: np.load(D / f'test_{{k}}.npy') for k, m in man.items()}}
```

AUC only sees order, so rank-transform before averaging. Keys are fingerprints of (model, inputs, code, settings, features):
the notebook reuses a model from here only when all of those are unchanged.

| model | inputs | OOF AUC | trained in | files |
|---|---|---|---|---|
{rows}

`inputs`: `raw` = the 24 columns as given; `fe` = plus in-fold target encodings, decimal digits and original-data lookups (see the notebook).
"""
    meta = {"title": f"S6E11 OOF + test predictions ({len(man)} models)", "id": DATASET, "licenses": [{"name": "CC0-1.0"}], "description": desc,
            "keywords": ["tabular", "classification"], "isPrivate": False}
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(files)} files, {len(man)} members")
    exists = subprocess.run(["kaggle", "datasets", "status", DATASET], capture_output=True, text=True).returncode == 0
    cmd = (["kaggle", "datasets", "version", "-p", str(out), "-m", f"notebook {version}"] if exists
           else ["kaggle", "datasets", "create", "-p", str(out), "--public"])
    print(" ".join(cmd)); subprocess.run(cmd, check=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[a.index("--rekey") + 1] if "--rekey" in a else None)
