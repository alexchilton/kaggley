"""S6E11 TabICL member: out-of-fold predictions on our folds (StratifiedKFold(5, shuffle=True, random_state=42) on the
train rows in file order, as tools/make_nb.py) and test predictions, for the rank-correlation screen.

Context per fit: every fraud row of the fitting rows plus negatives up to CONTEXT rows (AUC is rank-based, the prior
shift does not matter for it). Hard budget: BUDGET_MIN minutes; whatever is done by then is saved, with NaN elsewhere.
"""
import json, os, subprocess, sys, time
T0 = time.time()
BUDGET_MIN, CONTEXT, N_EST, CHUNK = 140, 60000, 4, 20000
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabicl"], check=True)
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tabicl import TabICLClassifier

root = next(p for p in ("/kaggle/input/competitions/playground-series-s6e11", "/kaggle/input/playground-series-s6e11") if os.path.exists(p))
train, test = pd.read_csv(f"{root}/train.csv"), pd.read_csv(f"{root}/test.csv")
y = train["is_fraud"].astype(int).values
feats = [c for c in train.columns if c not in ("id", "is_fraud")]
def prep(d):
    X = d[feats].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
        elif not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.Categorical(X[c], categories=sorted(pd.concat([train[c], test[c]]).astype(str).unique())).codes
    return X.values.astype(np.float32)
X, XT = prep(train), prep(test)
left = lambda: BUDGET_MIN - (time.time() - T0) / 60
rng = np.random.default_rng(0)
def context(idx):
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    neg = rng.choice(neg, min(len(neg), CONTEXT - len(pos)), replace=False)
    return np.concatenate([pos, neg])
def fit_predict(ctx, rows_X):
    m = TabICLClassifier(n_estimators=N_EST, random_state=0)
    m.fit(X[ctx], y[ctx])
    out = np.full(len(rows_X), np.nan)
    for s in range(0, len(rows_X), CHUNK):
        if left() < 3:
            print(f"budget: stopped at row {s} of {len(rows_X)}", flush=True); break
        out[s:s + CHUNK] = m.predict_proba(rows_X[s:s + CHUNK])[:, 1]
        print(f"  rows {s + CHUNK:,} of {len(rows_X):,}, {(time.time() - T0) / 60:.1f} min", flush=True)
    return out

t = time.time()                                   # smoke: the API and the speed on 20k rows before the real run
sm = np.arange(0, len(y), 25)
m = TabICLClassifier(n_estimators=N_EST, random_state=0).fit(X[sm[:5000]], y[sm[:5000]])
p = m.predict_proba(X[sm[5000:7000]])[:, 1]
print(f"smoke: 5,000-row context, 2,000 rows predicted in {time.time() - t:.1f} s; AUC {roc_auc_score(y[sm[5000:7000]], p):.4f}", flush=True)

oof = np.full(len(y), np.nan); log = []
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    if left() < 25:
        print(f"fold {k}: skipped, {left():.0f} min left (test needs its share)", flush=True); break
    oof[va] = fit_predict(context(tr), X[va])
    ok = ~np.isnan(oof[va])
    auc = roc_auc_score(y[va][ok], oof[va][ok]) if ok.sum() > 100 else None
    log.append(dict(fold=k, rows=int(ok.sum()), auc=auc, minutes=round((time.time() - T0) / 60, 1))); print(log[-1], flush=True)
tp = fit_predict(context(np.arange(len(y))), XT)
ok = ~np.isnan(oof)
summary = dict(oof_rows=int(ok.sum()), oof_auc=roc_auc_score(y[ok], oof[ok]) if ok.sum() > 100 else None,
               test_rows=int((~np.isnan(tp)).sum()), folds=log, context=CONTEXT, n_estimators=N_EST, minutes=round((time.time() - T0) / 60, 1))
np.save("tabicl_oof.npy", oof); np.save("tabicl_test.npy", tp)
pd.DataFrame({"id": train["id"], "tabicl": oof}).to_csv("tabicl_oof.csv", index=False)
pd.DataFrame({"id": test["id"], "tabicl": tp}).to_csv("tabicl_test.csv", index=False)
json.dump(summary, open("tabicl_summary.json", "w"), indent=1); print(summary, flush=True)
