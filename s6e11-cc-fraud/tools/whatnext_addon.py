# Offline add-on test for the "what next" members: v19's intended 59 members (scratch/banks/addon.py's 58 + sameershaikh27's
# ten-fold bank) plus one candidate at a time, then all candidates above the bar together. Same stacker and split as addon.py.
import sys, os
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[2])
src = open('scratch/banks/addon.py').read().split('CAND = {}')[0]
exec(src)
npy('bank sameershaikh27 GLM boost, ten folds', 'sameer10/gb10_bag_oof.npy', 'sameer10/gb10_bag_full.npy')
base_names = list(oof)
import numpy as np
from sklearn.metrics import roc_auc_score
W = 'scratch/whatnext/'
CAND = {n: (np.load(f'{W}{n}_oof.npy'), np.load(f'{W}{n}_test.npy')) for n in ['digits_triples', 'more_digits', 'digits_lr01', 'digits_ic', 'tabicl'] if Path(f'{W}{n}_oof.npy').exists()}
def lr(v):
    r = rankdata(v) / len(v); return logit(np.clip(r, 1e-6, 1 - 1e-6))
def score(Z, seed=7):
    s = np.zeros(len(Z))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(Z, y):
        s[va] = LogisticRegression(C=0.05, max_iter=1000).fit(Z[tr], y[tr]).decision_function(Z[va])
    return roc_auc_score(y, s)
Z0 = np.column_stack([lr(oof[k]) for k in base_names])
print('base members', len(base_names), flush=True)
base = {s: score(Z0, s) for s in (7, 11)}
print('base', {s: round(v, 5) for s, v in base.items()}, flush=True)
ref = {k: roc_auc_score(y, oof[k]) for k in ['LightGBM 4 leaves digits', 'LightGBM 4 leaves plus']}
print('reference members:', {k: round(v, 5) for k, v in ref.items()})
keep = []
for n, (o, _) in CAND.items():
    rho = max(np.corrcoef(rankdata(o), rankdata(oof[k]))[0, 1] for k in base_names)
    d = {s: score(np.column_stack([Z0, lr(o)]), s) - base[s] for s in base}
    print(f'{n:16s} OOF {roc_auc_score(y, o):.5f}  max rho {rho:.4f}  delta split7 {d[7]:+.5f}  split11 {d[11]:+.5f}', flush=True)
    if min(d.values()) >= 0.00005: keep.append(n)
if len(keep) > 1:
    Z = np.column_stack([Z0] + [lr(CAND[n][0]) for n in keep])
    print('together', keep, {s: round(score(Z, s) - base[s], 5) for s in base})
print('KEEP', keep)
