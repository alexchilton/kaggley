"""Train one "what next" candidate member locally on the notebook's own folds and features; save OOF + fold-mean test.

    python3 -u scratch/whatnext/member.py <name> <threads>
Runs the notebook's setup cells (built notebook, kernels/s6e11-first-look) with run() as a no-op, then the candidate.
Output: scratch/whatnext/<name>_oof.npy, <name>_test.npy
"""
import json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
NAME, THR = sys.argv[1], int(sys.argv[2])
import matplotlib; matplotlib.use('Agg')
nb = json.load(open(ROOT / 'kernels/s6e11-first-look/s6e11-first-look.ipynb'))
src = {i: ''.join(c['source']) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code'}
os.chdir(ROOT / 'kernels/s6e11-first-look')
ns = {'__name__': '__main__', 'display': print}
def ex(i, text=None):
    exec(compile(text or src[i], f'cell{i}', 'exec'), ns)
ex(2, src[2].replace('THREADS = os.cpu_count()', f'THREADS = {THR}'))
ex(9, src[9].split('fig, axs')[0])                                   # y, num, cat, boo; no plots
ex(13, src[13].split('adv = pd.concat')[0])                          # prep, X, XT; no adversarial check
ex(18)
ex(21, src[21].replace('__FP_CORE__', ''))
ns['run'] = lambda *a, **k: None
for i in (36, 39):
    ex(i)
g = ns
import numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score

def lgb4(lr=0.03, es=300, **kw):
    return lgb.LGBMClassifier(n_estimators=60000, learning_rate=lr, num_leaves=4, min_child_samples=100, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.5, reg_lambda=1.0, n_jobs=THR, verbose=-1, **kw), es

def fit(m_es, Xa, ya, Xb, yb, Xt):
    m, es = m_es
    m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric='auc', callbacks=[lgb.early_stopping(es, verbose=False)])
    print(f'    best iteration {m.best_iteration_}', flush=True)
    return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1]

# 1. digit keys and triple crosses in one member
def digits_triples(Xa, ya, Xb, yb, Xt):
    Xa, Xb, Xt = g['plus_features'](Xa, ya, Xb, Xt)
    Xa, Xb, Xt = g['digit_features'](Xa, ya, Xb, Xt)
    return fit(lgb4(), Xa, ya, Xb, yb, Xt)

# 2. digit keys on the integer columns and the count encodings as well
INTS = ['txn_count_last_24h', 'card_age_months', 'customer_age', 'cvv_retry_count', 'time_of_day_hour', 'day_of_week', 'prior_disputes']
def more_digits(Xa, ya, Xb, yb, Xt):
    from sklearn.preprocessing import TargetEncoder
    import pandas as pd
    Xa, Xb, Xt = g['plus_features'](Xa, ya, Xb, Xt)                  # adds the cnt_ columns
    cnts = [c for c in Xa.columns if c.startswith('cnt_')]
    def keys(d):
        k = {}
        for c in INTS + cnts:
            i = np.floor(d[c].astype(float).values + 1e-9).astype(np.int64); s = np.abs(i).astype(str)
            k[f'ten_{c}'] = i // 10; k[f'lead1_{c}'] = np.array([v[:1] for v in s]); k[f'lead2_{c}'] = np.array([v[:2] for v in s])
            k[f'len_{c}'] = np.array([len(v) for v in s]); k[f'last_{c}'] = np.abs(i) % 10
            if c.startswith('cnt_'): k[f'val_{c}'] = i
        return pd.DataFrame(k).astype(str)
    ka, kb, kt = keys(Xa), keys(Xb), keys(Xt)
    enc = TargetEncoder(target_type='binary', cv=5, smooth=20.0, random_state=0)
    cols = ['ite_' + c for c in ka.columns]
    te = [enc.fit_transform(ka, ya), enc.transform(kb), enc.transform(kt)]
    Xa, Xb, Xt = [pd.concat([d.reset_index(drop=True), pd.DataFrame(t, columns=cols).astype('float32')], axis=1) for d, t in zip((Xa, Xb, Xt), te)]
    Xa, Xb, Xt = g['digit_features'](Xa, ya, Xb, Xt)
    return fit(lgb4(), Xa, ya, Xb, yb, Xt)

# 3. the digit member at a lower learning rate
def digits_lr01(Xa, ya, Xb, yb, Xt):
    Xa, Xb, Xt = g['digit_features'](Xa, ya, Xb, Xt)
    return fit(lgb4(lr=0.01, es=1000), Xa, ya, Xb, yb, Xt)

# 4. the digit member with interaction constraints: a tree splits only within one source column's group
def digits_ic(Xa, ya, Xb, yb, Xt):
    Xa, Xb, Xt = g['digit_features'](Xa, ya, Xb, Xt)
    base = [c for c in g['X'].columns]
    groups = {c: [] for c in base}; alone = []
    for j, col in enumerate(Xa.columns):
        hit = [c for c in base if col == c or (col.endswith('_' + c) and '__' not in col)]
        (groups[max(hit, key=len)] if hit else alone).append(j)
    ic = [v for v in groups.values() if v] + [[j] for j in alone]
    print(f'    {len(ic)} groups, {len(alone)} columns alone (pairs, orig model, ...)', flush=True)
    return fit(lgb4(interaction_constraints=ic), Xa, ya, Xb, yb, Xt)

f = {'digits_triples': digits_triples, 'more_digits': more_digits, 'digits_lr01': digits_lr01, 'digits_ic': digits_ic}[NAME]
X, y, XT, FOLDS, frames = g['X'], g['y'], g['XT'], g['FOLDS'], g['frames']
if os.environ.get('SMOKE'):                                           # every candidate's features on 20k rows, 50 trees
    tr_i, va_i = FOLDS[0]; Xa, Xb, Xt = frames(0, 'fe'); ya, yb = y.iloc[tr_i].values, y.iloc[va_i].values
    Xa, ya, Xb, yb, Xt = Xa.iloc[:20000], ya[:20000], Xb.iloc[:20000], yb[:20000], Xt.iloc[:20000]
    _lgb4 = lgb4; lgb4 = lambda **kw: (_lgb4(**kw)[0].set_params(n_estimators=50), 10)
    for n, fn in [('digits_triples', digits_triples), ('more_digits', more_digits), ('digits_lr01', digits_lr01), ('digits_ic', digits_ic)]:
        vo, tp = fn(Xa, ya, Xb, yb, Xt); print('SMOKE', n, round(roc_auc_score(yb, vo), 4), vo.shape, tp.shape, flush=True)
    sys.exit(0)
t0 = time.time(); o = np.zeros(len(X)); p = np.zeros(len(XT))
for k, (tr_i, va_i) in enumerate(FOLDS):
    Xa, Xb, Xt = frames(k, 'fe')
    vo, tp = f(Xa, y.iloc[tr_i].values, Xb, y.iloc[va_i].values, Xt)
    o[va_i] = vo; p += tp / len(FOLDS)
    print(f'{NAME} fold {k}: AUC {roc_auc_score(y.iloc[va_i], vo):.5f} ({Xa.shape[1]} inputs before the member adds its own) {time.time() - t0:.0f}s', flush=True)
out = ROOT / 'scratch/whatnext'
np.save(out / f'{NAME}_oof.npy', o); np.save(out / f'{NAME}_test.npy', p)
print(f'{NAME}: OOF AUC {roc_auc_score(y, o):.5f} ({(time.time() - t0) / 60:.1f} min)', flush=True)
