# Review of tools/make_nb.py v3 (S6E11)

Read-only review. No training was run.
One measurement was made: pairs kept by the near-unique filter = **264 of 276** (local data/, 2026-11-01).
So `pairs` mode is 24 raw + 5 digits + 288 TE = **317 inputs**.
Runtimes below are scaled from your one-fold timings and are **ESTIMATES (unmeasured)**.

## Bugs and leakage risks, ranked

1. **The runtime is the main risk, not leakage.** `frames()` (l.200-211) refits the 288-column TargetEncoder in every fold for every model, which is 20 refits.
   Compute the frames once per fold and cache them.
2. **MODE has no noise margin** (l.238). A gain of +0.00002 picks `pairs`, and that sends CatBoost and RealMLP onto 317 inputs.
3. **The ledger is against all-pair crosses**: s6e8 -0.003580, s6e10 -0.000391, s6e9 +0.000169.
4. **Singles TE encodes noise columns** (l.184): account_balance_usd has 16 rows per level and signal ratio 1.1; amount_usd has 52 rows and 2.0.
5. Leakage: none found. The inner TE is cross-fitted, and the factorize over train+test is unsupervised.
   Early stopping and the hill climb both use the same OOF, so the OOF is slightly optimistic (about 1e-4, typical).
6. `dataset_sources` is empty (l.321). The original data cannot be mounted until it is added.

## Fit in 12 h on 4 CPU cores?

Not if MODE is `pairs`.
CatBoost needs about 7 min per fold at 24 features; at 317 features that is roughly 1 h or more per fold, so 5-7 h.
RealMLP on CPU with 317 inputs needs several hours.
Add LightGBM x3 (~35 min), XGBoost (~35 min) and the TE refits (~30 min), and the total is **9-13 h**.
Cut these:
- run CatBoost and RealMLP on raw + singles only;
- add an elapsed-time guard that skips a model past 9 h;
- save OOF and test predictions to disk after each model.

## Highest-value changes for this release

1. **Targeted TE instead of all pairs.**
   Change: exact-value TE of merchant_risk_score, velocity_score and customer_age, plus integer-rounded keys and pairs only among the top-10 signal columns.
   Mechanism: exact-value pooling is something range splits cannot reproduce.
   Disproved if `singles` does not beat `raw` by at least 0.0002 OOF.
2. **Original 20k rows as lookups, not as appended rows.**
   Change: per-value fraud rates from the original data (100% of test levels for those 3 columns appear in it), plus an original-only LightGBM prediction as a feature.
   Mechanism: these give label statistics that do not depend on the synthetic noise. Public s6e10 notebooks measured +0.0006 to +0.0010; appending rows measured -0.000245.
   Disproved if the LightGBM OOF delta is below 0.0002.
3. **Cache the frames, add the margin, add the time guard.** This makes the release finish.
4. Optional if time is left: 10 folds. The s6e9 winners measured about +0.0001 per model.

## The 0.925 public vs ~0.944 CV gap

Adversarial AUC is 0.5004, so there is no covariate shift.
With about 1,000 frauds on a 20% public split, the AUC standard error is about 0.005, and the gap is about 4 SE.
**HYPOTHESIS (unverified):** the public notebook is weak, not the board.
Test: run its code under our 5 folds. If it gets about 0.944 CV, the board has a label shift, and our first submission will show the same gap.
