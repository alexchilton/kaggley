# Playground Series S6E11: credit-card fraud

Kaggle competition: https://www.kaggle.com/competitions/playground-series-s6e11 — predict the probability that a card transaction is fraud; metric ROC AUC.

## Now

- Best public LB **0.95211** (version 17). Versions 18-21 added public prediction banks and members and scored 0.95190-0.95202, below it.
- The notebook: [Credit Card Fraud | LB 0.9521 | Shallow-Tree Stack](https://www.kaggle.com/code/alexchilton/credit-card-fraud-lb-0-9521-shallow-tree-stack), built by `tools/make_nb.py` (`kernels/s6e11-first-look/`).
  It is a tutorial: every version says what it changed and why, and section 4 keeps every version's LB score and the ideas that did not work, with numbers.
- Companion notebook: [Four months, same screens: what held](https://www.kaggle.com/code/alexchilton/four-months-same-screens-what-held-s6e11) — the noise floor an idea must beat, across four playgrounds.
- All model predictions are published as the dataset [s6e11-oof-predictions](https://www.kaggle.com/datasets/alexchilton/s6e11-oof-predictions).

## How it works

    features   in-fold target encoding (per value, pairs, triple crosses), decimal digits of the numeric columns, the original dataset as a lookup
    members    shallow LightGBM / XGBoost / CatBoost (depth 1-3, Optuna-tuned on one fold), a spline GLM, RealMLP and TabM, a digit CNN, public prediction banks
    stack      out-of-fold predictions turned into ranks; a logistic stacker and a hill climb, scored on rows they did not fit
    test side  full-data refits for the strongest members

## What failed and why

- **Deep trees overfit a near-additive target.** 4-leaf / depth-2 trees took v4's 0.94865 to 0.94960 (v5); the decimal-digit keys of v10 gave the largest gain since (+0.0009). In a synthetic playground, ask how the generator wrote the numbers.
- **Better versions of a member the stack already has add nothing.** The four "what next" ideas of v10 each made a slightly better member (rank correlation 0.991-0.998 with an existing one) and moved the stack by -0.00006 to +0.00003.
- **Members are judged by rank correlation first**: below about 0.99 with every member, or they will not add. The digit CNN (0.9795) passed; a GLM + shallow-tree blend (0.9938) did not.
- **More banks stopped helping on the board** after v17: the stacker's held-out AUC rose (0.95212 -> 0.95235) while the LB fell (0.95211 -> 0.95190). The public LB scores only part of the test rows; a few 0.0001 can be noise.
- **A bug worth knowing:** summing fold weights without normalising them put each fold on a different scale; the pooled AUC read 0.895. When an honest re-score is far below the in-sample one, suspect the code before the model.

## Map

- `tools/make_nb.py` — builds the notebook; the version history table is in it.
- `tools/publish_oof.py`, `tools/rekey_oof.py` — the prediction dataset. `tools/whatnext_member.py`, `tools/whatnext_addon.py` — the v20 member tests.
- `tools/cron/` — the daily run's instructions and launcher. `docs/cron/` — its reports. `docs/fable_review_v3.md`, `docs/fable_review_v7.md` — two external reviews.
- The working copy lives in a separate local folder; the cron copies the tracked files here.
