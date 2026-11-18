# HANDOFF (state, not a log; overwrite it)

Written 2026-11-17 18:10 CET from `docs/cron/2026-11-17.md`.
The daily cron writes the detailed state: read the newest `docs/cron/<date>.md` first.

## Board

- Our best public LB: 0.95417 (`submission_s417_wide10.csv`, v30).
- Our own stack: 0.95279 (`submission_wide_lgbglm.csv`, v32, Kaggle version 36); v32 `submission.csv` 0.95257 is the 65/35 mix, not the wide stack.
- Best public notebook: 0.95432 (goodpjw2008, LM Ladder), on 2026-11-17 07:02 UTC.
- Notebook: `alexchilton/credit-card-fraud-lb-0-9542-shallow-tree-stack`, 65 votes total (not counted).

## How it runs

- launchd `com.alexchilton.s6e11-daily` runs `tools/cron/s6e11_daily.sh` with the prompt `tools/cron/daily_prompt.md`.
- The cron edits `tools/make_nb.py`, pushes the kernel, waits, runs `tools/publish_oof.py v<N>`, submits and writes `docs/cron/<date>.md`.
- No git remote (memory: playgrounds are copied to the public kaggley repo).

## Running now

Nothing: no kernel runs between cron runs. Verify with `kaggle kernels list --mine --sort-by dateRun --page-size 10`.

## Waiting on Alex (carried by the cron since 2026-11-16)

1. The launcher's silver test compares three different slugs (`counted.json` `...lb-0-9530-...`, `tools/cron/s6e11_daily.sh` line 7 `...lb-0-9524-...`, the kernel `...lb-0-9542-...`), so it cannot fire.
2. Optuna cache: `publish_oof.py` drops the `optuna_*.json` files when a version loads both searches, so v33 may search again and retrain CatBoost tuned. A fix in `publish_oof.py` needs Alex's yes.

## Next (v33, from the 11-17 report)

1. Freeze yekenot's RealMLP bank (it changed upstream), so `submission.csv` returns to the wide stack.
2. Correct the v32 noise sentence (the measured pair is 3.66e-4 apart, not 0.79e-4).
3. Write v32's results into the notebook.

## Clean-up

2026-11-17: 14 scratch/out/logs entries and the 12 root `v*o/` Kaggle output downloads (1.5 GB) went to the Trash; list in `docs/cleanup_2026-11-17.md`.
