# Fable review v7: S6E11 notebook, smart-buildings RL, Squares 101

Read-only review, 2026-11-01 about 12:45-13:15 CET.
Nothing was edited, pushed, trained or submitted.
Two read-only Kaggle API calls were made (results in section 1.1).

Skipped, said up front:
- I did not run the v7 notebook, so every runtime below is an ESTIMATE (unmeasured).
- `tools/make_nb.py` was being edited while I read it (mtime 12:45:10, during this review).
  The `xgb` NameError I found in the shape-function cell was already fixed in that save.
  Everything below refers to the 12:45:10 version.
- Vote effects are judgement, not measurement. Each one is marked HYPOTHESIS.

## Priority list across all three

| # | what | why it earns votes or score | cost |
|---|---|---|---|
| 1 | S6E11: fix the four stale text spots and make the lesson numbers come from this run (1.2) | the notebook says "every number below is computed in this notebook", but the lessons quote v5 numbers. An expert who checks will find text that disagrees with the output | 30 min |
| 2 | S6E11: publish the OOF dataset from v6 output now, before v7 is pushed (1.1, item 1) | a free blending resource is what people fork and upvote (HYPOTHESIS). It also stops the v7 push from referring to a dataset that does not exist | 15 min plus the v6 finish |
| 3 | Smart buildings: add one hero figure, "what the RL changed on one cold weekday", and cut the notebook's lab-journal text by half (2.3) | at the moment the tutorial shows scores, not behaviour. An RL tutorial with no plot of what the policy does is the main gap | half a day, no training |
| 4 | Smart buildings: replace the four PPO knob arms with two cheaper tests, occupancy reseed and checkpoint action-ensembling (2.2) | both act directly on the overfitting the plan is trying to fix. The knob arms are likely below what the selection set can resolve | reseed is 2 lines plus 1 run; the ensemble needs evaluation only |
| 5 | Squares 101: fix the structure and stale numbers, then add one new section "where the remaining points are", with a worked fault-line DP figure (3) | the notebook is now mostly a map to other people's work. The fault-line DP is the method that matters above n = 324, and the notebook explains it only by link | 3-4 h |

## 1. S6E11 notebook (tools/make_nb.py, v7)

### 1.1 Bugs and risks, ranked

1. **The prediction cache will not hit on the first v7 run, and its dataset probably does not exist yet.**
   `kaggle datasets status alexchilton/s6e11-oof-predictions` returned **403 Forbidden** (13:05 CET).
   HYPOTHESIS: this means the dataset has not been created.
   `kernel-metadata.json` lists it in `dataset_sources`.
   I have not checked what the Kaggle push does with a missing source; it may refuse the push, or drop the source silently.
   Also: v6 writes files named by member (`oof_<name>_<inputs>.npy`). `tools/publish_oof.py` handles that through its `legacy` branch, but those keys are names, not fingerprints.
   So no v7 fingerprint can match, and **v7 retrains every member**. The cache starts to pay only from v8.
   Do this: run `publish_oof.py v6` when v6 finishes, so the dataset exists and has value from now on, then push v7.
   Say in the notebook text that the cache fills from v7's output.
2. **The Optuna search makes the tuned member impossible to cache, and it costs up to 30 min on every run.**
   `study.optimize(objective, n_trials=30, timeout=1800)` (make_nb.py:446).
   With `timeout`, the number of finished trials depends on the hardware, so `BEST` can change between runs.
   `BEST` is a dict global that the fingerprint hashes, so `LightGBM tuned` retrains each time.
   The search itself is not cached and is not under the 9 h budget guard.
   Fix: drop `timeout` (fixed `n_trials`), add `deterministic=True` to the LightGBM settings, and cache `study.best_params` in the manifest (or write the found values into the builder after the first run).
3. **`build(mlp_epochs=0)` now also drops TabM.**
   The filter removes any cell that contains `RealMLP_TD_Classifier` (make_nb.py:713), and TabM is in the same cell (543-570).
   This is harmless at the default `mlp_epochs=16`, but the switch no longer does what its comment says.
4. **The Spline GLM is not run through `run()`** (make_nb.py:486-491).
   So it is not fingerprinted, not saved by `_publish`, and not in the published dataset.
   The dataset text says "every member's out-of-fold and test predictions are published", which is false for this member.
5. **The fingerprint does not see functions called by functions.**
   `lgbm_orig` falls back to `lgbm`, a function global, and `_code_sig` hashes only the function's own bytecode plus scalar and container globals.
   If `lgbm` is edited, `lgbm_orig`'s cached predictions are still used.
   This is minor; one line fixes it (recurse into function-type globals).
6. **The runtime grows, and the guard does not cover the new cells.**
   v7 adds the num_leaves curve (6 fits), the depth-1 shape model, Optuna (up to 30 min) and `LightGBM tuned` at lr 0.03 with up to 30,000 rounds.
   ESTIMATE: about +1.5 h on top of v6.
   v6 was still RUNNING at 13:05 CET (`kaggle kernels status`).
   Read v6's total runtime from its log before you push v7. The members last in the order (CatBoost, RealMLP, TabM) are the ones the 9 h guard will skip first.
7. Leakage: none found in the new cells.
   The leak demo is correct: "leaky" fits the encoder on all 200k dev rows (validation labels included); "honest" cross-fits on training rows and only transforms validation and hold-out rows.
   Optuna tunes on fold 0's validation rows, and the text says so. This is honest enough.

### 1.2 Stale or inconsistent text (fix before the push)

- Intro: "four things", "LightGBM, XGBoost, CatBoost and RealMLP". The notebook now has 17 members, an Optuna section and a version history.
- Pipeline figure: `'15 models'` (make_nb.py:198). I count 17: 7 LightGBM (raw, +features, 4 leaves, 4 leaves raw, 4 leaves b, tuned, from original), Spline GLM, 5 XGBoost, 2 CatBoost, RealMLP, TabM.
- Section 5: "Tuning: each model is untuned here" (make_nb.py:694) contradicts the new Optuna section.
- Version table: v6 is "running" and there is no v7 row. The LB chart stops at v5 (make_nb.py:685).
- The three lessons quote v5 numbers as if they were measured in this notebook ("RealMLP scored 0.94704 ... weight 0").
  If v7's numbers differ, the notebook contradicts its own output.
  Fix: after the blend cell, print the lessons with f-strings from `oof`, `w` and `drop`, and keep the v5 numbers only in the version table.

### 1.3 What would make it stand out as a tutorial

The v7 additions are strong: the leak demo (CV against a hold-out), the AUC-vs-num_leaves curve, the shape functions and the drop-one chart each teach one thing with one picture.
Three changes would make it read as a tutorial, not as a log:

1. **A top block of "5 lessons, 5 numbers", each linked to its section.**
   Example: "a leaky target encoder: CV X, hold-out Y"; "4 leaves beat 63: 0.94838 vs 0.94701"; "a model with OOF 0.947 got weight 0"; "only N of 17 members matter (drop-one)"; "hill climb vs stacker tie on held-out rows".
   Experts skim the top. At the moment the top tells them the notebook covers EDA and a 4-model baseline.
2. **Join the num_leaves curve to the shape functions in the text.**
   The curve says the target is nearly additive; the depth-1 shape functions then show the additive parts.
   One sentence connecting them is the notebook's central insight.
3. **Put the OOF dataset in the title area and at the end, with a 6-line "blend these yourself" snippet** (the one in `publish_oof.py`).
   HYPOTHESIS: a reusable artifact draws more votes than one more LB point.

### 1.4 Score ideas worth testing next, ranked

For each one: what it changes, and what has to be true for it to win.

1. **Bag the two kept members over 3 fold seeds** (4-leaf LGBM fe and raw, XGB depth 2).
   Changes: test predictions become the mean of 15 fold models instead of 5; each OOF row is the mean of 3 predictions.
   Wins if: the members' fold-to-fold variance is a real part of their error. Fold AUCs that spread by more than about 0.0005 would show that.
   Expected: +0.0001 to +0.0003 (HYPOTHESIS). Cost: 2x those members' time.
2. **Lower learning rate for the shallow members** (0.05 to 0.02, early-stopping patience x2.5).
   Wins if: 4-leaf trees at lr 0.05 stop on a noisy plateau. Check the best iteration: if it is far below the cap, lr is not the limit.
   Cost: about 2.5x those members' time.
3. **Refit on all 700k rows** at 1.15 x the mean best iteration, next to the 5-fold mean.
   Wins if: 20% more data helps a model this simple. Score it on the public LB only; it has no OOF.
   This is risky as a tutorial point, because it cannot be checked in CV. Only a single side-by-side submission can show it.
4. **LightGBM `interaction_constraints`**: allow splits that combine only columns among the top-10 list, and keep every other column additive.
   Wins if: depth-2 trees are spending splits on noise pairs. The num_leaves curve already points that way.
   Cheap: one member.

Not worth it as a tutorial point: blending public notebooks' submissions. That is the usual route to 0.951 on a playground board, but it teaches nothing.

## 2. Smart-buildings RL

Sources: HANDOFF.md (last section 2026-11-01 10:30), skynet_train.py, skynet/env.py, skynet/nets.py, skynet/features.py (head), tools/make_tutorial.py, docs/FABLE_REVIEW5 and REVIEW6, logs/breakdown/run.log, starter-kit/README.md scoring section.

### 2.1 Critique of the plan

- **W1 vs C2 changes two things at once.** W1 is d128 at lr 2e-4 and C2 is d64 at lr 3e-4 (HANDOFF 10:30).
  If W1 wins, the cause can be the width or the lower lr. B1 already suggests that smaller effective steps help.
  Either run W1 at lr 3e-4, or add C3 = d64 at lr 2e-4. Otherwise do not call the result "capacity".
- **The four arms (clip 0.1, epochs 2, log-std schedule, gamma 0.999) are mostly one knob.**
  Clip, epochs and lr all set how far one update moves the policy, and B1's decay already turns that knob.
  The expected effects are probably below 0.0001, under the checkpoint noise (sd about 0.0002 on 20 buildings).
  selection60 helps, but if the 60 scores are independent the sd only falls by about sqrt(3), to about 0.00012 (HYPOTHESIS).
  One seed per arm cannot separate 0.0001 effects.
  Keep at most one "smaller step" arm, and keep gamma 0.999 only if you can name a mechanism.
  Its case is weak: the RL acts only in PI hours, and a 04:00 warm-up decision is 48 steps from 08:00. 0.995^48 = 0.79, so that credit is already mostly kept.
- **Each 500k arm needs its own 500k control.** `--steps` sets where the lr reaches 0 (skynet_train.py: `lr * max(0, 1 - steps/a.steps)`), so a 500k run is a different schedule from B1's 1M.
  The host lesson applies too: run the arm and its control on the same host with the same pool manifest.
- **B1's "monotone" is partly the decay itself.** As lr goes to 0 the checkpoints stop moving, so they agree with each other.
  0.010939 vs v11's 0.010958 is 0.00002 (HANDOFF), inside noise.
  The real value of B1 is that it makes checkpoint choice safe (no late bounce). That is a good result; it is not yet a score gain.

### 2.2 Is there a bigger lever? The gap to 0.0047

My short answer: on the honest bill, probably not with this simulator, and I would stop measuring progress against 0.0047.
- Review 5 (docs/FABLE_REVIEW5_2026-10-30.md section 1.2): the AHU has no heating coil, recirculation is capped at 0.7, and 30% outside air is always heated. In Fairbanks at 248 K, half of every heating kWh goes to ventilation.
  Review 5's physics bound says the 0.0048 cluster would need Fairbanks near 0.012, below that ventilation floor, unless the tank or return-air bugs are used. Not re-derived by me.
- logs/breakdown/run.log (plain zt, b8): Fairbanks cost is 31% energy and 76% carbon, and regret is about 0. Comfort is already almost free, so the only lever is energy, and the floor above bounds energy.
- The organisers say the final verification runs on the corrected simulator (docs/discussions-2026-10-30.md).
  So the public top is not the target that counts. Bronze (0.01055) is within reach of the honest-bill line, but the competition has no medals.

Two cheap tests that act on the plan's own problem (overfitting to the shipped buildings). I would run these before the knob arms:

1. **Occupancy reseed per episode (never run).**
   What it changes: each training episode draws new arrival and departure times. Today the seed 17321 replays the same draws every pass (review 5 idea 4; `grep -rn 17321 skynet/ sbrl.py` finds no reseed code).
   What must be true to win: part of the late drift is memorised arrival times. The selection-set result (everything past m3r got worse on unseen buildings) points that way.
   Cost: 2 lines in `env.py start()`, then one 1M B1-recipe run against B1.
   One caveat, stated once: the generated pool already varies the buildings, so the effect may be small.
2. **Action-ensembling of checkpoints, judged on selection60.**
   What it changes: the policy outputs the mean residual of 3-5 checkpoints (for example B1 at 762k, B1 at 1M, v11, p2 at 752k).
   This is not the weight averaging that failed (0.01054): averaging actions does not need the checkpoints to share one basin.
   What must be true to win: the checkpoints' errors are partly independent. Their 0.0002 bounce says they are.
   Cost: evaluation only. The nets are 48k-170k parameters, the simulator is 95-97% of a step, so inference cost does not change.
   Expected: 0.0001-0.0002 (HYPOTHESIS). If it wins, it is also a clean tutorial lesson.

### 2.3 Tutorial (tools/make_tutorial.py): what would earn 4 more expert votes

The content is honest and unusual: a residual that starts as the rule, the billing bug the RL found by itself, a judge that agrees with the board.
The problem is the form. It reads as a lab journal:
- the top table has 10 version rows;
- the failure table has about 40 rows;
- the text uses internal names (m3r, m4, zt_off, b8, p2) that a reader has never seen;
- there is no figure of what the policy actually does.

In order of value (all HYPOTHESIS for votes; no training needed):

1. **Hero figure: one Fairbanks weekday, rule vs RL.** Four stacked panels: zone temperatures against the comfort band, gas per hour, the RL's corrections (damper, reheat, zone target), and outside temperature.
   This is about 288 steps x 2 policies. At 13-23 ms per step (docs/TIMING.md) that is about 10-15 s plus setup. Skip it in the scoring rerun, as section 0 does.
   This answers the first question an RL expert asks: what did it learn?
2. **"Five lessons that transfer", with numbers**, near the top:
   - zero-initialised residual (day-0 score = the rule's);
   - building 8 picked the wrong checkpoint, and a 20-building judge agreed with the board (m3r over m4, 0.000164 vs 0.00018);
   - the RL found the billing bug, and 39% of the gain went with it;
   - lr decay to 0 removes the late bounce (B1 table);
   - full authority (A1) was no better than the gated residual.
   Alex asked for the B1, pool-mismatch, A1 and W1/C2 items to go into the next push (HANDOFF). This section is the natural place.
3. **Show the PPO core inline**, about 40 lines from skynet_train.py: GAE with the rollout-cut bootstrap, per-worker advantage normalisation, the clipped loss with the `fr` mask for held or off-hours steps, and the latch `vmask`.
   Experts vote for code they can learn from. At the moment the trainer only runs inside a subprocess.
4. **Shorten.** Fold the version rows into one LB-per-version chart. Group the failure table into 5 categories (base rule, exploration/PPO, inputs, actions/switches, data/pool), each with a one-line verdict. Put the full table in the dataset.
   Replace the internal names with descriptions on first use, or add a one-line glossary.
5. When the paper (round 1 due 2026-11-13) is on OpenReview, link it at the top.

## 3. Squares 101 (kernels/thinking-outside-the-grid)

There is no separate builder in tools/. The cron (`tools/cron/squares_daily.py`) refreshes the notebook, then pushes and retitles it.

What is there: sections 0-9 are a good first-principles explanation (n = 5 derived, the separating-axis picture, the float64 trap, the toy optimiser).
What holds it back:

1. **Structure and stale content** (30 min):
   - two sections are numbered 11: "What does not close the gap" inside the section-10 cell, and "11. Bonus";
   - the top contents table lists only 0-10, but there is also a section 12;
   - section 4 carries a long dated "Correction 2026-10-31 ... we had this backwards" block. Rewrite section 4 as one clean rule with one demonstration, and move the history to a footnote;
   - section 10 says "the board has moved by about 0.47" and "board top (2026-10-24) 22.225", and its point 4 says "about 0.08 below the best public merge". These are a week old: the HANDOFF of 10-30 has the top at 22.0611, about 0.2 below our 22.26;
   - section 11 "Bonus" is a growing merge log in prose. Make it a generated table: source, puzzles taken, date.
   HYPOTHESIS: experts discount a notebook whose own numbers disagree with each other.
2. **One new section that only this notebook has: "where the remaining points are"** (2-3 h):
   - per-n chart of our score against the grid and against the best known (couzo.py has the register), with the cumulative gap by n range;
   - the grid floor (457 puzzles where nobody beats ceil(sqrt(n))) shaded;
   - one worked fault-line DP example for a single n = K² + d: two K x K grids, the seam shifted by f, the gadgets drawn on it.
   src/fault_dp.py exists, so the figure can come from real code.
   Right now the notebook explains the DP, the method that matters above n = 324, only by linking out.
3. **A gadget gallery**: the 8-10 gadgets with the most leverage in the DP, each drawn with its overhang f, marked rigid (exact 45°) or squeezable.
   This is cheap and visual, and it shows readers the "different construction needed" frontier from section 11.

A note on the retitles: the slug has changed about 4 times a day (5,100+ to 5,800+ downloads between 10-30 and 10-31, per git log).
If Kaggle does not redirect old slugs, every forum link to the notebook breaks (HYPOTHESIS; I did not check the redirect).
Retitling at, say, every +500 downloads would keep the effect with fewer slug changes.
