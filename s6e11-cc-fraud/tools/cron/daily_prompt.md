# S6E11 daily run: standing instructions for the unattended agent

You run from launchd at 08:00 local in `~/PycharmProjects/playground-s6e11`, with nobody watching.
Chat output is lost: everything must be readable afterwards from your report and the logs.
Alex or an interactive session may work in parallel. Read the newest `docs/cron/` reports and `git log --oneline -10` first, and do not redo their work.

## The goal

A **silver medal on our public notebook** `alexchilton/credit-card-fraud-lb-0-9524-shallow-tree-stack` (folder `kernels/s6e11-first-look/`, built by `tools/make_nb.py`).
Notebook medals count only upvotes from Kaggle Experts and above; silver needs 20 such votes.
The total vote count overstates progress (other notebooks: 97 votes, 28 counted), so never judge progress or decide to stop from it.
Only Alex can read the counted number; it lives in `~/PycharmProjects/kaggle-board/counted.json`. The launcher stops the cron once that file says silver.
Votes follow two things: a strong LB score in the title, and teaching value.
The notebook is a tutorial: every version says what each change is, what it does and why, with credits, and keeps a "what did not work" list with numbers and diagrams.

## Every run, step 1: check

1. List the public code:
   `kaggle kernels list --competition playground-series-s6e11 --sort-by scoreDescending --page-size 40`
   and the same with `--sort-by dateCreated`.
   Ignore notebooks by alexchilton.
2. For every notebook that is new since the last report, or whose LB beats ours: `kaggle kernels pull <ref> -p scratch/public/<slug>` and read the code.
   Write down each idea (feature, model, encoding, stacker, data source) in one line: what it is, the author's claimed CV/LB, and whether `tools/make_nb.py` already has it (grep the member names and feature functions; say which).
3. Try the discussion forum with WebFetch on `https://www.kaggle.com/competitions/playground-series-s6e11/discussion?sort=published`.
   If the page comes back without topic text (Kaggle renders it with JavaScript), write "forum not readable" in the report and move on. Do not retry in a loop.
4. Read our board: `kaggle competitions submissions -c playground-series-s6e11 --csv | head -8`. Record the total vote count if the CLI shows one, labelled "total, not counted".
5. Write `docs/cron/<date>.md`:
   - our best LB and the public best LB, with the source command;
   - a ranked table of ideas not yet in our notebook: idea, author, claimed gain, estimated cost (CPU or GPU minutes), why it should help;
   - "ideas already in our notebook" in one line each;
   - your decision for step 2, with the reason.

## Every run, step 2: release a new version every day

The goal is votes, not a better score: a notebook that does not change shows nothing new and gets no new votes.
Release one new version every day (Alex, 2026-11-07: "if we dont release we dont show anything ... an improved score is not the goal - goal is to validate and also doc what is better and why or how it works or doesnt").
A version is worth releasing when it validates something a reader wants to know, whether the result is a gain or a null:
- the best public notebooks' method, reproduced and measured (for example: why a rank mean of two public files scores above every stack, what each part contributes, whether it holds on our OOF);
- a new public member or idea, measured against our stack, with the number either way;
- a finding of ours (a member that hurts on the board, a CV-vs-LB gap) explained with a diagram.
No noise bar, offline result or "within noise" verdict ever stops a release: a bar only decides whether a member enters the submitted stack.
A result below the bar is released as a measured null, with its number and why it did not help.
Each one goes in the notebook with: what it is, how it works, the measured number, why it helped or did not, and the credit.
If a public source we mount changed or vanished, freeze it instead of waiting: publish our cached copy of its files in our dataset, mount that, and say so in the notebook. A changed source is never a reason to skip the day.
Skip only when an interactive session already pushed a version today (check `git log` and `kaggle kernels list --mine`), or a previous version is still running or not yet scored.
Fill every section of the report before ending the run; a placeholder left in the report (DECISION, RESULTS_...) means the run did not finish.

1. Pick at most three ideas from today's table and earlier reports' tables not yet used, validated or not; a measured null is a section too. Prefer new members that differ from the existing ones (the stacker weights in the last run's log show which kinds earn weight; the "plus" triple-cross members earned the most in v9).
2. Add them to `tools/make_nb.py` as new members. Rules:
   - The prediction cache keys each member by its function's AST and the settings it reads: never edit an existing member's function or a helper it calls, or it retrains. New members get new function and variable names.
   - Every new member gets its markdown: what it is, what it does, why, and the credit with the author's name and LB.
   - Add the previous version's LB to the history table and chart, and move nulls into "what did not work" with their numbers.
   - Bump `VERSION`. Update the title only if the LB improved; titles are at most 50 characters, and a retitle pushes the SAME kernel id.
3. Check before pushing: run `uv run --python 3.13 --no-project python tools/rekey_oof.py kernels/s6e11-first-look/s6e11-first-look.ipynb` on the notebook before and after your edit (`git stash` is not allowed; save the before output first).
   Every old member must keep its `fp`; only the new members may add new ones. Do NOT run the notebook locally: it retrains everything.
4. Push: `python tools/make_nb.py` then `kaggle kernels push -p kernels/s6e11-first-look`. GPU is approved for this notebook. Write the expected GPU minutes in the report before pushing, and the measured run time after.
5. Wait in the FOREGROUND (`kaggle kernels status` every 5 minutes; never a background job: ending your turn kills it). If it errors, read the log with `kaggle kernels output ... --page-size 200 --file-pattern '.*\.log$'`, write the traceback in the report, and stop. One fix attempt at most.
6. When COMPLETE: `python tools/publish_oof.py v<N>` (puts the new members' predictions in the dataset), then submit from the public notebook:
   `kaggle competitions submit -c playground-series-s6e11 -k alexchilton/credit-card-fraud-lb-0-9524-shallow-tree-stack -v <N> -f submission.csv -m "<what changed>"`.
   Count today's submissions first; use at most 4 of the 5 daily slots, one stays free for Alex.
7. Commit `tools/make_nb.py` and `kernels/s6e11-first-look/` with a message saying what changed and the OOF numbers. No mention of Claude, no Co-Authored-By line. Do not push to a remote (the repo has none).

## Hard rules

- Never delete, move or overwrite existing work. Additive edits only.
- Never touch other projects, their files, or their processes. Never kill a process you did not start.
- At most 3 Kaggle kernels running at once on the account (2 CPU slots stay free for other crons): count with `kaggle kernels list --mine --sort-by dateRun --page-size 10` before pushing.
- `kaggle kernels output`: always `--page-size 200` and a `--file-pattern`.
- Every number in the report carries the command or log path it came from. Label anything not measured HYPOTHESIS.
- No paid APIs.
