# Kaggle Learn

Every exercise of ten Kaggle Learn courses, answered, plus the two agents from
the Intro to Game AI course.

| Directory | Course | Exercises |
| --- | --- | --- |
| `connectx/` | Intro to Game AI and Reinforcement Learning | 4 |
| `halite/` | the bonus lesson after it | competition closed |
| `ml-explainability/` | Machine Learning Explainability | 4 |
| `time-series/` | Time Series | 6 |
| `advanced-sql/` | Advanced SQL | 4 |
| `ai-ethics/` | Intro to AI Ethics | 4 |
| `geospatial/` | Geospatial Analysis | 5 |
| `intermediate-ml/` | Intermediate Machine Learning | 7 |
| `feature-engineering/` | Feature Engineering | 5 |
| `data-visualization/` | Data Visualization | 7 |
| `computer-vision/` | Computer Vision | 6 |

Each course directory holds `sources/` (the notebooks exactly as pulled from
Kaggle), `exN/` or `lessonN/` (the filled copies, each with the
`kernel-metadata.json` that pushes it), and a `build_notebooks.py` that writes
the second from the first.

`connectx/` and `halite/` also contain real agents: a bitboard negamax player
for ConnectX and a bot for Halite, with the harnesses that measure them.
Tests are in `tests/`.

## Where the answers come from

The learntools grader, not the question text. Most of these are
`EqualityCheckProblem` compared by value, several are checked by running a
query against BigQuery, and a few compare a Series the grader builds itself.
Reading the answer off `learntools/<course>/exN.py` is the only way to be sure,
and in Feature Engineering it is the only way to notice that the module numbers
are offset by one from the lesson order.

## `_tools/`

Four scripts, each written after a specific failure:

- **`audit.py`** — the important one. It does not look for stubs, because this
  course family marks a blank at least five ways (`____`, an empty `""" """`
  query, a complete query under "# Amend the query below", a bare assignment
  like `trend = food_sales`, and a function body of `pass`) and the last three
  run without error. Instead it diffs each built notebook against its source
  and reports every `check()` cell left byte-identical. Exits non-zero.
- **`add_outcome_report.py`** / **`collect_outcomes.py`** — a COMPLETE run only
  proves the notebook executed. learntools prints "Incorrect" rather than
  raising, and the rendered grade is display data the kernel log does not
  carry. These print each question's `_last_outcome` to stdout and read it back.
  Result across the five courses measured this way: 119 PASS, 0 FAIL.
- **`fix_input_paths.py`** — Kaggle mounts attached data either at
  `../input/<slug>/` or at `/kaggle/input/datasets/<owner>/<slug>/`, and
  `/kaggle/input` is read-only. learntools reads `../input/...` itself at import
  time, so the path has to be made real rather than rewritten: this symlinks
  the mounted data under `/kaggle/working` and chdirs next to it. It also
  replaces the courses' own `os.symlink` calls, which raise `FileExistsError`
  against a dangling link because `os.path.exists` returns False for one.
- **`fill_forks.py`** — pushing to `<slug>` claims it, so opening the same
  exercise from the course page forks an empty copy at `<slug>-<hash>`, and
  that is the one the course page links to. This copies the built notebook into
  those forks.

## Two things worth knowing before running any of this

A batch run does not credit a Kaggle Learn lesson. `KAGGLE_KERNEL_RUN_TYPE` is
`Batch`, `USE_KAGGLESDK` is False, and progress is reported by
`parent.postMessage` to a browser frame that is not there. The lesson ticks only
when the notebook is opened and run in the browser.

A fork sometimes ignores a data source it was forked with. Re-listing the same
slug is a no-op and leaves nothing mounted; attaching a second, byte-identical
mirror under a different owner makes Kaggle attach it. This was needed for Data
Visualization and Intermediate ML and not for Computer Vision, so it is a quirk
rather than a rule.
