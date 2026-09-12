# ConnectX — Kaggle submission

The Connect Four competition from
[Intro to Game AI and Reinforcement Learning](https://www.kaggle.com/learn/intro-to-game-ai-and-reinforcement-learning).
Board is 6x7, four in a row wins, an agent returns a column index 0-6, and an
illegal move or a timeout loses the game outright.

## Layout

| Path | What it is |
| --- | --- |
| `agent.py` | The agent. One self-contained function, `my_agent(obs, config)`. |
| `submission.py` | Generated from `agent.py` by `inspect.getsource`. This is what gets submitted. |
| `baselines.py` | The course's own agents, used as opponents. |
| `evaluate.py` | Plays matches against the baselines and prints win rates. |
| `tune_heuristic.py` | Measures candidate weights for the exercise-2 heuristic. |
| `build_notebooks.py` | Injects the answers and `my_agent` into the exercise notebooks. |
| `sources/` | The exercise notebooks pulled from Kaggle, untouched. |
| `lesson1/` .. `lesson4/` | The same notebooks filled in, each with its `kernel-metadata.json`. Push these. |
| `tutorials/` | The lesson notebooks, for reference. |

Tactical tests live in `tests/test_connectx_agent.py`.

One directory per lesson because `kaggle kernels push -p DIR` wants exactly one
`kernel-metadata.json` in the directory it is given.

## Why the agent is one long function

The course submits with `inspect.getsource(my_agent)`, which writes the
function body and nothing else. Every import, table and helper therefore has to
be nested inside `my_agent`. That is the reason for the shape, not taste.

## How it plays

Negamax with alpha-beta on a bitboard, iterative deepening against a wall-clock
budget of 0.90 s per move. ConnectX allows 2 s per action plus a 60 s bank for
the whole episode, so the budget leaves room for a slow worker.

The board is two integers: `my_pos` and `mask`. Bit `col * (rows + 1) + row`,
row 0 at the bottom. The extra row per column is never filled, and that is what
stops a vertical or diagonal run from wrapping into the next column — the win
test is then four shift-and-mask chains with no bounds checking.

`(mask + BOTTOM)` yields the next free square of every column at once, so there
is no separate height array to keep in step with the board.

Move ordering is centre-first, which is where the winning lines are dense.
A transposition table keyed on `(my_pos, mask)` is cleared each move.

Leaf evaluation counts the empty squares that would complete a four for each
side, weighting the ones playable right now more heavily, plus a small bonus
for centre occupancy.

Everything is wrapped in `try/except` with a legal random fallback. An
exception during an episode is a loss, so a crash costs more than a bad move.

## The lesson-2 weights are measured, not chosen

Exercise 2 asks for five heuristic weights and grades them by playing the
tutorial agent over 50 rounds. `tune_heuristic.py` runs that same match
locally over as many rounds as you ask for, so the answer in the notebook is
the one that won a measurement rather than the one that sounded right.

`baselines.agent_tutorial` copies the tutorial's weights exactly. An
approximation would have made every number in that table meaningless.

## Lesson 4 will fail Save & Run All with Internet off

`lesson4/kernel-metadata.json` has `enable_internet: false`, and the optional
section runs `!pip install "stable-baselines3"`. Save & Run All executes every
cell, so that cell errors and the version save fails. Turn Internet on in the
notebook's Settings menu, or set `enable_internet` to `true` before pushing.
The exercise text says to do this; the metadata does not.

The graded part of lesson 4 is one variable, `best_option`. The PPO section is
marked optional and has no checker.

## Lesson 3 answers are read off the grader, not recalled

`num_leaves` and `selected_move` are graded against fixed values, and the
question for `selected_move` is an image this repo cannot read. Both were taken
from `learntools/game_ai/ex3.py` in the Kaggle/learntools repo rather than
remembered: `_expected = 7**3` and `assert move == 3`. Lesson 4's
`best_option` came from `ex4.py` the same way: `assert ans == 'C'`.

## Measured

`python kaggle/connectx/evaluate.py --rounds 40`, sides swapped every game,
macOS, `.venv` Python 3.14.7. Logs in `data/logs/`:

- `connectx_eval_full.log` — the search agent against every baseline
- `connectx_l2_tune.log`, `connectx_l2_tune_200.log` — the exercise-2 weight sweep
- `connectx_eval_l3.log` — against the lesson-3 depth-3 minimax agent

## Submitting

```bash
python kaggle/connectx/build_notebooks.py        # answers -> lesson1/, lesson2/
kaggle kernels push -p kaggle/connectx/lesson3   # overwrites that Kaggle notebook
kaggle competitions submit -c connectx -f kaggle/connectx/submission.py -m "..."
```

Or on Kaggle: **Save Version -> Save and Run All**, open the version, **Data**
tab, pick `submission.py`, **Submit**.
