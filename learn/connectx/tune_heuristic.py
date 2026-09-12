"""Exercise 2 asks you to pick five heuristic weights. This measures them.

The grader plays your weights against the tutorial agent over 50 rounds and
wants at least half, with C and D both nonzero. That is a measurement, so run
it rather than guessing:

    python kaggle/connectx/tune_heuristic.py --rounds 50
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

CONFIG = {"rows": 6, "columns": 7, "inarow": 4}

# (name, A, B, C, D, E). A = my fours, B = my threes, C = my twos,
# D = opponent twos, E = opponent threes. C and D must be nonzero to pass.
CANDIDATES = [
    ("mild",       1e6,  1e1, 1,   -1,  -1e2),
    ("tutorial+2s", 1e6, 1e2, 1,   -1,  -1e4),
    ("balanced",   1e6,  1e3, 1e1, -1e1, -1e4),
    ("defensive",  1e6,  1e2, 1,   -2,  -1e3),
    ("steep",      1e10, 1e4, 1e2, -1,  -1e6),
]


def _play(args):
    weights, we_go_first = args
    from kaggle_environments import make

    from baselines import agent_tutorial, make_onestep

    mine = make_onestep(*weights)
    env = make("connectx", configuration=CONFIG)
    pair = [mine, agent_tutorial] if we_go_first else [agent_tutorial, mine]
    env.run(pair)
    return env.state[0 if we_go_first else 1].reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    print(f"{args.rounds} rounds vs the lesson-2 tutorial agent, sides swapped each game")
    print(f"{'name':<12} {'A':>7} {'B':>7} {'C':>6} {'D':>6} {'E':>8}   W   L   D   win")
    for name, *w in CANDIDATES:
        jobs = [(tuple(w), i % 2 == 0) for i in range(args.rounds)]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            out = list(ex.map(_play, jobs))
        wins = out.count(1)
        losses = out.count(-1)
        draws = len(out) - wins - losses
        flag = "PASS" if wins / len(out) >= 0.5 else "fail"
        print(
            f"{name:<12} {w[0]:>7.0e} {w[1]:>7.0e} {w[2]:>6.0f} {w[3]:>6.0f} {w[4]:>8.0e} "
            f"{wins:>3} {losses:>3} {draws:>3}  {wins/len(out):.2f}  {flag}",
            flush=True,
        )


if __name__ == "__main__":
    main()
