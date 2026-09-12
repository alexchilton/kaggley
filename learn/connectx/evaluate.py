"""Measure the ConnectX agent's win rate against the built-in agents.

Each matchup plays both sides an equal number of times. Games run in parallel
processes because the agent spends real wall-clock time searching.

    python kaggle/connectx/evaluate.py --rounds 40 --opponents random negamax
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

CONFIG = {"rows": 6, "columns": 7, "inarow": 4}


def _resolve(name):
    """`random` and `negamax` are built in; the rest come from baselines.py."""
    if name in ("random", "negamax"):
        return name
    import baselines

    return getattr(baselines, "agent_" + name)


def _play(args):
    """One game. Returns (reward_for_us, invalid_by_us, n_moves)."""
    opponent, we_go_first = args
    from kaggle_environments import make
    from agent import my_agent

    opponent = _resolve(opponent)
    env = make("connectx", configuration=CONFIG)
    pair = [my_agent, opponent] if we_go_first else [opponent, my_agent]
    env.run(pair)
    us = 0 if we_go_first else 1
    reward = env.state[us].reward
    invalid = env.state[us].status not in ("DONE", "INACTIVE", "ACTIVE")
    return reward, invalid, len(env.steps)


def match(opponent, rounds, workers):
    jobs = [(opponent, i % 2 == 0) for i in range(rounds)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_play, jobs))
    elapsed = time.time() - t0

    wins = sum(1 for r, _, _ in results if r == 1)
    losses = sum(1 for r, _, _ in results if r == -1)
    draws = sum(1 for r, _, _ in results if r == 0)
    invalid = sum(1 for _, i, _ in results if i)
    moves = sum(n for _, _, n in results)
    return {
        "opponent": opponent,
        "rounds": rounds,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "invalid": invalid,
        "win_pct": wins / rounds,
        "sec_per_game": elapsed / rounds,
        "sec_per_move": elapsed / max(moves, 1) * workers,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--opponents", nargs="+", default=["random", "negamax", "q2", "tutorial", "minimax3"])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    rows = []
    for opp in args.opponents:
        r = match(opp, args.rounds, args.workers)
        rows.append(r)
        print(
            f"vs {r['opponent']:<10} {r['wins']:>3}W {r['losses']:>3}L {r['draws']:>3}D  "
            f"win={r['win_pct']:.2f}  invalid={r['invalid']}  "
            f"{r['sec_per_game']:.1f}s/game  ~{r['sec_per_move']:.2f}s/move",
            flush=True,
        )
    return rows


if __name__ == "__main__":
    main()
