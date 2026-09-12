"""Play the bot against the tutorial bot and report final halite.

Halite is a four-player game, so a "win rate" needs a seat assignment: the bot
plays each of the four seats in turn, with the named opponent in the other
three. Final score is a player's halite at step 400.

    python kaggle/halite/evaluate.py --games 8 --opponent tutorial
"""

import argparse
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def _play(args):
    """One game. Returns (our_halite, [all four scores], our_seat, steps)."""
    opponent, seat = args
    from kaggle_environments import make

    mine = str(HERE / "agent.py")
    opp = "random" if opponent == "random" else str(HERE / "tutorial_bot.py")

    players = [opp, opp, opp, opp]
    players[seat] = mine

    env = make("halite", configuration={"episodeSteps": 400})
    env.run(players)

    final = env.steps[-1]
    scores = [p.reward if p.reward is not None else 0 for p in final]
    return scores[seat], scores, seat, len(env.steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--opponent", default="tutorial", choices=["tutorial", "random"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    jobs = [(args.opponent, i % 4) for i in range(args.games)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(_play, jobs))
    elapsed = time.time() - t0

    ours = [r[0] for r in results]
    wins = sum(1 for our, scores, _, _ in results if our == max(scores))
    print(f"bot vs 3x {args.opponent}, {args.games} games, seat rotated")
    for our, scores, seat, steps in results:
        mark = "W" if our == max(scores) else " "
        pretty = " ".join(f"{s:>7.0f}" for s in scores)
        print(f"  seat {seat}  [{pretty}]  ours={our:>7.0f} {mark}  steps={steps}")
    print(
        f"wins {wins}/{args.games}  "
        f"our halite median={statistics.median(ours):.0f} "
        f"min={min(ours):.0f} max={max(ours):.0f}  "
        f"{elapsed / args.games:.1f}s/game"
    )


if __name__ == "__main__":
    main()
