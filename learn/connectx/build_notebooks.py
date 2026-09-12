"""Fill the Kaggle exercise notebooks with the answers, ready to push.

Reads the untouched notebooks in `sources/`, replaces the stub cells, and
writes a pushable copy into `lesson1/` and `lesson2/` next to the
`kernel-metadata.json` each one needs.

`my_agent` is injected from `agent.py`, so the notebooks and the local agent
cannot drift apart. The lesson-2 heuristic weights come from `WEIGHTS` below,
which is whatever `tune_heuristic.py` measured as best.

    python kaggle/connectx/build_notebooks.py
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).parent

# Chosen by `tune_heuristic.py`. See data/logs/connectx_l2_tune_200.log.
# Three candidates tied at the top over 200 rounds (0.70-0.71, sigma ~= 0.032),
# so this is "tied best and simplest", not a measured winner over the others.
WEIGHTS = dict(A=1e6, B=1e2, C=1, D=-1, E=-1e4)
MEASURED = "143 of 200 rounds (0.71)"

Q1_LESSON1 = '''import random

def agent_q1(obs, config):
    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]
    # Take a win whenever one is on the board.
    for col in valid_moves:
        if check_winning_move(obs, config, col, obs.mark):
            return col
    return random.choice(valid_moves)

# Check your answer
q_1.check()
'''

Q2_LESSON1 = '''def agent_q2(obs, config):
    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]
    # Win if we can.
    for col in valid_moves:
        if check_winning_move(obs, config, col, obs.mark):
            return col
    # Otherwise stop the opponent from winning next turn.
    for col in valid_moves:
        if check_winning_move(obs, config, col, obs.mark % 2 + 1):
            return col
    return random.choice(valid_moves)

# Check your answer
q_2.check()
'''

Q1_LESSON2 = '''# Measured, not guessed: these weights beat the tutorial agent in
# {wins} locally, sides swapped. See kaggle/connectx/tune_heuristic.py.
#
# A  my four in a row      -- win now, so it has to dominate everything else
# B  my three plus a gap   -- a threat the opponent must answer
# C  my two plus two gaps  -- weak, but it breaks ties towards useful shapes
# D  their two plus gaps   -- small penalty, enough to prefer crowding them
# E  their three plus gap  -- must outweigh B, or the agent builds while it loses
A = {A:g}
B = {B:g}
C = {C:g}
D = {D:g}
E = {E:g}

# Check your answer (this will take a few seconds to run!)
q_1.check()
'''

Q2_LESSON3 = '''# Depth 3, every column legal at every ply on an empty board:
# 7 agent moves, 7 opponent replies to each, 7 agent replies to each of those.
num_leaves = 7*7*7

# Check your answer
q_2.check()
'''

Q3_LESSON3 = '''# Score each move by the WORST leaf under it, not the best -- the opponent
# picks the reply, and it picks the one that hurts most. Then take the move
# whose worst case is highest. Move 3 has the best worst case.
selected_move = 3

# Check your answer
q_3.check()
'''

Q1_LESSON4 = '''# The network has to score every move it could make, so the output layer is
# one node per possible move -- 7 for Connect Four, 4672 for chess.
# Not 64: a chess move is a (from, to) pair plus promotions, not a square.
best_option = 'C'

# Check your answer
q_1.check()
'''

LESSONS = {
    "lesson1": {
        "source": "exercise-play-the-game.ipynb",
        "cells": {"def agent_q1": Q1_LESSON1, "def agent_q2": Q2_LESSON1, "def my_agent": None},
    },
    "lesson2": {
        "source": "exercise-one-step-lookahead.ipynb",
        "cells": {"A = ____": Q1_LESSON2, "def my_agent": None},
    },
    "lesson3": {
        "source": "exercise-n-step-lookahead.ipynb",
        "cells": {
            "num_leaves = ____": Q2_LESSON3,
            "selected_move = ____": Q3_LESSON3,
            "def my_agent": None,
        },
    },
    "lesson4": {
        "source": "exercise-deep-reinforcement-learning.ipynb",
        "cells": {"best_option = ____": Q1_LESSON4},
    },
}


def agent_source():
    text = (HERE / "agent.py").read_text()
    match = re.search(r"^def my_agent\(obs, config\):.*", text, re.S | re.M)
    if not match:
        raise SystemExit("could not find `def my_agent` in agent.py")
    return match.group(0).rstrip() + "\n"


def build(name, spec, body, wins):
    src = HERE / "sources" / spec["source"]
    out = HERE / name / spec["source"]
    nb = json.loads(src.read_text())

    done = set()
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        for needle, new in spec["cells"].items():
            if needle in text and needle not in done:
                new = body if new is None else new.format(wins=wins, **WEIGHTS)
                cell["source"] = new.splitlines(keepends=True)
                cell["outputs"] = []
                cell["execution_count"] = None
                done.add(needle)
                break

    missing = set(spec["cells"]) - done
    if missing:
        raise SystemExit(f"{name}: stub cells not found: {sorted(missing)}")

    out.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {out.relative_to(HERE.parents[1])}")


def main():
    body = agent_source()
    for name, spec in LESSONS.items():
        build(name, spec, body, MEASURED)
    print(f"my_agent: {len(body.splitlines())} lines")


if __name__ == "__main__":
    main()
