"""Tactical tests for the ConnectX submission agent.

A win rate against `random` proves almost nothing -- the win/block shortcut
alone gets you there. These positions have one correct answer each, and three
of them can only be found by searching past the immediate reply.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import importlib.util


def _load(name, path):
    """Load a module by path.

    Both kaggle bots are called `agent.py`. Importing either by bare name off
    `sys.path` means whichever test file runs first wins and the other gets the
    wrong module, so neither test may rely on `sys.path`.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]   # <repo>/learn
my_agent = _load("connectx_agent", ROOT / "connectx" / "agent.py").my_agent


class Config:
    rows = 6
    columns = 7
    inarow = 4


class Obs:
    def __init__(self, board, mark):
        self.board = board
        self.mark = mark


def build(rows, mark=1):
    """Rows are given top first, '.' empty, '1' us, '2' them -- as obs.board is laid out."""
    assert len(rows) == Config.rows
    board = []
    for row in rows:
        assert len(row) == Config.columns
        board += [0 if ch == "." else int(ch) for ch in row]
    return Obs(board, mark)


EMPTY = ["......."] * 6


def test_opening_takes_the_centre():
    """Connect Four is solved: the first player wins only by opening in the centre."""
    assert my_agent(build(EMPTY), Config) == 3


def test_takes_the_win_on_offer():
    obs = build(EMPTY[:5] + ["111...."])
    assert my_agent(obs, Config) == 3


def test_blocks_the_immediate_threat():
    obs = build(EMPTY[:5] + ["222...."])
    assert my_agent(obs, Config) == 3


def test_winning_beats_blocking():
    """Both sides are one move from winning on the same square. Ours counts."""
    obs = build(EMPTY[:5] + ["111.222"])
    assert my_agent(obs, Config) == 3


def test_stops_the_open_three():
    """`. . 2 2 . . .` on the bottom row.

    Leave it and the opponent plays either open end for `. 2 2 2 .`, which has
    two winning squares and cannot be blocked. This needs three plies of search:
    the loss is not on the board yet.
    """
    obs = build(EMPTY[:5] + ["..22..."])
    assert my_agent(obs, Config) in (1, 4)


def test_never_plays_a_full_column():
    """One legal column left. Returning anything else forfeits the game."""
    rows = [
        "12121.1",
        "21212.2",
        "12121.1",
        "21212.2",
        "12121.1",
        "21212.2",
    ]
    assert my_agent(build(rows), Config) == 5


def test_returns_a_valid_column_from_many_random_positions():
    import random

    rng = random.Random(0)
    for _ in range(30):
        board = [0] * 42
        for col in range(7):
            height = rng.randint(0, 6)
            for i in range(height):
                board[(5 - i) * 7 + col] = rng.choice([1, 2])
        valid = [c for c in range(7) if board[c] == 0]
        if not valid:
            continue
        move = my_agent(Obs(board, rng.choice([1, 2])), Config)
        assert move in valid


@pytest.mark.parametrize("mark", [1, 2])
def test_works_as_either_player(mark):
    """obs.mark flips the meaning of every cell. Getting this backwards is silent."""
    them = 2 if mark == 1 else 1
    obs = build(EMPTY[:5] + [f"{them}{them}{them}...."], mark=mark)
    assert my_agent(obs, Config) == 3
