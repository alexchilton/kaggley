"""Tests for the Halite bot.

The pathing tests are the point: the tutorial bot fails all three because it
compares raw coordinates on a board that wraps.
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
_halite = _load("halite_agent", ROOT / "halite" / "agent.py")
axis_delta = _halite.axis_delta
distance = _halite.distance
step_towards = _halite.step_towards
wrap = _halite.wrap
MOVES = _halite.MOVES
target_cell = _halite.target_cell

SIZE = 21


def names(actions):
    return {a.name for a in actions}


def test_distance_takes_the_short_way_round():
    """(1,1) to (19,19) is 3 steps on each axis through the wrap, not 18."""
    assert distance((1, 1), (19, 19), SIZE) == 6


def test_direction_goes_through_the_wrap():
    assert names(step_towards((1, 1), (19, 19), SIZE)) == {"SOUTH", "WEST"}


def test_direction_is_plain_when_no_wrap_is_shorter():
    assert names(step_towards((10, 10), (10, 12), SIZE)) == {"NORTH"}


def test_no_direction_when_already_there():
    assert step_towards((5, 5), (5, 5), SIZE) == []


@pytest.mark.parametrize("a,b", [((0, 0), (10, 0)), ((0, 0), (11, 0))])
def test_axis_delta_never_exceeds_half_the_board(a, b):
    assert abs(axis_delta(a[0], b[0], SIZE)) <= SIZE // 2


def test_wrap_folds_both_ways():
    assert wrap(-1, SIZE) == 20
    assert wrap(21, SIZE) == 0


def test_distance_is_symmetric():
    import random

    rng = random.Random(0)
    for _ in range(50):
        a = (rng.randrange(SIZE), rng.randrange(SIZE))
        b = (rng.randrange(SIZE), rng.randrange(SIZE))
        assert distance(a, b, SIZE) == distance(b, a, SIZE)


def test_ships_are_never_ordered_onto_the_same_cell():
    """Two of our ships on one cell destroys the loaded one. Play a real game
    and assert the bot never issues a pair of moves that collide.

    `env.steps[i].action` is the action that PRODUCED state i, so it has to be
    read against the observation at i-1. Pairing them at the same index looks
    almost right and reports collisions that never happened.
    """
    from kaggle_environments import make
    from kaggle_environments.envs.halite.helpers import Board

    env = make("halite", configuration={"episodeSteps": 60})
    env.run([str(ROOT / "halite" / "agent.py"), "random", "random", "random"])

    checked = 0
    for before, after in zip(env.steps, env.steps[1:]):
        obs = before[0].observation
        board = Board(obs, env.configuration)
        me = board.players[0]
        if len(me.ships) < 2:
            continue
        actions = after[0].action or {}
        seen = set()
        for ship in me.ships:
            raw = actions.get(ship.id)
            if raw == "CONVERT":
                continue
            act = next((a for a in MOVES if a.name == raw), None)
            cell = target_cell((ship.position[0], ship.position[1]), act, board.configuration.size)
            assert cell not in seen, f"two ships ordered onto {cell} at step {board.step}"
            seen.add(cell)
        checked += 1
    assert checked > 0, "no multi-ship turns were checked"


def test_agent_returns_actions_and_does_not_crash():
    from kaggle_environments import make

    env = make("halite", configuration={"episodeSteps": 40}, debug=True)
    env.run([str(ROOT / "halite" / "agent.py"), "random", "random", "random"])
    assert [p.status for p in env.steps[-1]] == ["DONE"] * 4
