"""The course's own agents, used as local opponents.

`random` and `negamax` ship with kaggle_environments. These are the ones the
lessons build, so they are the yardstick that means something.

`agent_tutorial` is the lesson-2 agent copied weight for weight. Exercise 2
grades you by playing your weights against exactly this, so it has to match it
exactly -- a "stronger tutorial agent" would make every number here a lie.
"""

import random

import numpy as np


def _drop(grid, col, piece, config):
    nxt = grid.copy()
    for row in range(config.rows - 1, -1, -1):
        if nxt[row][col] == 0:
            break
    nxt[row][col] = piece
    return nxt


def _windows(grid, config):
    rows, cols, k = config.rows, config.columns, config.inarow
    for row in range(rows):
        for col in range(cols - k + 1):
            yield list(grid[row, col:col + k])
    for row in range(rows - k + 1):
        for col in range(cols):
            yield list(grid[row:row + k, col])
    for row in range(rows - k + 1):
        for col in range(cols - k + 1):
            yield list(grid[range(row, row + k), range(col, col + k)])
    for row in range(k - 1, rows):
        for col in range(cols - k + 1):
            yield list(grid[range(row, row - k, -1), range(col, col + k)])


def _count(grid, num_discs, piece, config):
    """Windows holding exactly `num_discs` of `piece`, rest empty."""
    n = 0
    for w in _windows(grid, config):
        if w.count(piece) == num_discs and w.count(0) == config.inarow - num_discs:
            n += 1
    return n


def _wins(obs, config, col, piece):
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    nxt = _drop(grid, col, piece, config)
    return any(w.count(piece) == config.inarow for w in _windows(nxt, config))


def agent_q2(obs, config):
    """Exercise 1: take the win, else block the win, else play at random."""
    valid = [c for c in range(config.columns) if obs.board[c] == 0]
    for col in valid:
        if _wins(obs, config, col, obs.mark):
            return col
    for col in valid:
        if _wins(obs, config, col, obs.mark % 2 + 1):
            return col
    return random.choice(valid)


def make_onestep(A, B, C, D, E):
    """Lesson-2 one-step lookahead with the exercise-2 five-term heuristic.

    score = A*fours + B*threes + C*twos + D*twos_opp + E*threes_opp
    """

    def agent(obs, config):
        mark = obs.mark
        opp = mark % 2 + 1
        grid = np.asarray(obs.board).reshape(config.rows, config.columns)
        valid = [c for c in range(config.columns) if obs.board[c] == 0]

        def score(col):
            nxt = _drop(grid, col, mark, config)
            return (
                A * _count(nxt, 4, mark, config)
                + B * _count(nxt, 3, mark, config)
                + C * _count(nxt, 2, mark, config)
                + D * _count(nxt, 2, opp, config)
                + E * _count(nxt, 3, opp, config)
            )

        scores = {c: score(c) for c in valid}
        best = max(scores.values())
        return random.choice([c for c, v in scores.items() if v == best])

    return agent


# The tutorial's weights, exactly: score = threes - 1e2*threes_opp + 1e6*fours.
agent_tutorial = make_onestep(A=1e6, B=1, C=0, D=0, E=-1e2)


def make_minimax(n_steps):
    """Lesson-3 minimax, depth `n_steps`, no pruning. The tutorial agent.

    Two deviations from the notebook, both forced and neither changing play:
    `np.Inf` was removed in NumPy 2.x, so this uses `math.inf`; and the board
    is scanned once per heuristic call instead of four times, because the
    unpruned tree is slow enough already.
    """

    def heuristic(grid, mark, config):
        opp = mark % 2 + 1
        k = config.inarow
        threes = fours = threes_opp = fours_opp = 0
        for w in _windows(grid, config):
            mine, theirs, empty = w.count(mark), w.count(opp), w.count(0)
            if mine == k:
                fours += 1
            elif mine == k - 1 and empty == 1:
                threes += 1
            if theirs == k:
                fours_opp += 1
            elif theirs == k - 1 and empty == 1:
                threes_opp += 1
        return threes - 1e2 * threes_opp - 1e4 * fours_opp + 1e6 * fours

    def terminal(grid, config):
        if list(grid[0, :]).count(0) == 0:
            return True
        k = config.inarow
        return any(w.count(1) == k or w.count(2) == k for w in _windows(grid, config))

    def minimax(node, depth, maximizing, mark, config):
        if depth == 0 or terminal(node, config):
            return heuristic(node, mark, config)
        valid = [c for c in range(config.columns) if node[0][c] == 0]
        if maximizing:
            return max(
                minimax(_drop(node, c, mark, config), depth - 1, False, mark, config)
                for c in valid
            )
        opp = mark % 2 + 1
        return min(
            minimax(_drop(node, c, opp, config), depth - 1, True, mark, config)
            for c in valid
        )

    def agent(obs, config):
        grid = np.asarray(obs.board).reshape(config.rows, config.columns)
        valid = [c for c in range(config.columns) if obs.board[c] == 0]
        scores = {
            c: minimax(_drop(grid, c, obs.mark, config), n_steps - 1, False, obs.mark, config)
            for c in valid
        }
        best = max(scores.values())
        return random.choice([c for c, v in scores.items() if v == best])

    return agent


agent_minimax3 = make_minimax(3)
