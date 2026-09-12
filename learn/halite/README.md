# Halite — the bonus lesson

[Getting Started with Halite](https://www.kaggle.com/code/alexisbcook/getting-started-with-halite),
the bonus lesson after the Intro to Game AI course. A different competition
from ConnectX: 21x21 torus, four players, 400 steps, ships mine halite and
deposit it at shipyards.

## The competition is closed

The notebook says so in its first cell, and it checks out:

```
$ kaggle competitions list -s halite
https://www.kaggle.com/competitions/halite   deadline 2020-09-22 23:59:00
$ kaggle competitions submissions -c halite
403 Client Error: Forbidden
```

So there is no leaderboard to climb and no submission to make. The environment
still runs locally, which is the whole value of this directory: the bot can be
built and measured, just not submitted.

## Files

| Path | What it is |
| --- | --- |
| `agent.py` | The bot. |
| `tutorial_bot.py` | The lesson's bot, verbatim, as the opponent to beat. |
| `evaluate.py` | Plays four-player games with the seat rotated and reports final halite. |
| `tutorials/` | The lesson notebook, for reference. |

Tests are in `tests/test_halite_agent.py`.

Unlike ConnectX, a Halite submission is a whole `.py` file rather than one
function recovered by `inspect.getsource`, so `agent.py` is an ordinary module
with top-level helpers.

## What the bot does that the tutorial's does not

**It moves on the torus.** The tutorial's `getDirTo` compares raw coordinates:

```python
fromX, fromY = divmod(fromPos[0], size), divmod(fromPos[1], size)
if fromY < toY: return ShipAction.NORTH
```

Positions are always below `size`, so `divmod` returns `(0, coordinate)` and
the tuple comparison reduces to comparing the coordinates. It gets the
direction right on a flat board and ignores that the board wraps. Measured:
`getDirTo((1,1), (19,19))` returns NORTH, a 36-step walk. Going SOUTH and WEST
through the wrap is 6.

**It does not let its own ships collide.** Two ships ordered onto one cell
destroy the loaded one, so every destination is reserved before it is issued.

Only the destinations have to be unique. An earlier version also blocked
moving into a square still holding one of our own ships, reasoning that the
occupant might not vacate. That is wrong: ships may follow each other, and
forbidding it gridlocked the fleet. Eighteen ships packed around one shipyard
each found every neighbour occupied, all fell back to standing still, and the
agent issued no actions at all for the last 150 steps -- 5,812 halite frozen in
cargo, 141 banked, 1 win in 24 games.

What keeps the fallback safe is the order ships choose in, not a blocking set:

1. Ships standing on one of our shipyards go first, so they can get out of the
   way. A ship that has just deposited holds zero cargo, so under plain
   heaviest-first it is considered last -- every loaded neighbour finds the
   yard occupied and gives up, and the ring never dissolves.
2. Then ships with the fewest safe squares, so a nearly-trapped ship claims
   somewhere before a ship with options takes its last one.
3. Then the heaviest, which have the most to lose in a collision.

**It always has somewhere to go.** A ship with no halite within its search
radius used to stand still for the rest of the game, and a ship standing still
is a wall. If it happened to stop on our own shipyard it blocked every deposit.
The neighbourhood search now falls back to scanning the whole board.

**It keeps spawning.** The tutorial spawns only when the fleet is empty, so it
plays 400 steps with one ship. This one buys ships while one can still earn
back its 500 before the end.

**It empties its cargo.** Halite in a hold at step 400 scores nothing, so
every ship heads home near the end.

## A note on `config`

The raw `config` handed to the agent is a `Struct` with the JSON's camelCase
keys. `config.spawn_cost` raises `AttributeError`; the key is `spawnCost`. The
snake_case properties belong to `board.configuration`, which is what the bot
reads. The tutorial only ever uses `config.size`, where both spellings agree,
so it never hits this.

## Measured

`python kaggle/halite/evaluate.py --games 24`, the bot in each seat in turn
against three copies of the opponent, score = halite at step 400. Logs in
`data/logs/halite_eval_final.log` and `halite_eval_random.log`.
