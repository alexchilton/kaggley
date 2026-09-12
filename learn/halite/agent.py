"""Halite bot.

Unlike ConnectX, a Halite submission is a whole .py file rather than one
function pulled out by `inspect.getsource`, so this can be an ordinary module
with top-level helpers.

Four things it does that the tutorial bot does not:

1. Moves on the torus. The tutorial's `getDirTo` compares raw coordinates, so
   it walks 18 cells where wrapping takes 2. Measured: (1,1) -> (19,19)
   returns NORTH.
2. Does not let its own ships collide. Two ships ordered onto one cell destroy
   the loaded one. Every move is reserved before it is issued.
3. Keeps spawning while a new ship can still pay for itself, instead of only
   when the fleet is empty.
4. Empties its cargo before the game ends. Halite in a hold at step 400 scores
   nothing.
"""

from kaggle_environments.envs.halite.helpers import (
    Board,
    ShipAction,
    ShipyardAction,
)

# A ship costs 500 and has to earn that back before the game ends. Past this
# step it cannot, so stop buying.
LAST_SPAWN_STEP = 300
# Ships in hand past this are more collision risk than income on a 21x21 board.
MAX_SHIPS = 18
# Run everything home with enough steps left to actually arrive.
HOMING_STEP = 370
# Cargo worth a trip back rather than one more mining turn.
CARGO_FULL = 500
# Only look this far for a mining target; beyond it the travel cost dominates.
# When nothing is found inside it the whole board is scanned instead, because a
# ship with no target stands still, and a ship standing still is a wall.
SEARCH_RADIUS = 8

MOVES = {
    ShipAction.NORTH: (0, 1),
    ShipAction.SOUTH: (0, -1),
    ShipAction.EAST: (1, 0),
    ShipAction.WEST: (-1, 0),
}


def wrap(value, size):
    """Coordinates live on a torus, so every offset folds back into [0, size)."""
    return value % size


def axis_delta(a, b, size):
    """Signed steps from a to b along one axis, taking the short way round."""
    d = (b - a) % size
    return d if d <= size // 2 else d - size


def distance(a, b, size):
    return abs(axis_delta(a[0], b[0], size)) + abs(axis_delta(a[1], b[1], size))


def step_towards(a, b, size):
    """Directions that reduce the toroidal distance from a to b."""
    out = []
    dx = axis_delta(a[0], b[0], size)
    dy = axis_delta(a[1], b[1], size)
    if dy > 0:
        out.append(ShipAction.NORTH)
    elif dy < 0:
        out.append(ShipAction.SOUTH)
    if dx > 0:
        out.append(ShipAction.EAST)
    elif dx < 0:
        out.append(ShipAction.WEST)
    return out


def target_cell(position, action, size):
    if action is None:
        return (position[0], position[1])
    dx, dy = MOVES[action]
    return (wrap(position[0] + dx, size), wrap(position[1] + dy, size))


def enemy_threats(board, me, size):
    """Cells an enemy ship could reach next turn, and the cargo it carries.

    A collision destroys whichever ship holds more, so a loaded ship must
    avoid squares a lighter enemy can step onto.
    """
    threats = {}
    for player in board.players.values():
        if player.id == me.id:
            continue
        for ship in player.ships:
            x, y = ship.position
            for dx, dy in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:
                cell = (wrap(x + dx, size), wrap(y + dy, size))
                prev = threats.get(cell)
                if prev is None or ship.halite < prev:
                    threats[cell] = ship.halite
    return threats


def _best_within(ship, board, size, radius):
    best = None
    best_score = 0.0
    x, y = ship.position
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            steps = abs(dx) + abs(dy)
            if steps > radius:
                continue
            pos = (wrap(x + dx, size), wrap(y + dy, size))
            cell = board.cells[pos]
            if cell.halite <= 0:
                continue
            # Someone else's shipyard is a place to die, not to mine.
            if cell.shipyard is not None and cell.shipyard.player_id != ship.player_id:
                continue
            score = cell.halite / (steps + 1.0)
            if score > best_score:
                best_score = score
                best = pos
    return best


def mining_target(ship, board, size):
    """Best cell to mine, scored as halite per turn spent getting there.

    Falls back to the whole board when the neighbourhood is mined out. Without
    the fallback a ship with nothing nearby stands still for the rest of the
    game, and if it happens to be standing on our own shipyard it blocks every
    other ship from depositing -- measured as a permanent deadlock that banked
    364 halite in 400 steps.
    """
    found = _best_within(ship, board, size, SEARCH_RADIUS)
    if found is None:
        found = _best_within(ship, board, size, size // 2)
    return found


def agent(obs, config):
    board = Board(obs, config)
    # The raw `config` is a Struct with the JSON's camelCase keys. The Board
    # wraps it in a Configuration that exposes snake_case properties, so read
    # costs from there -- `config.spawn_cost` is an AttributeError.
    cfg = board.configuration
    size = cfg.size
    me = board.current_player
    step = board.step

    # No shipyard and no way to make one later: convert the first ship now.
    if not me.shipyards and me.ships:
        me.ships[0].next_action = ShipAction.CONVERT
        return me.next_actions

    threats = enemy_threats(board, me, size)
    yards = [sy.position for sy in me.shipyards]

    # Cells already spoken for this turn. Shipyards that will spawn are
    # occupied too, so nothing may move onto them.
    reserved = set()

    halite = me.halite
    spawning = []
    if step < LAST_SPAWN_STEP and len(me.ships) < MAX_SHIPS:
        for yard in me.shipyards:
            if halite < cfg.spawn_cost:
                break
            # Do not spawn onto one of our own ships.
            if yard.cell.ship is not None and yard.cell.ship.player_id == me.id:
                continue
            spawning.append(yard)
            reserved.add((yard.position[0], yard.position[1]))
            halite -= cfg.spawn_cost

    # Only the destination cells have to be unique -- ships may follow each
    # other into a square the occupant is vacating this same turn. Order is
    # what keeps the fallback safe and the traffic moving:
    #
    # 1. Ships on one of our shipyards go first so they can get out of the way.
    #    A ship that just deposited holds zero cargo, so heaviest-first would
    #    consider it last, every loaded neighbour would find the yard occupied,
    #    and the ring would never dissolve. Measured: 3,661 halite stuck in
    #    cargo at step 375, and the agent issuing no actions at all.
    # 2. Then the ships with the fewest safe squares, so a nearly-trapped ship
    #    claims somewhere before a ship that has options takes its last one.
    # 3. Then the heaviest, which have the most to lose in a collision.
    def mobility(s):
        pos = (s.position[0], s.position[1])
        free = 0
        for action in MOVES:
            cell = target_cell(pos, action, size)
            risk = threats.get(cell)
            if risk is None or risk > s.halite or cell in yards:
                free += 1
        return free

    def order(s):
        pos = (s.position[0], s.position[1])
        return (0 if pos in yards else 1, mobility(s), -s.halite)

    for ship in sorted(me.ships, key=order):
        pos = (ship.position[0], ship.position[1])
        going_home = (
            ship.halite >= CARGO_FULL
            or (step >= HOMING_STEP and ship.halite > 0)
        )

        if going_home and yards:
            nearest = min(yards, key=lambda y: distance(pos, y, size))
            wanted = step_towards(pos, nearest, size)
            if not wanted:
                wanted = [None]  # already home; depositing happens on arrival
        else:
            target = mining_target(ship, board, size)
            if target is None or target == pos:
                wanted = [None]
            else:
                wanted = step_towards(pos, target, size)
            # Staying put mines the current cell. Prefer it when this cell is
            # richer than the step we were about to take.
            here = ship.cell.halite
            if here > 0 and wanted and wanted[0] is not None:
                ahead = board.cells[target_cell(pos, wanted[0], size)].halite
                if here >= ahead:
                    wanted = [None] + wanted
            # Never idle on our own shipyard. Standing there with nothing to do
            # blocks every loaded ship from depositing.
            if pos in yards and (wanted == [None] or here <= 0):
                wanted = [d for d in wanted if d is not None] or list(MOVES)

        # Fall back through every legal option, then accept standing still.
        options = list(wanted) + [None] + list(MOVES)
        chosen = None
        cell = pos
        found = False
        for action in options:
            candidate = target_cell(pos, action, size)
            if candidate in reserved:
                continue
            risk = threats.get(candidate)
            if risk is not None and risk <= ship.halite and candidate not in yards:
                continue
            chosen, cell, found = action, candidate, True
            break
        if not found:
            # Every square is taken or dangerous. Stand still.
            chosen, cell = None, pos

        reserved.add(cell)
        ship.next_action = chosen

    for yard in spawning:
        yard.next_action = ShipyardAction.SPAWN

    return me.next_actions
