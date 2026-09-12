"""The bot from the Halite bonus lesson, copied verbatim.

Kept exactly as the notebook writes it, bug included, because it is the
baseline every number in `evaluate.py` is measured against. See README.md for
what `getDirTo` actually does.
"""

from kaggle_environments.envs.halite.helpers import *


# Returns best direction to move from one position (fromPos) to another (toPos)
# Example: If I'm at pos 0 and want to get to pos 55, which direction should I choose?
def getDirTo(fromPos, toPos, size):
    fromX, fromY = divmod(fromPos[0], size), divmod(fromPos[1], size)
    toX, toY = divmod(toPos[0], size), divmod(toPos[1], size)
    if fromY < toY: return ShipAction.NORTH
    if fromY > toY: return ShipAction.SOUTH
    if fromX < toX: return ShipAction.EAST
    if fromX > toX: return ShipAction.WEST


directions = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST]

ship_states = {}


def agent(obs, config):
    size = config.size
    board = Board(obs, config)
    me = board.current_player

    if len(me.ships) == 0 and len(me.shipyards) > 0:
        me.shipyards[0].next_action = ShipyardAction.SPAWN

    if len(me.shipyards) == 0 and len(me.ships) > 0:
        me.ships[0].next_action = ShipAction.CONVERT

    for ship in me.ships:
        if ship.next_action is None:

            if ship.halite < 200:
                ship_states[ship.id] = "COLLECT"
            if ship.halite > 500:
                ship_states[ship.id] = "DEPOSIT"

            if ship_states[ship.id] == "COLLECT":
                if ship.cell.halite < 100:
                    neighbors = [ship.cell.north.halite, ship.cell.east.halite,
                                 ship.cell.south.halite, ship.cell.west.halite]
                    best = max(range(len(neighbors)), key=neighbors.__getitem__)
                    ship.next_action = directions[best]
            if ship_states[ship.id] == "DEPOSIT":
                direction = getDirTo(ship.position, me.shipyards[0].position, size)
                if direction: ship.next_action = direction

    return me.next_actions
