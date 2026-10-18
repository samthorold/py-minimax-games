"""The cheat: alpha-beta search on the true state.

A benchmark and an upper bound, not a legitimate player: it sees the whole
`GameState` (both hands, every deck's order, the shuffle seed), which no other
bot is given (docs/star-realms.md, Search).
"""

import logging

from minimax_games.games.star_realms.action import Action
from minimax_games.games.star_realms.node import StarRealmsNode, ends_turn
from minimax_games.games.star_realms.state import GameState
from minimax_games.search.alphabeta import alphabeta

logger = logging.getLogger(__name__)


class CheatBot:
    """Benchmark only, not a legitimate player: searches the true state.

    It runs `alphabeta` on the whole state, `turns` turns deep (by default
    2: the rest of its own turn plus the opponent's reply), scores the
    positions it reaches with `evaluate`, and plays the first move of the
    best line. The match runner gives it the true state because it is a
    `bots.BenchmarkBot`.
    """

    def __init__(self, turns: int = 2) -> None:
        if turns < 1:
            msg = f"the horizon must be at least 1 turn, not {turns}"
            raise ValueError(msg)
        self.turns = turns

    def choose_from_state(
        self, state: GameState, actions: tuple[Action, ...]
    ) -> Action:
        root = StarRealmsNode(state, state.current)
        best = alphabeta(
            root, root.minimum(), root.maximum(), depth=self.turns, ply_cost=ends_turn
        )
        action = best.moves[0]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("best line %s scores %s", best.moves, best.score())
        assert action in actions, (action, actions)
        return action
