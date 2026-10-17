"""Bots: anything that picks one of the legal actions.

For now a bot sees the whole state; the observation views slice will hand
bots only what their player knows (docs/star-realms.md).
"""

import random
from typing import Protocol

from minimax_games.games.star_realms.action import Action
from minimax_games.games.star_realms.state import GameState


class Bot(Protocol):
    def choose(self, state: GameState, actions: tuple[Action, ...]) -> Action:
        """Pick one of `actions`, the legal actions in `state`."""
        ...


class RandomBot:
    """Picks uniformly at random among the legal actions, from its own seed."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def choose(self, state: GameState, actions: tuple[Action, ...]) -> Action:
        return self._rng.choice(actions)
