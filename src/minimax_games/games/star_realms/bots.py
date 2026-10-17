"""Bots: anything that picks one of the legal actions.

A bot sees only its player's view of the game (`view.View`), never the state
(docs/star-realms.md).
"""

import random
from typing import Protocol

from minimax_games.games.star_realms.action import Action
from minimax_games.games.star_realms.view import View


class Bot(Protocol):
    def choose(self, view: View, actions: tuple[Action, ...]) -> Action:
        """Pick one of `actions`, the legal actions in the game `view` shows."""
        ...


class RandomBot:
    """Picks uniformly at random among the legal actions, from its own seed."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def choose(self, view: View, actions: tuple[Action, ...]) -> Action:
        return self._rng.choice(actions)
