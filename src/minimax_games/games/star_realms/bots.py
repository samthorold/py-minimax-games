"""Bots: anything that picks one of the legal actions.

A bot sees only its player's view of the game (`view.View`), never the state
(docs/star-realms.md).
"""

import random
from typing import Protocol

from minimax_games.games.star_realms.action import (
    Acquire,
    Action,
    AttackBase,
    AttackPlayer,
    EndTurn,
    PlayCard,
    UseAlly,
    UseBase,
)
from minimax_games.games.star_realms.card import CardType
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


class GreedyBot:
    """The greedy baseline: play everything, buy the dearest card, attack.

    Deterministic: the same actions give the same choice, whatever their
    order. It needs nothing from the view beyond what the legal actions
    already say. In priority order, it:

    1. plays every card it can and uses every ally and base ability, in
       `actions` order. Bots are only offered the plays automatic play leaves
       (draws and bases), and all of them are gains; playing before buying
       makes the most trade. Drawing first may reshuffle the discard pile
       before this turn's buys reach it, which a baseline ignores.
    2. acquires the most expensive affordable card, ties broken by name
       (Explorers are cost-2 options), and repeats while it can afford one.
    3. attacks the opponent: the player whenever it is allowed, which spends
       all the combat; otherwise, while Outposts guard the player, destroys
       the one with the least defense (ties by name). Weakest first destroys
       the most Outposts with the combat it has, so it reaches the player
       soonest. Other bases are ignored.
    4. ends the turn. It never scraps: greedy keeps every card.
    """

    def choose(self, view: View, actions: tuple[Action, ...]) -> Action:
        for action in actions:
            if isinstance(action, PlayCard | UseAlly | UseBase):
                return action
        buys = [a for a in actions if isinstance(a, Acquire)]
        if buys:
            return min(buys, key=lambda a: (-a.card.cost, a.card.name))
        if AttackPlayer() in actions:
            return AttackPlayer()
        outposts = [
            a
            for a in actions
            if isinstance(a, AttackBase) and a.card.type is CardType.OUTPOST
        ]
        if outposts:
            return min(outposts, key=lambda a: (a.card.defense or 0, a.card.name))
        return EndTurn()
