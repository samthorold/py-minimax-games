"""Bots: anything that picks one of the legal actions.

A bot sees only its player's view of the game (`view.View`), never the state
(docs/star-realms.md). The one exception is a `BenchmarkBot`, such as the
cheat: a benchmark, not a legitimate player, which the match runner gives
the true state.
"""

import random
from typing import Protocol, runtime_checkable

from minimax_games.games.star_realms.action import (
    Acquire,
    Action,
    AttackBase,
    AttackPlayer,
    EndTurn,
    PlayCard,
    Scrap,
    UseAlly,
    UseBase,
)
from minimax_games.games.star_realms.card import Card, CardType
from minimax_games.games.star_realms.catalogue import EXPLORER
from minimax_games.games.star_realms.effect import Combat, Effect, Trade
from minimax_games.games.star_realms.state import GameState
from minimax_games.games.star_realms.view import PlayerView, View


class Bot(Protocol):
    def choose(self, view: View, actions: tuple[Action, ...]) -> Action:
        """Pick one of `actions`, the legal actions in the game `view` shows."""
        ...


@runtime_checkable
class BenchmarkBot(Protocol):
    """A benchmark-only bot: it sees the true state, so it is a yardstick, not
    a legitimate player. Implementing `choose_from_state` is how a bot marks
    itself benchmark-only; every other bot is a `Bot` and gets views only."""

    def choose_from_state(
        self, state: GameState, actions: tuple[Action, ...]
    ) -> Action:
        """Pick one of `actions`, the legal actions in the true `state`."""
        ...


@runtime_checkable
class Budgeted(Protocol):
    """A search bot: one that states its search settings (horizon, budget and
    so on), which the tournament command prints."""

    def settings(self) -> str:
        """The bot's search settings, as a phrase."""
        ...


type AnyBot = Bot | BenchmarkBot
"""A bot the match runner accepts: a `Bot`, or a `BenchmarkBot`."""


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


SYNERGY = 0.5
"""What each card of its faction the heuristic bot owns adds to a buy."""

ECONOMY_TURNS = 2
"""For this many of its own first turns, the heuristic bot buys for trade;
after them, for combat."""

ECONOMY = 1.0
"""What each point of primary trade adds to a buy in the economy turns."""

AGGRESSION = 1.5
"""What each point of primary combat adds to a buy after the economy turns."""

BASE_DEFENSE = 0.5
"""What each point of defense adds to buying a base that isn't an Outpost."""


def _trade(effects: tuple[Effect, ...]) -> int:
    """The trade among `effects`."""
    return sum(e.amount for e in effects if isinstance(e, Trade))


def _combat(effects: tuple[Effect, ...]) -> int:
    """The combat among `effects`."""
    return sum(e.amount for e in effects if isinstance(e, Combat))


def _owned(view: View) -> tuple[Card, ...]:
    """Every card the observer owns, in whichever pile."""
    me = view.players[view.observer]
    return me.hand + me.unseen + me.in_play + me.discard + me.bases


def _guard(player: PlayerView) -> int:
    """The total defense of `player`'s Outposts."""
    return sum(c.defense or 0 for c in player.bases if c.type is CardType.OUTPOST)


def _worth(card: Card, view: View, owned: tuple[Card, ...]) -> float:
    """How much the heuristic bot wants to acquire `card` (see its docstring)."""
    worth = card.cost
    if card.faction is not None:
        worth += SYNERGY * sum(c.faction is card.faction for c in owned)
    if card.type is CardType.BASE:
        worth += BASE_DEFENSE * (card.defense or 0)
    own_turn = (view.turn + 1) // 2
    if own_turn <= ECONOMY_TURNS:
        worth += ECONOMY * _trade(card.primary)
    else:
        worth += AGGRESSION * _combat(card.primary)
    return worth


def _dearest(action: AttackBase) -> tuple[int, str]:
    """A sort key for attacks: the dearest base first, ties by name."""
    return -action.card.cost, action.card.name


class HeuristicBot:
    """The heuristic baseline: faction synergy, bases, scrapping and targets.

    Deterministic, and it uses only its view and the legal actions. The
    weights are module constants, tuned against greedy on seeds outside the
    test range. In priority order, it:

    1. plays every card it can and uses every ally and base ability, in
       `actions` order, as greedy does: all of them are gains.
    2. acquires the affordable card worth most, ties broken by name, and
       repeats while it can afford one. A card is worth its cost, plus
       `SYNERGY` for each card of its faction it already owns (for ally
       abilities), plus `BASE_DEFENSE` per point of defense for a base that
       isn't an Outpost (it keeps working every turn), plus, in its first
       `ECONOMY_TURNS` turns, `ECONOMY` per point of primary trade, and after
       them `AGGRESSION` per point of primary combat: economy early, combat
       late. Outposts get no bonus for their defense: tuning found it lost
       games, as an Outpost's defense only delays the opponent.
    3. scraps cards for combat (after buying, so their trade is spent):
       every card with a combat scrap ability when the combat in hand and in
       those abilities reaches the opponent's Authority plus the defense of
       their Outposts, so this turn wins; otherwise only Explorers, whose
       trade helps only the turn they are played, so 2 combat and a thinner
       deck are worth more than keeping them. Other cards are kept. (Scrap
       abilities that give trade, and scrapping Scouts and Vipers from hand or
       discard pile, need effects a later card slice adds.)
    4. attacks: while Outposts guard the opponent, destroys the weakest one
       first if the combat clears them all (the order doesn't matter then),
       otherwise the dearest one it can (ties by name), as it can't reach the
       player this turn anyway. With no Outposts, it hits the player if that
       wins; otherwise it destroys the dearest base it can first (ties by
       name), since a base works every turn and the damage to the player
       can wait, then hits the player with the rest.
    5. ends the turn.
    """

    def choose(self, view: View, actions: tuple[Action, ...]) -> Action:
        for action in actions:
            if isinstance(action, PlayCard | UseAlly | UseBase):
                return action
        if (buy := self._buy(view, actions)) is not None:
            return buy
        if (scrap := self._scrap(view, actions)) is not None:
            return scrap
        if (attack := self._attack(view, actions)) is not None:
            return attack
        return EndTurn()

    def _buy(self, view: View, actions: tuple[Action, ...]) -> Action | None:
        owned = _owned(view)
        buys = [a for a in actions if isinstance(a, Acquire)]
        if not buys:
            return None
        return min(buys, key=lambda a: (-_worth(a.card, view, owned), a.card.name))

    def _scrap(self, view: View, actions: tuple[Action, ...]) -> Action | None:
        scraps = [
            a for a in actions if isinstance(a, Scrap) and _combat(a.card.scrap) > 0
        ]
        if not scraps:
            return None
        me, opponent = view.players[view.observer], view.players[1 - view.observer]
        in_play = me.in_play + me.bases
        reach = view.combat + sum(
            _combat(a.card.scrap) * in_play.count(a.card) for a in scraps
        )
        if reach >= opponent.authority + _guard(opponent):
            return min(scraps, key=lambda a: a.card.name)
        if Scrap(EXPLORER) in scraps:
            return Scrap(EXPLORER)
        return None

    def _attack(self, view: View, actions: tuple[Action, ...]) -> Action | None:
        opponent = view.players[1 - view.observer]
        bases = [a for a in actions if isinstance(a, AttackBase)]
        outposts = [a for a in bases if a.card.type is CardType.OUTPOST]
        if outposts:
            if view.combat >= _guard(opponent):
                return min(outposts, key=lambda a: (a.card.defense or 0, a.card.name))
            return min(outposts, key=_dearest)
        if AttackPlayer() in actions and view.combat >= opponent.authority:
            return AttackPlayer()
        if bases:
            return min(bases, key=_dearest)
        if AttackPlayer() in actions:
            return AttackPlayer()
        return None
