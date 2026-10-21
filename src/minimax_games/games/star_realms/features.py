"""The feature encoder a learned value reads (docs/star-realms.md, Learned
values): one row of numbers per turn-boundary state.

The row is from the side of the player who has just ended their turn
(`1 - state.current`, "ended") and then the player about to move ("next"),
so a model of it estimates how likely the player who just ended is to win.
Per player: how many of each card they own in their **draw pool** (deck,
hand, discard pile and cards in play: at a turn boundary the hand was just
drawn from the deck, so which cards were drawn is part of the deck's order,
which the row never sees), how many of each base they have in play, their
Authority, their bases' defense and the discards they owe. Then the turn,
and how many of each card the Trade Row shows.

Nothing in the row depends on the order of a pile: the planner scores
sampled worlds, and a value that could read deck order would score the
sample instead of the position.
"""

from collections.abc import Iterable

import numpy as np

from minimax_games.games.star_realms.card import CardType
from minimax_games.games.star_realms.catalogue import CATALOGUE
from minimax_games.games.star_realms.rules import original
from minimax_games.games.star_realms.state import GameState, PlayerState

CARDS: tuple[str, ...] = tuple(CATALOGUE)
"""Every card's name, in catalogue order: one pool and one Trade Row
feature each."""

BASES: tuple[str, ...] = tuple(
    name for name, card in CATALOGUE.items() if card.type is not CardType.SHIP
)
"""Every base's and Outpost's name: one in-play feature each."""

_PLAYER: tuple[str, ...] = (
    *(f"pool {name}" for name in CARDS),
    *(f"base {name}" for name in BASES),
    "authority",
    "base defense",
    "to discard",
)

NAMES: tuple[str, ...] = (
    *(f"ended: {name}" for name in _PLAYER),
    *(f"next: {name}" for name in _PLAYER),
    "turn",
    *(f"row {name}" for name in CARDS),
)
"""Each feature's name, in row order."""

_POOL = {name: i for i, name in enumerate(CARDS)}
_BASE = {name: len(CARDS) + i for i, name in enumerate(BASES)}
_AUTHORITY = len(CARDS) + len(BASES)
_TURN = 2 * len(_PLAYER)
_ROW = _TURN + 1


def _player(row: np.ndarray, offset: int, player: PlayerState) -> None:
    """Write `player`'s features into `row` from `offset`."""
    for card in player.deck + player.hand + player.discard + player.in_play:
        row[offset + _POOL[original(card).name]] += 1
    for base in player.bases:
        row[offset + _BASE[base.name]] += 1
    row[offset + _AUTHORITY] = player.authority
    row[offset + _AUTHORITY + 1] = sum(base.defense or 0 for base in player.bases)
    row[offset + _AUTHORITY + 2] = player.to_discard


def encode(state: GameState) -> np.ndarray:
    """`state`'s row, from the side of the player who just ended their turn."""
    row = np.zeros(len(NAMES))
    ended = 1 - state.current
    _player(row, 0, state.players[ended])
    _player(row, len(_PLAYER), state.players[state.current])
    row[_TURN] = state.turn
    for card in state.trade_row:
        row[_ROW + _POOL[card.name]] += 1
    return row


def encode_all(states: Iterable[GameState]) -> np.ndarray:
    """One row per state, stacked: shape `(states, len(NAMES))`."""
    rows = [encode(state) for state in states]
    return np.array(rows).reshape(len(rows), len(NAMES))
