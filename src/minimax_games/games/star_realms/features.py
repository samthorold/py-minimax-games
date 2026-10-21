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

`extra` adds hand-made features (#173), computed from a row's counts alone,
so a dataset of rows needs no regenerating: per player, the combat a hand
of 5 drawn from their pool holds on average plus their bases' combat, the
kill threat (that combat less the other player's Authority and base
defense), and per faction the pool's density of it, the expected number of
same-faction pairs in a hand of 5 (ally abilities fire on pairs), and the
faction's bases in play. The Rust forward pass computes them identically,
from the tables `extra_tables` gives it.
"""

from collections.abc import Iterable

import numpy as np

from minimax_games.games.star_realms.card import Card, CardType, Faction
from minimax_games.games.star_realms.catalogue import CATALOGUE
from minimax_games.games.star_realms.effect import Combat
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


HAND = 5
"""The cards a hand draws, for the extra features."""

_FACTIONS: tuple[Faction, ...] = tuple(Faction)

_SIDE: tuple[str, ...] = (
    "combat per hand",
    "kill threat",
    *(
        f"{faction.value} {what}"
        for faction in _FACTIONS
        for what in ("density", "pairs", "bases")
    ),
)

EXTRA: tuple[str, ...] = (
    *(f"ended: {name}" for name in _SIDE),
    *(f"next: {name}" for name in _SIDE),
)
"""Each extra feature's name, in `extra`'s column order."""


def _combat(card: Card) -> int:
    """The combat `card`'s primary ability gives unconditionally."""
    return sum(e.amount for e in card.primary if isinstance(e, Combat))


def extra_tables() -> dict[str, np.ndarray]:
    """What `extra` reads of the cards, by catalogue index: each card's
    primary combat, its faction's index in `Faction` (-1 for none), and each
    base's index among the cards. The Rust forward pass takes these."""
    cards = [CATALOGUE[name] for name in CARDS]
    return {
        "combat": np.array([_combat(c) for c in cards], dtype=np.float64),
        "faction": np.array(
            [-1 if c.faction is None else _FACTIONS.index(c.faction) for c in cards],
            dtype=np.int64,
        ),
        "base_card": np.array([_POOL[name] for name in BASES], dtype=np.int64),
    }


_TABLES = extra_tables()
_PLAYER_WIDTH = len(_PLAYER)


def extra(x: np.ndarray) -> np.ndarray:
    """The extra features (`EXTRA`) of rows `x` of `NAMES`, as float64,
    shape `(rows, len(EXTRA))`. Every sum is of small integers, so exact in
    any order; each feature then takes one division."""
    x = np.asarray(x, dtype=np.float64)
    combat, faction = _TABLES["combat"], _TABLES["faction"]
    base_card = _TABLES["base_card"]
    cards, bases = len(CARDS), len(BASES)
    sides = []
    for offset in (0, _PLAYER_WIDTH):
        pool = x[:, offset : offset + cards]
        in_play = x[:, offset + cards : offset + cards + bases]
        n = pool.sum(axis=1)
        held = pool @ combat
        per_hand = np.divide(HAND * held, n, out=np.zeros_like(n), where=n > 0)
        per_hand = per_hand + in_play @ combat[base_card]
        columns = [per_hand]
        for f in range(len(_FACTIONS)):
            nf = pool[:, faction == f].sum(axis=1)
            pairs = n * (n - 1)
            columns.append(np.divide(nf, n, out=np.zeros_like(n), where=n > 0))
            columns.append(
                np.divide(
                    HAND * (HAND - 1) // 2 * nf * (nf - 1),
                    pairs,
                    out=np.zeros_like(n),
                    where=pairs > 0,
                )
            )
            columns.append(in_play[:, faction[base_card] == f].sum(axis=1))
        sides.append(columns)
    out = []
    for side, other in ((0, _PLAYER_WIDTH), (1, 0)):
        authority = x[:, other + _AUTHORITY]
        defense = x[:, other + _AUTHORITY + 1]
        per_hand, *factions = sides[side]
        out += [per_hand, per_hand - (authority + defense), *factions]
    return np.stack(out, axis=1)
