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

`extra(x, more=True)` appends `MORE` to them (#178), also from the counts:
per player the trade and the Authority a hand of 5 holds on average plus
their bases', the pool's density of cards that draw, that scrap and that
make the opponent discard (in any of a card's abilities) and of bases, and
the defense of their outposts in play; then the Authority, combat and trade
leads of the player who just ended over the other.
"""

from collections.abc import Iterable
from typing import Any

import numpy as np

from minimax_games.games.star_realms.card import Card, CardType, Faction
from minimax_games.games.star_realms.catalogue import CATALOGUE
from minimax_games.games.star_realms.effect import (
    Authority,
    Choice,
    Combat,
    DiscardThenDraw,
    Draw,
    DrawIfBases,
    DrawPerFactionCard,
    Effect,
    OpponentDiscards,
    ScrapFromHand,
    ScrapHandOrDiscard,
    ScrapThenDraw,
    Trade,
)
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


_MORE_SIDE: tuple[str, ...] = (
    "trade per hand",
    "authority per hand",
    "draw density",
    "scrap density",
    "discard density",
    "base density",
    "outpost defense",
)

MORE: tuple[str, ...] = (
    *(f"ended: {name}" for name in _MORE_SIDE),
    *(f"next: {name}" for name in _MORE_SIDE),
    "authority lead",
    "combat lead",
    "trade lead",
)
"""Each of the more features' names (#178), in the order `extra(x,
more=True)` appends them after `EXTRA`'s."""


def _combat(card: Card) -> int:
    """The combat `card`'s primary ability gives unconditionally."""
    return sum(e.amount for e in card.primary if isinstance(e, Combat))


def _gives(card: Card, kind: type[Trade] | type[Authority]) -> int:
    """The trade or Authority `card`'s primary ability gives
    unconditionally."""
    return sum(e.amount for e in card.primary if isinstance(e, kind))


def _effects(effects: Iterable[Effect]) -> Iterable[Effect]:
    """`effects` and every option's, through `Choice`s."""
    for e in effects:
        if isinstance(e, Choice):
            for option in e.options:
                yield from _effects(option)
        else:
            yield e


_DRAWS = (Draw, DrawIfBases, DrawPerFactionCard, DiscardThenDraw, ScrapThenDraw)
_SCRAPS = (ScrapHandOrDiscard, ScrapThenDraw, ScrapFromHand)


def _has(card: Card, kinds: tuple[type, ...]) -> float:
    """1 if any of `card`'s abilities has an effect of `kinds`, else 0."""
    every = _effects((*card.primary, *card.ally, *card.scrap))
    return float(any(isinstance(e, kinds) for e in every))


def extra_tables(more: bool = False) -> dict[str, Any]:
    """What `extra` reads of the cards, by catalogue index: each card's
    primary combat, its faction's index in `Faction` (-1 for none), and each
    base's index among the cards; with `more`, `MORE`'s tables too, under
    `"more"`: each card's primary trade and Authority, whether it draws,
    scraps, makes the opponent discard or is a base (1 or 0), and each
    base's defense if it is an Outpost (else 0). The Rust forward pass takes
    these."""
    cards = [CATALOGUE[name] for name in CARDS]
    tables: dict[str, Any] = {
        "combat": np.array([_combat(c) for c in cards], dtype=np.float64),
        "faction": np.array(
            [-1 if c.faction is None else _FACTIONS.index(c.faction) for c in cards],
            dtype=np.int64,
        ),
        "base_card": np.array([_POOL[name] for name in BASES], dtype=np.int64),
    }
    if more:

        def column(values: Iterable[float]) -> np.ndarray:
            return np.array(list(values), dtype=np.float64)

        tables["more"] = {
            "trade": column(_gives(c, Trade) for c in cards),
            "authority": column(_gives(c, Authority) for c in cards),
            "draw": column(_has(c, _DRAWS) for c in cards),
            "scrap": column(_has(c, _SCRAPS) for c in cards),
            "discard": column(_has(c, (OpponentDiscards,)) for c in cards),
            "base": column(float(c.type is not CardType.SHIP) for c in cards),
            "outpost_defense": column(
                (b.defense or 0) if b.type is CardType.OUTPOST else 0
                for b in (CATALOGUE[name] for name in BASES)
            ),
        }
    return tables


_TABLES = extra_tables(more=True)
_PLAYER_WIDTH = len(_PLAYER)


def extra(x: np.ndarray, more: bool = False) -> np.ndarray:
    """The extra features (`EXTRA`) of rows `x` of `NAMES`, as float64,
    shape `(rows, len(EXTRA))`, and with `more`, `MORE`'s after them. Every
    sum is of small integers, so exact in any order; each feature then takes
    at most one division, and a lead one subtraction."""
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
    if more:
        out += _more(x, sides[0][0], sides[1][0])
    return np.stack(out, axis=1)


def _more(x: np.ndarray, combat: np.ndarray, other: np.ndarray) -> list[np.ndarray]:
    """`MORE`'s columns of rows `x` (float64), given the combat per hand of
    the player who ended and of the other."""
    tables = _TABLES["more"]
    base_card = _TABLES["base_card"]
    cards, bases = len(CARDS), len(BASES)
    columns: list[np.ndarray] = []
    trade: list[np.ndarray] = []
    for offset in (0, _PLAYER_WIDTH):
        pool = x[:, offset : offset + cards]
        in_play = x[:, offset + cards : offset + cards + bases]
        n = pool.sum(axis=1)
        for name in ("trade", "authority"):
            w = tables[name]
            held = np.divide(HAND * (pool @ w), n, out=np.zeros_like(n), where=n > 0)
            columns.append(held + in_play @ w[base_card])
        trade.append(columns[-2])
        for name in ("draw", "scrap", "discard", "base"):
            columns.append(
                np.divide(pool @ tables[name], n, out=np.zeros_like(n), where=n > 0)
            )
        columns.append(in_play @ tables["outpost_defense"])
    columns.append(x[:, _AUTHORITY] - x[:, _PLAYER_WIDTH + _AUTHORITY])
    columns.append(combat - other)
    columns.append(trade[0] - trade[1])
    return columns
