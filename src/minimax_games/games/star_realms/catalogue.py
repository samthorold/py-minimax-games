"""The card catalogue: card definitions read from `cards.toml`."""

import tomllib
from collections import Counter
from collections.abc import Callable, Mapping
from importlib import resources
from types import MappingProxyType
from typing import Any

from minimax_games.games.star_realms.card import Card, CardType, Faction
from minimax_games.games.star_realms.effect import (
    AcquireShipFreeOnTop,
    AllyForAllFactions,
    Authority,
    Choice,
    Combat,
    CopyShip,
    DestroyBase,
    DiscardThenDraw,
    Draw,
    DrawIfBases,
    DrawPerFactionCard,
    Effect,
    NextShipOnTop,
    OpponentDiscards,
    ScrapFromHand,
    ScrapHandOrDiscard,
    ScrapThenDraw,
    ScrapTradeRow,
    ShipsGainCombat,
    Trade,
)

_COUNTED: dict[str, Callable[[int], Effect]] = {
    "trade": Trade,
    "combat": Combat,
    "authority": Authority,
    "draw": Draw,
    "opponent_discards": OpponentDiscards,
    "scrap_hand_or_discard": ScrapHandOrDiscard,
    "scrap_trade_row": ScrapTradeRow,
    "discard_then_draw": DiscardThenDraw,
    "scrap_then_draw": ScrapThenDraw,
    "ships_gain_combat": ShipsGainCombat,
}
"""Effects written `{ name = n }`."""

_UNIT: dict[str, Effect] = {
    "destroy_base": DestroyBase(),
    "next_ship_on_top": NextShipOnTop(),
    "acquire_ship_free_on_top": AcquireShipFreeOnTop(),
    "draw_per_faction_card": DrawPerFactionCard(),
    "scrap_from_hand": ScrapFromHand(),
    "copy_ship": CopyShip(),
    "ally_for_all_factions": AllyForAllFactions(),
}
"""Effects written as a bare string."""


def _effect(raw: Any) -> Effect:
    match raw:
        case str() if raw in _UNIT:
            return _UNIT[raw]
        case {"choice": [*options]} if len(raw) == 1 and len(options) >= 2:
            return Choice(tuple(_effects(option) for option in options))
        case {"draw_if_bases": {"bases": int(bases), "cards": int(cards)} as args} if (
            len(raw) == 1 and len(args) == 2
        ):
            return DrawIfBases(bases, cards)
        case dict() if len(raw) == 1:
            ((key, value),) = raw.items()
            if key in _COUNTED and type(value) is int:
                return _COUNTED[key](value)
        case _:
            pass
    msg = f"unknown effect {raw!r}"
    raise ValueError(msg)


def _effects(raw: Any) -> tuple[Effect, ...]:
    if not isinstance(raw, list):
        msg = f"expected a list of effects, got {raw!r}"
        raise TypeError(msg)
    return tuple(_effect(e) for e in raw)


def _card(raw: dict[str, Any]) -> Card:
    try:
        return _fields(raw)
    except (KeyError, TypeError, ValueError) as e:
        msg = f"card {raw.get('name')!r}: {e}"
        raise ValueError(msg) from e


_KEYS = {
    "name",
    "faction",
    "type",
    "cost",
    "defense",
    "copies",
    "primary",
    "ally",
    "scrap",
}


def _int(raw: dict[str, Any], key: str, minimum: int) -> int:
    value = raw[key]
    if type(value) is not int or value < minimum:
        msg = f"{key} must be an integer of at least {minimum}, got {value!r}"
        raise ValueError(msg)
    return value


def _fields(raw: dict[str, Any]) -> Card:
    unknown = raw.keys() - _KEYS
    if unknown:
        msg = f"unknown keys {sorted(unknown)}"
        raise ValueError(msg)
    faction = raw.get("faction")
    card_type = CardType(raw["type"])
    defense = None
    if card_type is CardType.SHIP:
        if "defense" in raw:
            msg = "a ship has no defense"
            raise ValueError(msg)
    else:
        defense = _int(raw, "defense", 1)
    return Card(
        name=raw["name"],
        cost=_int(raw, "cost", 0),
        faction=None if faction is None else Faction(faction),
        type=card_type,
        defense=defense,
        primary=_effects(raw.get("primary", [])),
        ally=_effects(raw.get("ally", [])),
        scrap=_effects(raw.get("scrap", [])),
        copies=_int(raw, "copies", 1),
    )


def parse(text: str) -> tuple[Card, ...]:
    """Parse catalogue TOML into card definitions, in file order.

    Raises `ValueError`, naming the card, for anything malformed: an unknown
    key, faction, type or effect, a missing field, a ship with defense or a
    base without, or a name used twice.
    """
    cards = tuple(_card(raw) for raw in tomllib.loads(text)["card"])
    names = Counter(card.name for card in cards)
    duplicates = sorted(name for name, n in names.items() if n > 1)
    if duplicates:
        msg = f"cards named more than once: {duplicates}"
        raise ValueError(msg)
    return cards


def load() -> tuple[Card, ...]:
    """The base-set catalogue, read from the packaged `cards.toml`."""
    text = resources.files(__package__).joinpath("cards.toml").read_text("utf-8")
    return parse(text)


CATALOGUE: Mapping[str, Card] = MappingProxyType({card.name: card for card in load()})
"""Every card definition by name."""

SCOUT = CATALOGUE["Scout"]
VIPER = CATALOGUE["Viper"]
EXPLORER = CATALOGUE["Explorer"]

STARTING_DECK: tuple[Card, ...] = (SCOUT,) * 8 + (VIPER,) * 2
"""Each player's starting deck."""

TRADE_DECK: tuple[Card, ...] = tuple(
    card
    for card in CATALOGUE.values()
    if card.faction is not None
    for _ in range(card.copies)
)
"""The 80 faction cards, each repeated `copies` times, in catalogue order."""

IMPLEMENTED: frozenset[type[Effect]] = frozenset(
    {Trade, Combat, Authority, Draw, OpponentDiscards}
)
"""The kinds of effect the rules implement. Each card slice adds its own
(docs/star-realms.md): the playable Trade Deck and which abilities `rules`
offers both follow from this one set."""


def playable(card: Card, implemented: frozenset[type[Effect]] = IMPLEMENTED) -> bool:
    """Whether every effect of `card`, primary, ally or scrap, is of a kind in
    `implemented`."""
    return all(type(e) in implemented for e in card.primary + card.ally + card.scrap)


def playable_deck(
    implemented: frozenset[type[Effect]] = IMPLEMENTED,
) -> tuple[Card, ...]:
    """Every copy of the Trade Deck cards whose effects are all of a kind in
    `implemented` (ships, bases and outposts), in catalogue order."""
    return tuple(card for card in TRADE_DECK if playable(card, implemented))


PLAYABLE_TRADE_DECK: tuple[Card, ...] = playable_deck()
"""The Trade Deck games use: every copy of the cards whose effects the rules
all implement."""

EXPLORER_PILE = 10
"""How many Explorers the game starts with."""
