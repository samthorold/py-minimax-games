"""A hand-written linear evaluation of a state (docs/star-realms.md, Search).

Each player's terms are scored alike and differenced, so the score is
antisymmetric: `evaluate(s, p) == -evaluate(s, 1 - p)`. A finished game
scores `WIN` or `-WIN`, beyond any score of a game still going on.
"""

from dataclasses import dataclass
from typing import assert_never

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.card import Card, CardType, Faction
from minimax_games.games.star_realms.catalogue import CATALOGUE, SCOUT, VIPER
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
from minimax_games.games.star_realms.state import GameState, PlayerState


@dataclass(frozen=True, slots=True)
class Weights:
    """What each term of the evaluation is worth, in points of Authority.

    The deck-economy term is the average value of the cards the player owns.
    A card's value is its primary abilities' value, plus its ally abilities'
    discounted by the chance they fire (`ally_hand`), plus its scrap
    abilities' discounted by `scrap_ability`. An ability's value is the sum
    of its effects' values, each from the per-effect weights below; a
    `Choice` is worth its best option. Each card's value is precomputed.
    """

    base_defense: float = 1
    """Per point of defense of a base in play (not an Outpost)."""
    outpost_defense: float = 2
    """Per point of defense of an Outpost in play."""
    trade: float = 20
    """Per point of trade a card gives."""
    combat: float = 20
    """Per point of combat a card gives."""
    authority: float = 8
    """Per point of Authority a card gives."""
    draw: float = 30
    """Per card a card draws."""
    opponent_discards: float = 20
    """Per card a card makes the opponent discard."""
    destroy_base: float = 20
    """Per base a card may destroy."""
    scrap_hand_or_discard: float = 15
    """Per card a card may scrap from the hand or discard pile."""
    scrap_trade_row: float = 6
    """Per card a card may scrap from the Trade Row."""
    next_ship_on_top: float = 15
    """For putting the next ship acquired on top of the deck."""
    acquire_ship_free_on_top: float = 80
    """For a free ship from the Trade Row, on top of the deck (Blob Carrier)."""
    draw_if_bases: float = 15
    """Per card a card draws if enough bases are in play (Embassy Yacht)."""
    draw_per_faction_card: float = 45
    """For a card per card of its faction played this turn (Blob World)."""
    discard_then_draw: float = 10
    """Per card a card may discard and redraw (Recycling Station)."""
    scrap_then_draw: float = 35
    """Per card a card may scrap and replace with a draw (Brain World)."""
    scrap_from_hand: float = 5
    """For a forced scrap from the hand (Machine Base)."""
    copy_ship: float = 40
    """For copying another ship played this turn (Stealth Needle)."""
    ships_gain_combat: float = 60
    """Per point of combat every ship played gains (Fleet HQ)."""
    ally_for_all_factions: float = 30
    """For counting as an ally of every faction (Mech World)."""
    scrap_ability: float = 0.25
    """What a scrap ability is worth against the same effects as a primary
    ability: it is used once, and costs the card."""
    ally_hand: int = 4
    """How many other cards a hand holds that could be a card's ally: an ally
    ability counts for the chance that at least one of them is of the card's
    faction, from the share of the player's other cards that are."""
    starter_card: float = -1
    """Per Scout or Viper the player still owns (a penalty, so negative)."""


WEIGHTS = Weights()
"""The weights `evaluate` uses unless given others."""

WIN = 10**9
"""The score of a won game (and, negated, of a lost one). Scores of games
still going on are capped strictly inside `-WIN`..`WIN`."""


def _owned(player: PlayerState) -> tuple[Card, ...]:
    """Every card `player` owns, in whichever pile."""
    return player.deck + player.hand + player.in_play + player.discard + player.bases


def effect_value(effect: Effect, weights: Weights = WEIGHTS) -> float:
    """What one effect is worth to a card's value."""
    match effect:
        case Trade(amount=amount):
            return weights.trade * amount
        case Combat(amount=amount):
            return weights.combat * amount
        case Authority(amount=amount):
            return weights.authority * amount
        case Draw(cards=cards):
            return weights.draw * cards
        case OpponentDiscards(cards=cards):
            return weights.opponent_discards * cards
        case DestroyBase():
            return weights.destroy_base
        case ScrapHandOrDiscard(cards=cards):
            return weights.scrap_hand_or_discard * cards
        case ScrapTradeRow(cards=cards):
            return weights.scrap_trade_row * cards
        case NextShipOnTop():
            return weights.next_ship_on_top
        case Choice(options=options):
            return max(_sum(option, weights) for option in options)
        case AcquireShipFreeOnTop():
            return weights.acquire_ship_free_on_top
        case DrawIfBases(cards=cards):
            return weights.draw_if_bases * cards
        case DrawPerFactionCard():
            return weights.draw_per_faction_card
        case DiscardThenDraw(cards=cards):
            return weights.discard_then_draw * cards
        case ScrapThenDraw(cards=cards):
            return weights.scrap_then_draw * cards
        case ScrapFromHand():
            return weights.scrap_from_hand
        case CopyShip():
            return weights.copy_ship
        case ShipsGainCombat(amount=amount):
            return weights.ships_gain_combat * amount
        case AllyForAllFactions():
            return weights.ally_for_all_factions
        case _:
            assert_never(effect)


def _sum(effects: tuple[Effect, ...], weights: Weights) -> float:
    return sum(effect_value(e, weights) for e in effects)


def _ally_chance(share: float, weights: Weights) -> float:
    """The chance an ally ability fires, if `share` of the player's other
    cards are of the card's faction."""
    return 1 - (1 - share) ** weights.ally_hand


_FACTIONS = {faction: i for i, faction in enumerate(Faction)}


@dataclass(frozen=True, slots=True)
class _CardValue:
    """One card's precomputed share of its owner's value."""

    card: Card
    own: float
    """Its primary and scrap abilities' value."""
    faction: int
    """Its faction's index in `Faction`, or -1 if it has none."""
    ally: float
    """Its ally abilities' value, before the chance they fire."""
    starter: float
    """The starter-card penalty, if it's a Scout or Viper."""


def _card_value(card: Card, weights: Weights) -> _CardValue:
    return _CardValue(
        card=card,
        own=_sum(card.primary, weights)
        + weights.scrap_ability * _sum(card.scrap, weights),
        faction=-1 if card.faction is None else _FACTIONS[card.faction],
        ally=_sum(card.ally, weights),
        starter=weights.starter_card if card in (SCOUT, VIPER) else 0,
    )


_TABLES: dict[int, tuple[Weights, dict[str, _CardValue]]] = {}
"""Card values by card name, per `Weights` (by identity): `evaluate` is in
the search's hot path, and a card's value never changes. Each table starts
with the catalogue's cards, whose names are unique. A lookup checks it found
the very card (or an equal one), so hand-built cards that reuse a name are
valued correctly, just not cached."""


def _table(weights: Weights) -> dict[str, _CardValue]:
    entry = _TABLES.get(id(weights))
    if entry is None or entry[0] is not weights:
        cards = {name: _card_value(c, weights) for name, c in CATALOGUE.items()}
        entry = _TABLES[id(weights)] = (weights, cards)
    return entry[1]


def _economy(owned: tuple[Card, ...], weights: Weights) -> float:
    """The average value of the cards the player owns, plus the starter-card
    penalty."""
    table = _table(weights)
    total = 0.0
    starters = 0.0
    counts = [0, 0, 0, 0]
    allies = [0.0, 0.0, 0.0, 0.0]
    for card in owned:
        value = table.get(card.name)
        if value is None or (value.card is not card and value.card != card):
            value = _card_value(card, weights)
            table.setdefault(card.name, value)
        total += value.own
        starters += value.starter
        if value.faction >= 0:
            counts[value.faction] += 1
            allies[value.faction] += value.ally
    others = len(owned) - 1
    if others:
        for count, ally in zip(counts, allies, strict=True):
            if ally:
                total += ally * _ally_chance((count - 1) / others, weights)
    return total / len(owned) + starters


def _value(player: PlayerState, weights: Weights) -> float:
    value: float = player.authority
    owned = _owned(player)
    if owned:
        value += _economy(owned, weights)
    for base in player.bases:
        per_point = (
            weights.outpost_defense
            if base.type is CardType.OUTPOST
            else weights.base_defense
        )
        value += per_point * (base.defense or 0)
    return value


def evaluate(state: GameState, perspective: int, weights: Weights = WEIGHTS) -> int:
    """How good `state` is for player `perspective`: bigger is better."""
    won_by = rules.winner(state)
    if won_by is not None:
        return WIN if won_by == perspective else -WIN
    me, them = state.players[perspective], state.players[1 - perspective]
    score = round(_value(me, weights) - _value(them, weights))
    return max(-WIN + 1, min(WIN - 1, score))
