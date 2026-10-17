"""What a card does: the `Effect` enum.

The engine doesn't implement every variant yet: later card slices add them
(see docs/star-realms.md). Abilities a card says you "may" use are optional
when the engine implements them; the variant docstrings say which.
"""

from superenums import Variant


class Trade(Variant):
    amount: int


class Combat(Variant):
    amount: int


class Authority(Variant):
    amount: int


class Draw(Variant):
    cards: int


class OpponentDiscards(Variant):
    """Target opponent discards `cards` cards."""

    cards: int


class DestroyBase(Variant):
    """You may destroy target base."""


class ScrapHandOrDiscard(Variant):
    """You may scrap up to `cards` cards from your hand and/or discard pile."""

    cards: int


class ScrapTradeRow(Variant):
    """You may scrap up to `cards` cards in the Trade Row."""

    cards: int


class NextShipOnTop(Variant):
    """You may put the next ship you acquire this turn on top of your deck."""


class Choice(Variant):
    """Do exactly one of the `options`, each a group of effects."""

    options: "tuple[tuple[Effect, ...], ...]"


class AcquireShipFreeOnTop(Variant):
    """Acquire any ship for free and put it on top of your deck (Blob Carrier)."""


class DrawIfBases(Variant):
    """If you have at least `bases` bases in play, draw `cards` (Embassy Yacht)."""

    bases: int
    cards: int


class DrawPerFactionCard(Variant):
    """Draw a card for each card of this card's faction played this turn.

    Blob World: "for each Blob card that you've played this turn".
    """


class DiscardThenDraw(Variant):
    """Discard up to `cards` cards, then draw that many (Recycling Station)."""

    cards: int


class ScrapThenDraw(Variant):
    """Scrap up to `cards` cards from your hand and/or discard pile, then draw
    a card for each card scrapped (Brain World)."""

    cards: int


class ScrapFromHand(Variant):
    """Scrap a card from your hand: not optional (Machine Base)."""


class CopyShip(Variant):
    """Copy another ship you've played this turn (Stealth Needle)."""


class ShipsGainCombat(Variant):
    """Every ship you play gains `amount` combat (Fleet HQ)."""

    amount: int


class AllyForAllFactions(Variant):
    """This card counts as an ally for every faction (Mech World)."""


type Effect = (
    Trade
    | Combat
    | Authority
    | Draw
    | OpponentDiscards
    | DestroyBase
    | ScrapHandOrDiscard
    | ScrapTradeRow
    | NextShipOnTop
    | Choice
    | AcquireShipFreeOnTop
    | DrawIfBases
    | DrawPerFactionCard
    | DiscardThenDraw
    | ScrapThenDraw
    | ScrapFromHand
    | CopyShip
    | ShipsGainCombat
    | AllyForAllFactions
)
