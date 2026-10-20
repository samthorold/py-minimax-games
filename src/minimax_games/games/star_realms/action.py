"""What a player can do: the `Action` enum.

The variants the base set's cards need (docs/star-realms.md).
"""

from superenums import Variant

from minimax_games.games.star_realms.card import Card
from minimax_games.games.star_realms.effect import Effect


class PlayCard(Variant):
    """Play a card from hand.

    Identical cards are interchangeable, so the action names the card, not a
    position in the hand.
    """

    card: Card


class Acquire(Variant):
    """Spend trade equal to `card`'s cost to put it in the discard pile.

    The card comes from the Trade Row, which is then refilled from the Trade
    Deck, or, for an Explorer, from the Explorer pile.
    """

    card: Card


class AcquireOnTop(Variant):
    """`Acquire` `card`, a ship (an Explorer among them), but put it on top
    of the deck instead of in the discard pile, for a `NextShipOnTop`
    ability (Freighter's ally ability, Central Office).

    Offered beside `Acquire` for each affordable ship while the turn has
    one left (`GameState.may_on_top`), using one.
    """

    card: Card


class AcquireFreeShip(Variant):
    """Acquire `card`, a ship in the Trade Row, without paying its cost, and
    put it on top of the deck, for an `AcquireShipFreeOnTop` ability (Blob
    Carrier's ally ability). The Trade Deck's top card takes its place.

    Offered, as the only legal actions, once the player has used the ability
    (`GameState.free_ships`): one per distinct ship in the row. Like
    `Acquire`, the action names the card.
    """

    card: Card


class Scrap(Variant):
    """Scrap a card in play for its scrap ability, removing it from the game.

    It goes to the scrap heap, or, for an Explorer, back to the Explorer pile.
    Like `PlayCard`, the action names the card: with two copies in play, each
    `Scrap` takes one of them.
    """

    card: Card


class UseAlly(Variant):
    """Use the ally ability of `card`, which is in play, once another card of
    its faction is in play too.

    Each card's ally ability is used at most once per turn. Like `PlayCard`,
    the action names the card: with two copies in play, each `UseAlly` uses
    one of them.
    """

    card: Card


class UseBase(Variant):
    """Use the primary ability of `base`, one of the player's bases in play.

    A base's primary ability is used once in each of its owner's turns.
    Playing a base uses it for the turn it is played; on later turns this
    action uses it. Like `PlayCard`, the action names the card: with two
    copies in play, each `UseBase` uses one of them.
    """

    card: Card


class AttackPlayer(Variant):
    """Deal all the current combat to the opponent's Authority.

    Not allowed while the opponent has an Outpost in play.
    """


class AttackBase(Variant):
    """Spend combat equal to `card`'s defense to destroy it, one of the
    opponent's bases; it goes to the opponent's discard pile.

    While the opponent has an Outpost in play, only Outposts can be attacked.
    """

    card: Card


class Discard(Variant):
    """Discard `card` from hand, for an opponent's `OpponentDiscards` effect.

    A player with discards pending (`PlayerState.to_discard`) makes them at
    the start of their turn, before anything else, one `Discard` each.
    Like `PlayCard`, the action names the card, not a position in the hand.
    """

    card: Card


class ScrapHandCard(Variant):
    """Scrap `card` from hand: it goes to the scrap heap (an Explorer back to
    its pile), without its scrap ability.

    Offered while the turn has optional scraps left (`GameState.may_scrap`,
    from `ScrapHandOrDiscard` effects), and, as the only legal actions, while
    a scrap from hand is owed (`GameState.must_scrap`, Machine Base). Like
    `PlayCard`, the action names the card, not a position in the hand.
    """

    card: Card


class ScrapDiscardCard(Variant):
    """Scrap `card` from the discard pile, like `ScrapHandCard` from hand.

    Offered while the turn has optional scraps left (`GameState.may_scrap`).
    """

    card: Card


class ScrapTradeRowCard(Variant):
    """Scrap `card` from the Trade Row, for a `ScrapTradeRow` ability: it goes
    to the scrap heap, and the Trade Deck's top card takes its place.

    Offered while the turn has Trade Row scraps left
    (`GameState.may_scrap_row`). Like `PlayCard`, the action names the card,
    not a position in the row.
    """

    card: Card


class DestroyTarget(Variant):
    """Destroy `card`, one of the opponent's bases, for a `DestroyBase`
    ability: it goes to its owner's discard pile, with no combat spent.

    Offered while the turn has destroy-base abilities left
    (`GameState.may_destroy`), for any of the opponent's bases: Outposts
    don't protect the others from it. Like `PlayCard`, the action names the
    card.
    """

    card: Card


class Choose(Variant):
    """Pick `option`, one of the options of `card`'s "or" choice (a `Choice`
    effect), whose effects then apply.

    Offered, as the only legal actions, once the player has used the
    ability (`GameState.choosing`): one `Choose` per option, in the card's
    order. The action names the card and the option's effects, so a log
    or a bot can read the choice without looking it up.
    """

    card: Card
    option: tuple[Effect, ...]


class Recycle(Variant):
    """Discard `cards` from hand, then draw as many, for a `DiscardThenDraw`
    ability (Recycling Station: "discard up to two cards, then draw that
    many cards").

    Offered, as the only legal actions, once the player has chosen that
    option (`GameState.recycling`): one `Recycle` per distinct set of one
    card up to that many from hand, `cards` sorted by name. Discarding none
    does nothing, so it isn't offered (the other option was better). The
    draw may shuffle the discard pile, discarded cards included.
    """

    cards: tuple[Card, ...]


class Copy(Variant):
    """Make Stealth Needle, just played, a copy of `card`, another ship in
    play, for its `CopyShip` ability ("copy another ship you've played this
    turn"): `card`'s primary ability applies, and until the end of the turn
    the Needle has its ally and scrap abilities and its faction, as well as
    Machine Cult.

    Offered, as the only legal actions, once the Needle is played
    (`GameState.copying`): one `Copy` per distinct other ship in play. Like
    `PlayCard`, the action names the card.
    """

    card: Card


class ScrapAndDraw(Variant):
    """Scrap `hand` from hand and `discard` from the discard pile, then draw
    a card for each card scrapped, for a `ScrapThenDraw` ability (Brain
    World: "scrap up to two cards from your hand and/or discard pile, then
    draw a card for each card scrapped this way").

    Offered, as the only legal actions, once the player has used the
    ability (`GameState.scrap_to_draw`): one `ScrapAndDraw` per distinct
    choice of up to that many cards from hand and discard pile, none
    included, each tuple sorted by name. The scrapped cards go to the scrap
    heap (Explorers back to their pile) without their scrap abilities.
    """

    hand: tuple[Card, ...]
    discard: tuple[Card, ...]


class EndTurn(Variant):
    """Discard the hand and ships in play (bases stay), lose unspent trade and
    combat, draw 5."""


type Action = (
    PlayCard
    | UseAlly
    | UseBase
    | Scrap
    | Acquire
    | AcquireOnTop
    | AcquireFreeShip
    | AttackPlayer
    | AttackBase
    | DestroyTarget
    | Discard
    | ScrapHandCard
    | ScrapDiscardCard
    | ScrapTradeRowCard
    | Choose
    | Copy
    | Recycle
    | ScrapAndDraw
    | EndTurn
)
