"""What a player can do: the `Action` enum.

Only the variants the first slice's ships need exist so far; later slices add
`UseAlly`, `AttackBase` and `Choose` (docs/star-realms.md).
"""

from superenums import Variant

from minimax_games.games.star_realms.card import Card


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


class Scrap(Variant):
    """Scrap a card in play for its scrap ability, removing it from the game.

    It goes to the scrap heap, or, for an Explorer, back to the Explorer pile.
    Like `PlayCard`, the action names the card: with two copies in play, each
    `Scrap` takes one of them.
    """

    card: Card


class AttackPlayer(Variant):
    """Deal all the current combat to the opponent's Authority."""


class EndTurn(Variant):
    """Discard the hand and cards in play, lose unspent trade and combat, draw 5."""


type Action = PlayCard | Scrap | Acquire | AttackPlayer | EndTurn
