"""What a player can do: the `Action` enum.

Only the variants the starting decks need exist so far; later slices add
`UseAlly`, `Scrap`, `Acquire`, `AttackBase` and `Choose` (docs/star-realms.md).
"""

from superenums import Variant

from minimax_games.games.star_realms.card import Card


class PlayCard(Variant):
    """Play a card from hand.

    Identical cards are interchangeable, so the action names the card, not a
    position in the hand.
    """

    card: Card


class AttackPlayer(Variant):
    """Deal all the current combat to the opponent's Authority."""


class EndTurn(Variant):
    """Discard the hand and cards in play, lose unspent trade and combat, draw 5."""


type Action = PlayCard | AttackPlayer | EndTurn
