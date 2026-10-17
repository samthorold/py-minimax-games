"""What a card does when played: the `Effect` enum.

Only the variants the starting decks need exist so far; later card slices
add more (see docs/star-realms.md).
"""

from superenums import Variant


class Trade(Variant):
    amount: int


class Combat(Variant):
    amount: int


type Effect = Trade | Combat
