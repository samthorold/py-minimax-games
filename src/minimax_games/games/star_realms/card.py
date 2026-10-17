"""Cards: plain data, so the catalogue can later supply them from a file."""

from dataclasses import dataclass

from minimax_games.games.star_realms.effect import Combat, Effect, Trade


@dataclass(frozen=True, slots=True)
class Card:
    name: str
    effects: tuple[Effect, ...]


SCOUT = Card("Scout", (Trade(1),))
VIPER = Card("Viper", (Combat(1),))

STARTING_DECK: tuple[Card, ...] = (SCOUT,) * 8 + (VIPER,) * 2
