"""Card definitions: plain frozen data, supplied by the catalogue (`cards.toml`)."""

from dataclasses import dataclass
from enum import Enum

from minimax_games.games.star_realms.effect import Effect


class Faction(Enum):
    BLOB = "Blob"
    TRADE_FEDERATION = "Trade Federation"
    STAR_EMPIRE = "Star Empire"
    MACHINE_CULT = "Machine Cult"


class CardType(Enum):
    SHIP = "ship"
    BASE = "base"
    OUTPOST = "outpost"


@dataclass(frozen=True, slots=True)
class Card:
    """One card's game data.

    `primary` effects happen when the card is played, `ally` ones once another
    card of the same faction is in play, `scrap` ones when the card is
    scrapped for its ability. `faction` is `None` for unaligned cards (Scout,
    Viper, Explorer); `defense` is `None` for ships. `copies` is how many of
    the card the game has.
    """

    name: str
    cost: int = 0
    faction: Faction | None = None
    type: CardType = CardType.SHIP
    defense: int | None = None
    primary: tuple[Effect, ...] = ()
    ally: tuple[Effect, ...] = ()
    scrap: tuple[Effect, ...] = ()
    copies: int = 1

    def __hash__(self) -> int:
        """The name's hash: equal cards have equal names, so this agrees with
        `==`, and the catalogue rejects two cards with one name, so its cards
        don't collide. The generated hash, over every field, was a quarter of
        the search's time (two fields are enums, hashed in Python); `str`
        caches its hash."""
        return hash(self.name)
