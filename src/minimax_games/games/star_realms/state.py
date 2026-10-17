"""The immutable game state.

Every pile is a tuple, and the deck's order is concrete: the top card is
`deck[0]`. Randomness lives in `GameState.rng_seed`, an integer. Each shuffle
seeds a fresh `random.Random` from it, shuffles, and draws the next seed from
that generator, so the state stays immutable and hashable and one starting
seed fixes every shuffle in the game.

The Trade Row is `trade_row`, refilled from the top of `trade_deck`; the
Explorer pile holds `explorers` identical Explorers, so it is just a count.
Scrapped cards go to `scrap_heap`, in the order they were scrapped (except
Explorers, which go back to their pile).

`allies_used` lists the cards in play whose ally ability has been used this
turn, one entry per use. Identical copies are interchangeable, so a card's
ally ability is still available while it has more copies in play than entries
here. The list empties at the end of the turn.
"""

from dataclasses import dataclass

from minimax_games.games.star_realms.card import Card

STARTING_AUTHORITY = 50
HAND_SIZE = 5
FIRST_HAND_SIZE = 3
TRADE_ROW_SIZE = 5


@dataclass(frozen=True, slots=True)
class PlayerState:
    authority: int
    deck: tuple[Card, ...]
    hand: tuple[Card, ...] = ()
    in_play: tuple[Card, ...] = ()
    discard: tuple[Card, ...] = ()


@dataclass(frozen=True, slots=True)
class GameState:
    players: tuple[PlayerState, PlayerState]
    rng_seed: int
    trade_deck: tuple[Card, ...] = ()
    trade_row: tuple[Card, ...] = ()
    explorers: int = 0
    scrap_heap: tuple[Card, ...] = ()
    current: int = 0
    turn: int = 1
    trade: int = 0
    combat: int = 0
    allies_used: tuple[Card, ...] = ()
