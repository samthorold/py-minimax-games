"""The immutable game state.

Every pile is a tuple, and the deck's order is concrete: the top card is
`deck[0]`. Randomness lives in `GameState.rng_seed`, an integer. Each shuffle
seeds a fresh `random.Random` from it, shuffles, and draws the next seed from
that generator, so the state stays immutable and hashable and one starting
seed fixes every shuffle in the game.
"""

from dataclasses import dataclass

from minimax_games.games.star_realms.card import Card

STARTING_AUTHORITY = 50
HAND_SIZE = 5
FIRST_HAND_SIZE = 3


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
    current: int = 0
    turn: int = 1
    trade: int = 0
    combat: int = 0
