"""The Rust Star Realms engine (rust/src/star_realms/), for the first slice.

Actions are `(kind, card name or None)` pairs named as the Python `Action`
variants, and states dicts of the `GameState` fields the first slice uses,
piles as lists of card names.
"""

from typing import Any

type NativeAction = tuple[str, str | None]

class Game:
    """A first-slice game dealt from `seed` and auto-played to its first
    decision, as `rules.auto_play(rules.new_game(seed, deck))`."""

    def __init__(self, seed: int) -> None: ...
    def legal_actions(self) -> list[NativeAction]: ...
    def advance(self, kind: str, card: str | None = None) -> None:
        """Take a decision, then auto-play. Raises `ValueError` if illegal."""
    def state(self) -> dict[str, Any]: ...
    def winner(self) -> int | None: ...
    def greedy_move(self) -> NativeAction: ...

def cards() -> list[dict[str, Any]]:
    """Each card's data, in catalogue order."""

def play_greedy(seed: int, max_turns: int = 1000) -> tuple[int | None, int, int]:
    """A greedy-vs-greedy game: `(winner, turns, decisions)`."""

def bench_games(games: int, base_seed: int) -> tuple[int, int, float]:
    """Time greedy games: `(games, decisions, seconds)`."""

def bench_advance(games: int, base_seed: int) -> tuple[int, float]:
    """Time `advance` over recorded greedy games' decisions: `(calls, seconds)`."""
