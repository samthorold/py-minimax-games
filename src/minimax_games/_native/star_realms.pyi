"""The Rust Star Realms engine (rust/src/star_realms/), for the whole base set.

Actions are tuples `(kind, *fields)` named as the Python `Action` variants
and their fields: a card as its name, a tuple of cards as a tuple of names,
and a `Choose`'s option as its index among the card's options. States and
views are dicts of the `GameState` and `View` fields, piles as lists of card
names.
"""

from typing import Any

import numpy as np

type NativeAction = tuple[Any, ...]

class Game:
    """A game dealt from `seed` and auto-played to its first decision, as
    `rules.auto_play(rules.new_game(seed))`."""

    def __init__(self, seed: int) -> None: ...
    def legal_actions(self) -> list[NativeAction]: ...
    def advance(self, action: NativeAction) -> None:
        """Take a decision, then auto-play. Raises `ValueError` if illegal."""
    def state(self) -> dict[str, Any]: ...
    def winner(self) -> int | None: ...
    def greedy_move(self) -> NativeAction: ...
    def heuristic_move(self) -> NativeAction:
        """The heuristic bot's choice from the current player's view."""
    def observe(self, observer: int) -> dict[str, Any]:
        """What `observer` knows (`view.observe`)."""
    def sample_world(self, observer: int, seed: int) -> Game:
        """A world consistent with `observer`'s view, as
        `view.sample_world(view, random.Random(seed))` samples it."""

def cards() -> list[dict[str, Any]]:
    """Each card's data, in catalogue order."""

def play_greedy(seed: int, max_turns: int = 1000) -> tuple[int | None, int, int]:
    """A greedy-vs-greedy game: `(winner, turns, decisions)`."""

def play_heuristic(seed: int, max_turns: int = 1000) -> tuple[int | None, int, int]:
    """A heuristic-vs-heuristic game: `(winner, turns, decisions)`."""

def bench_heuristic_games(games: int, base_seed: int) -> tuple[int, int, float]:
    """Time heuristic games: `(games, decisions, seconds)`."""

def bench_games(games: int, base_seed: int) -> tuple[int, int, float]:
    """Time greedy games: `(games, decisions, seconds)`."""

def bench_advance(games: int, base_seed: int) -> tuple[int, float]:
    """Time `advance` over recorded greedy games' decisions: `(calls, seconds)`."""

def plan_ends(
    worlds: list[dict[str, Any]], budget: int, ends: bool = False
) -> dict[str, Any]:
    """The whole-turn planner's enumeration over `worlds` (state dicts), at
    most `budget` states visited: the ends' feature rows (`features`, a
    float64 matrix) with each row's `world`, `winner` (-1 for none) and
    `current`; each candidate's line (`lines`, indices into
    `legal_actions`); per world the candidates `reached`, and the rows each
    one's value is the best of (`rows[offsets[k]:offsets[k + 1]]`);
    `states` and `complete`, as `planner.Plan`'s; with `ends`, the ends as
    state dicts."""

def predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """The win probability for each row of `x` (float64), by the Rust
    forward pass of `model` (`selfplay.native_model`)."""

def self_play(
    players: list[dict[str, Any]],
    games: list[tuple[int, int, int]],
    threads: int = 1,
    max_turns: int = 300,
    record: bool = False,
) -> dict[str, Any]:
    """Play `games`, `(seed, seat 0, seat 1)` with the seats as indices into
    `players` (`{"kind": "greedy"}`, `{"kind": "heuristic"}` or `{"kind":
    "planner", "worlds": n, "budget": n, "model": {...}}`) on `threads`
    threads, the GIL released: the finished games' turn-boundary rows
    (`features`, float32) with each row's label `y` and `game`; per game
    its `winner` (None if unfinished) and `turns`, and `plans` and
    `diverged` (a column per seat); with `record`, each game's `actions`."""
