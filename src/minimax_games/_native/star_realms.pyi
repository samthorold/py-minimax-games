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
    forward pass of `model` (`selfplay.native_model`: `form`, `mean`, `std`
    and `members`, a list of each member's parameters, and for a model
    reading the extra features, `extra`, their card tables, with the more
    features' under `extra["more"]` and the Trade Row features' under
    `extra["trade_row"]` if it reads them), the mean of its members'."""

def rollouts(
    model: dict[str, Any],
    state: dict[str, Any],
    lines: list[list[NativeAction]],
    seeds: list[int],
    worlds: int,
    budget: int,
    keyed: bool = False,
    threads: int = 1,
    max_turns: int = 300,
    keyed_worlds: bool = False,
) -> dict[str, np.ndarray]:
    """The regret instrument's rollouts: every line of `lines` played from
    `state` (the mover to move) in the world of every seed of `seeds`
    (sampled from the mover's view), then played out by the planner with
    `model` (`selfplay.native_model`) on both sides at `worlds` worlds and
    `budget` states, with keyed shuffles if `keyed` and the planners'
    worlds keyed by the seed if `keyed_worlds`, on `threads` threads,
    each stopping unfinished past `max_turns`: `winner` (int64, a row per
    line and a column per seed, -1 if unfinished) and `turns`. Raises
    `ValueError` if a line isn't legal in a seed's world."""

def shortlist(
    model: dict[str, Any], state: dict[str, Any], worlds: int, budget: int, seed: int
) -> list[tuple[list[NativeAction], float, float]]:
    """The candidates the planner with `model` at `worlds` worlds and
    `budget` states compares at `state` (the mover to move), its worlds
    sampled with an rng seeded with `seed` as its bot seeds it, one per
    distinct turn: `(line, score, chance)` best first, the score the mean
    over worlds of the line's best end's from the mover's side (`SPAN * (2p
    - 1)` for a probability `p`, a win as `WIN`), which the planner ranks
    by, and the chance the mean of those ends' chances (a win 1). The first
    is the line the planner plays."""

def self_play(
    players: list[dict[str, Any]],
    games: list[tuple[int, int, int]],
    threads: int = 1,
    max_turns: int = 300,
    record: bool = False,
    judge: bool = False,
    diagnose: bool = False,
    replies: bool = False,
) -> dict[str, Any]:
    """Play `games`, `(seed, seat 0, seat 1)` with the seats as indices into
    `players` (`{"kind": "greedy"}`, `{"kind": "heuristic"}` or `{"kind":
    "planner", "worlds": n, "budget": n, "model": {...}}`, with optionally
    `"reply": {"candidates": n, "worlds": n, "budget": n}` to look through
    the opponent's replies, or None) on `threads`
    threads, the GIL released: the finished games' turn-boundary rows
    (`features`, float32) with each row's label `y`, `game`, `position` in
    it (0 for the first turn's end) and `seat` that ended it; per game its
    `winner` (None if unfinished) and `turns`, and `plans` and `diverged` (a
    column per seat); with `record`, each game's `actions`; with `judge`,
    each row's `judgement` (float64) by the seat that ended it, NaN for a
    seat that doesn't judge; with `diagnose`, each row's `left` (int64, a
    column for the seat that ended it and one for the other), the cards in
    each seat's deck before its discard pile is reshuffled, and per plan of
    the finished games its `plan_game`, `plan_turn`, `plan_seat` (int64) and
    `plan_complete` (bool: whether it examined every end of the turn in
    every world), in play order; with `replies`, per plan of the finished
    games its `reply_game`, `reply_turn`, `reply_seat`, `reply_states` (the
    states it visited, its replies' included), `reply_looked` (candidates
    looked through, 0 if none), `reply_rank` (the pick's rank among them by
    the judgement, 1 for the judgement's own, 0 if none) (int64) and
    `reply_complete` (bool: whether every reply examined every end of the
    opponent's turn), in play order, and per game each seat's `seconds`
    spent planning (float64, a column per seat). None changes a game."""
