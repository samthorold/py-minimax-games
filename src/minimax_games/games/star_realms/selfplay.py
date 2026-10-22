"""Self-play in Rust (docs/star-realms.md, Compiled engine: self-play): one
call plays a batch of whole seeded games across threads, the planner
scoring with a learned value's forward pass written in Rust, and returns the
training rows `fit` builds and each game's result.

A player is the whole-turn planner with a model (`Planner`: an earlier
model is just another `Planner`), or `"heuristic"` or `"greedy"`. A game is
`(seed, seat 0, seat 1)`, the seats' bots seeded with `seed` and `seed + 1`
as a tournament seeds them, so a game here is the game
`match.play_match` plays with `planner.PlannerBot(..., backend="rust")` and
`value.ValueEvaluator` (a test checks whole transcripts), except that it
stops unfinished past `max_turns`, as `fit.boundaries` does.

The rows are `fit.Dataset`'s `x` and `y`: the features of each state
`EndTurn` leaves (`features.encode`), labelled 1.0 if the player who just
ended went on to win, for the finished games, in game order, with each
row's game and position in it (`fit.Dataset` records the same), from which
`td` builds temporal-difference targets. `value.fit` fits them directly.
(`fit.Dataset.evaluate`, the gate's comparison, needs the Python
`evaluate`, so it isn't produced: self-play rounds are gated by
tournaments.)
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from minimax_games._native import star_realms as native
from minimax_games.games.star_realms import features
from minimax_games.games.star_realms.fit import MAX_TURNS
from minimax_games.games.star_realms.planner import BUDGET, WORLDS
from minimax_games.games.star_realms.value import Model, reads_more


@dataclass(frozen=True, slots=True, eq=False)
class Planner:
    """The whole-turn planner scoring ends with `model`, over `worlds`
    worlds and at most `budget` states per plan (`planner.PlannerBot`)."""

    model: Model
    worlds: int = WORLDS
    budget: int = BUDGET

    def __post_init__(self) -> None:
        if self.model.names != features.NAMES:
            msg = "the model was fitted to other features than features.NAMES"
            raise ValueError(msg)
        if self.worlds < 1 or self.budget < 2 * self.worlds:
            msg = (
                "the planner needs at least 1 world and 2 states per world:"
                f" {self.worlds} worlds, budget {self.budget}"
            )
            raise ValueError(msg)


type Player = Planner | Literal["heuristic", "greedy"]
type Pairing = tuple[int, Player, Player]
"""A game: its seed, then the players of seats 0 and 1."""


def native_model(model: Model) -> dict[str, Any]:
    """`model` as the Rust forward pass takes it: its form, its
    standardisation and each member's parameters, as float64 arrays, and
    for a model reading the extra features, the card tables they read
    (`features.extra_tables`, with `MORE`'s if it reads them)."""

    def arrays(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            name: np.ascontiguousarray(a, dtype=np.float64)
            for name, a in params.items()
        }

    crossing: dict[str, Any] = {
        "form": model.form,
        **arrays({"mean": model.mean, "std": model.std}),
        "members": [arrays(params) for params in model.members],
    }
    if model.extra:
        crossing["extra"] = features.extra_tables(more=reads_more(model.extra))
    return crossing


def _native_player(player: Player) -> dict[str, Any]:
    if isinstance(player, Planner):
        return {
            "kind": "planner",
            "worlds": player.worlds,
            "budget": player.budget,
            "model": native_model(player.model),
        }
    return {"kind": player}


@dataclass(frozen=True, slots=True, eq=False)
class SelfPlay:
    """What a batch of games came to: identical for the same games, with
    any number of threads."""

    x: np.ndarray
    """The finished games' turn-boundary rows (float32), in game order."""
    y: np.ndarray
    """Each row's label: 1.0 if the player who just ended won."""
    game: np.ndarray
    """Each row's game, as its index in the batch."""
    position: np.ndarray
    """Each row's place in its game: 0 for the state the first turn's
    `EndTurn` leaves, then 1, 2, ..., one per turn."""
    winners: tuple[int | None, ...]
    """Each game's winner, `None` if it passed `max_turns`."""
    turns: tuple[int, ...]
    plans: np.ndarray
    """Plans per game (rows) and seat (columns); 0 for a bot that doesn't
    plan."""
    diverged: np.ndarray
    """Plans made again because the game went otherwise than planned."""
    actions: list[list[native.NativeAction]] | None
    """Each game's decisions, if recorded."""

    @property
    def games(self) -> int:
        """The finished games, whose rows these are."""
        return sum(winner is not None for winner in self.winners)


def play(
    games: Sequence[Pairing],
    threads: int = 1,
    max_turns: int = MAX_TURNS,
    record: bool = False,
) -> SelfPlay:
    """Play `games` in Rust on `threads` threads, the GIL released, each
    stopping unfinished past `max_turns`; with `record`, keep every
    decision."""
    players: list[Player] = []
    index: dict[tuple[str, object], int] = {}

    def seat(player: Player) -> int:
        # each distinct player crosses once, however many games it plays
        key = (
            ("planner", id(player)) if isinstance(player, Planner) else ("bot", player)
        )
        if key not in index:
            index[key] = len(players)
            players.append(player)
        return index[key]

    seats = [(seed, seat(a), seat(b)) for seed, a, b in games]
    found = native.self_play(
        [_native_player(p) for p in players], seats, threads, max_turns, record
    )
    return SelfPlay(
        x=found["features"],
        y=found["y"],
        game=found["game"],
        position=found["position"],
        winners=tuple(found["winner"]),
        turns=tuple(found["turns"]),
        plans=found["plans"],
        diverged=found["diverged"],
        actions=found.get("actions"),
    )


def alternating(a: Player, b: Player, seeds: Iterable[int]) -> list[Pairing]:
    """Games from `seeds`, `a` moving first in the 1st, 3rd, ... and `b` in
    the others, as a tournament seats them."""
    return [
        (seed, a, b) if i % 2 == 0 else (seed, b, a) for i, seed in enumerate(seeds)
    ]
