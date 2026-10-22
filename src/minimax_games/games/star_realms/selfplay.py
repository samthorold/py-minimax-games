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


@dataclass(frozen=True, slots=True)
class Reply:
    """The Rust planner's reply option (docs/star-realms.md, Learned values:
    looking through the reply): the plan's best `candidates` are each worth
    the mean over worlds of the player's chance after the opponent replies
    from the candidate's best end in that world, the opponent planning its
    turn from `worlds` sub-worlds of its own view at `budget` states in
    all, and the best of them is played. Rust only: the Python planner has
    no reply."""

    candidates: int = 3
    worlds: int = 2
    budget: int = 5_000

    def __post_init__(self) -> None:
        if self.candidates < 1 or self.worlds < 1 or self.budget < 2 * self.worlds:
            msg = (
                "a reply needs at least 1 candidate, 1 world and 2 states per"
                f" world: {self}"
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True, eq=False)
class Planner:
    """The whole-turn planner scoring ends with `model`, over `worlds`
    worlds and at most `budget` states per plan (`planner.PlannerBot`),
    looking through the opponent's replies if `reply` is given (Rust only).
    """

    model: Model
    worlds: int = WORLDS
    budget: int = BUDGET
    reply: Reply | None = None

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
            "reply": None
            if player.reply is None
            else {
                "candidates": player.reply.candidates,
                "worlds": player.reply.worlds,
                "budget": player.reply.budget,
            },
        }
    return {"kind": player}


@dataclass(frozen=True, slots=True, eq=False)
class Examined:
    """The planners' plans in the finished games, in play order: each
    plan's game (its index in the batch), turn (the game's, both players'
    counted, 1 first), seat, and whether it examined every end of the turn
    in every sampled world within its budget (`planner.Plan.complete`). A
    turn has as many plans as it has chance events to plan past, and one
    more for each time the game went otherwise than planned."""

    game: np.ndarray
    turn: np.ndarray
    seat: np.ndarray
    complete: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class Replies:
    """The planners' plans in the finished games, in play order, with what
    looking through the opponent's replies came to: each plan's game, turn,
    seat, `states` visited (its replies' included), the candidates
    `looked` through (0 if fewer than 2 competed, or the planner has no
    reply), the pick's `rank` among them by the judgement (1 for the
    judgement's own pick, so above 1 the reply changed the choice; 0 if none
    was looked through), and whether every reply examined every end of the
    opponent's turn (`complete`)."""

    game: np.ndarray
    turn: np.ndarray
    seat: np.ndarray
    states: np.ndarray
    looked: np.ndarray
    rank: np.ndarray
    complete: np.ndarray


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
    seat: np.ndarray
    """Each row's seat that had just ended its turn (0 moves first)."""
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
    judgement: np.ndarray | None = None
    """If asked for, each row's judgement by the player who ended the turn:
    its model's win probability for the row (`Model.predict`), NaN for a
    player who doesn't judge (the heuristic, greedy)."""
    left: np.ndarray | None = None
    """If asked for (`diagnose`), each row's cards left to draw before a
    reshuffle: the cards in the deck of the player who had just ended
    (column 0) and of the next player (column 1), which their next draws
    take before their discard pile is shuffled into a new deck."""
    examined: Examined | None = None
    """If asked for (`diagnose`), the planners' plans and whether each
    examined every end of its turn."""
    replies: Replies | None = None
    """If asked for (`replies`), the planners' plans and their replies."""
    seconds: np.ndarray | None = None
    """If asked for (`replies`), each game's (rows) seconds each seat
    (columns) spent planning: measured, so it varies by run."""

    @property
    def games(self) -> int:
        """The finished games, whose rows these are."""
        return sum(winner is not None for winner in self.winners)


def play(
    games: Sequence[Pairing],
    threads: int = 1,
    max_turns: int = MAX_TURNS,
    record: bool = False,
    judge: bool = False,
    diagnose: bool = False,
    replies: bool = False,
) -> SelfPlay:
    """Play `games` in Rust on `threads` threads, the GIL released, each
    stopping unfinished past `max_turns`; with `record`, keep every
    decision; with `judge`, each played end's judgement by whoever played
    it; with `diagnose`, what the diagnostics need (`SelfPlay.left` and
    `SelfPlay.examined`); with `replies`, each plan's states and replies
    and each seat's time planning (`SelfPlay.replies`, `SelfPlay.seconds`).
    None changes a game."""
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
        [_native_player(p) for p in players],
        seats,
        threads,
        max_turns,
        record,
        judge,
        diagnose,
        replies,
    )
    return SelfPlay(
        x=found["features"],
        y=found["y"],
        game=found["game"],
        position=found["position"],
        seat=found["seat"],
        winners=tuple(found["winner"]),
        turns=tuple(found["turns"]),
        plans=found["plans"],
        diverged=found["diverged"],
        actions=found.get("actions"),
        judgement=found.get("judgement"),
        left=found.get("left"),
        examined=Examined(
            game=found["plan_game"],
            turn=found["plan_turn"],
            seat=found["plan_seat"],
            complete=found["plan_complete"],
        )
        if diagnose
        else None,
        replies=Replies(
            game=found["reply_game"],
            turn=found["reply_turn"],
            seat=found["reply_seat"],
            states=found["reply_states"],
            looked=found["reply_looked"],
            rank=found["reply_rank"],
            complete=found["reply_complete"],
        )
        if replies
        else None,
        seconds=found.get("seconds"),
    )


def alternating(a: Player, b: Player, seeds: Iterable[int]) -> list[Pairing]:
    """Games from `seeds`, `a` moving first in the 1st, 3rd, ... and `b` in
    the others, as a tournament seats them."""
    return [
        (seed, a, b) if i % 2 == 0 else (seed, b, a) for i, seed in enumerate(seeds)
    ]
