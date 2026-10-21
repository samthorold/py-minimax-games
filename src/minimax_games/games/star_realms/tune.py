"""Tune `evaluate`'s weights by tournament (docs/star-realms.md, Evaluation).

`search` is a coordinate search: for each weight in turn it tries the weight
multiplied and divided by a factor, plays each candidate (the cheat with
those weights against the heuristic) on the same seeds as the incumbent, and
keeps the better candidate only if it beats the incumbent by more than noise.
The comparison is paired: on the same seeds, the games a weight change
doesn't affect are the same games, so only the seeds whose outcome changed
(the discordant ones) carry information. A candidate is kept if its extra
wins exceed `z` times the square root of the discordant seeds (a sign test,
normal approximation).

Every step is printed, and each trial's per-seed outcomes can be kept in a
cache file (`TrialCache`), so a long run stopped part way resumes where it
stopped, replaying the trials it already played from the file.

`screen` is a cheap screen for which weights matter: at states the cheat
decides in, how often does multiplying or dividing one weight by the factor
change its move? A weight that never changes a decision can't change a game.
"""

import json
import math
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
from functools import partial
from pathlib import Path

from minimax_games.games.star_realms import cheat, profiling
from minimax_games.games.star_realms.bots import HeuristicBot
from minimax_games.games.star_realms.cheat import CheatBot
from minimax_games.games.star_realms.evaluate import WEIGHTS, Linear, Weights
from minimax_games.games.star_realms.state import GameState
from minimax_games.games.star_realms.tournament import MAX_TURNS, play_tournament

FACTOR = 2.0
"""How far each step moves a weight: multiplied or divided by this."""

Z = 1.64
"""How many standard deviations of noise a gain must pass (one-sided 5%)."""

FIELDS = tuple(field.name for field in fields(Weights))
"""Every weight, in `Weights`' order."""


type Outcomes = tuple[bool, ...]
"""Per seed, in order: did the cheat win? (An unfinished game is no win.)"""


def _cheat(seed: int, budget: int, weights: Weights) -> CheatBot:
    return CheatBot(budget=budget, evaluator=Linear(weights))


def _heuristic(seed: int) -> HeuristicBot:
    return HeuristicBot()


def play(
    weights: Weights, base_seed: int, games: int, budget: int, jobs: int = 1
) -> Outcomes:
    """The cheat with `weights` against the heuristic, `games` games from
    `base_seed` with alternating seats (the cheat first in even games), as
    the tournament command plays them."""
    played = play_tournament(
        (partial(_cheat, budget=budget, weights=weights), _heuristic),
        games,
        base_seed,
        MAX_TURNS,
        jobs,
    )
    return tuple(game.winner == 0 for game in played.result.games)


@dataclass(frozen=True, slots=True)
class Comparison:
    """A candidate's outcomes against the incumbent's, seed by seed."""

    gain: int
    """The candidate's wins less the incumbent's."""
    discordant: int
    """Seeds one won and the other didn't."""

    @property
    def noise(self) -> float:
        """One standard deviation of `gain` if the two were equally good."""
        return math.sqrt(self.discordant)

    def improves(self, z: float = Z) -> bool:
        """Is the gain more than `z` standard deviations of noise?"""
        return self.gain > z * self.noise


def compare(candidate: Outcomes, incumbent: Outcomes) -> Comparison:
    """Pair two runs on the same seeds."""
    if len(candidate) != len(incumbent):
        msg = f"different seeds: {len(candidate)} and {len(incumbent)} games"
        raise ValueError(msg)
    return Comparison(
        gain=sum(candidate) - sum(incumbent),
        discordant=sum(c != i for c, i in zip(candidate, incumbent, strict=True)),
    )


def neighbours(weights: Weights, field: str, factor: float = FACTOR) -> list[Weights]:
    """`weights` with `field` multiplied, then divided, by `factor`. A whole
    number (`ally_hand`) stays whole and at least 1, and a step that rounds
    back to the same value is left out."""
    value = getattr(weights, field)
    steps: list[float] = [value * factor, value / factor]
    if isinstance(value, int):
        steps = [max(1, round(step)) for step in steps]
    return [
        replace(weights, **{field: step})
        for step in dict.fromkeys(steps)
        if step != value
    ]


def describe(weights: Weights, start: Weights = WEIGHTS) -> str:
    """The weights that differ from `start`, as `name=value` pairs."""
    changed = [
        f"{name}={value:g}"
        for name, value in asdict(weights).items()
        if value != getattr(start, name)
    ]
    return ", ".join(changed) or "unchanged"


def search(
    play: Callable[[Weights], Outcomes],
    fields: Sequence[str] = FIELDS,
    start: Weights = WEIGHTS,
    factor: float = FACTOR,
    z: float = Z,
    passes: int = 1,
    log: Callable[[str], None] = print,
) -> Weights:
    """Coordinate search from `start` over `fields`, in order, for up to
    `passes` passes (stopping after a pass that changes nothing): the
    weights it ends with. `play` plays a set of weights on the tuning seeds;
    each step is logged."""
    incumbent, outcomes = start, play(start)
    games = len(outcomes)
    log(f"start: {sum(outcomes)}/{games} wins")
    for number in range(1, passes + 1):
        log(f"pass {number}")
        changed = False
        for field in fields:
            best: tuple[Comparison, Weights, Outcomes] | None = None
            for candidate in neighbours(incumbent, field, factor):
                tried = play(candidate)
                comparison = compare(tried, outcomes)
                log(
                    f"  {field} {getattr(incumbent, field):g} ->"
                    f" {getattr(candidate, field):g}: {sum(tried)}/{games} wins,"
                    f" {comparison.gain:+d} on {comparison.discordant} discordant"
                    f" (needs > {z * comparison.noise:.1f})"
                )
                if comparison.improves(z) and (
                    best is None or comparison.gain > best[0].gain
                ):
                    best = comparison, candidate, tried
            if best is not None:
                _, incumbent, outcomes = best
                changed = True
                log(
                    f"  keep {field} = {getattr(incumbent, field):g}:"
                    f" {sum(outcomes)}/{games} wins"
                )
        if not changed:
            break
    log(f"end: {sum(outcomes)}/{games} wins; {describe(incumbent, start)}")
    return incumbent


class TrialCache:
    """Trials already played, one JSON line each in `path`, keyed by the
    weights and the run's settings, so a stopped search resumes."""

    def __init__(self, path: Path, settings: dict[str, int]) -> None:
        self.path = path
        self.settings = settings
        self.trials: dict[str, Outcomes] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                entry = json.loads(line)
                self.trials[entry["key"]] = tuple(c == "1" for c in entry["wins"])

    def _key(self, weights: Weights) -> str:
        return json.dumps({**self.settings, **asdict(weights)}, sort_keys=True)

    def play(self, weights: Weights, play: Callable[[Weights], Outcomes]) -> Outcomes:
        """`play(weights)`, from the file if it was played before."""
        key = self._key(weights)
        if key not in self.trials:
            outcomes = play(weights)
            self.trials[key] = outcomes
            wins = "".join("1" if won else "0" for won in outcomes)
            with self.path.open("a") as file:
                file.write(json.dumps({"key": key, "wins": wins}) + "\n")
        return self.trials[key]


def decision_states(
    seeds: Iterable[int], budget: int, every: int = 1
) -> list[GameState]:
    """Every `every`-th state the cheat (default weights) decides in, over
    the seeded games it plays first against the heuristic."""
    states = []
    for seed in seeds:
        states += profiling.game_decisions(seed, MAX_TURNS, budget)
    return states[::every]


def _moves(states: Sequence[GameState], budget: int, weights: Weights) -> list[object]:
    return [cheat.decide(state, 2, budget, Linear(weights)).move for state in states]


def screen(
    states: Sequence[GameState],
    budget: int,
    fields: Sequence[str] = FIELDS,
    start: Weights = WEIGHTS,
    factor: float = FACTOR,
    jobs: int = 1,
) -> dict[str, float]:
    """Per field, the share of `states` where multiplying or dividing it by
    `factor` changes the cheat's move (2-turn horizon, `budget` nodes),
    most influential first."""
    candidates = [(f, w) for f in fields for w in neighbours(start, f, factor)]
    everything = [start, *(w for _, w in candidates)]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        moves = list(pool.map(partial(_moves, states, budget), everything))
    base = moves[0]
    changed = {field: [False] * len(states) for field in fields}
    for (field, _), tried in zip(candidates, moves[1:], strict=True):
        for i, (move, other) in enumerate(zip(base, tried, strict=True)):
            changed[field][i] |= move != other
    shares = {f: sum(c) / len(states) if states else 0.0 for f, c in changed.items()}
    return dict(sorted(shares.items(), key=lambda item: -item[1]))
