"""How it chooses, the third test: does the champion's plan change with more
sampled worlds (docs/star-realms.md, How it chooses: hidden cards are chance)?

At each plan, the planner's worlds are sampled once, as many as the largest
count needs, and the same position is planned from the first `count` of them
for each count, each at the same states per world (`per_world`, the budget
`per_world * count`), so the larger counts add worlds to the smaller ones'.
A plan's line *changes* at a count when it differs from the base count's
(the first). As a yardstick for that, the base count is also planned from
the next `base` worlds, disjoint from its own: how often a fresh sample of
the same size changes the line is the noise of sampling alone.

Two lines are the same when they reach the same position up to pile order
(`planner.seen` of the view before their last move) and end with the same
move: a line is planned in its first world, and the order of plays a world
lists first can depend on its hand order, which the view doesn't fix.
"""

import time
from collections.abc import Hashable, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from functools import cache, partial
from operator import itemgetter

from minimax_games.games.star_realms import rules, value
from minimax_games.games.star_realms.action import Action
from minimax_games.games.star_realms.bench import table
from minimax_games.games.star_realms.evaluate import LINEAR, Evaluator
from minimax_games.games.star_realms.match import MatchTooLong, play_match
from minimax_games.games.star_realms.planner import (
    Backend,
    Plan,
    PlannerBot,
    plan,
    seen,
)
from minimax_games.games.star_realms.state import GameState
from minimax_games.games.star_realms.view import View, observe, sample_world

COUNTS = (4, 8, 16)
"""The default counts of worlds compared: today's, then twice and four times."""

MAX_TURNS = 300
"""The default turn cap: a game past it stops, its plans still counted."""

PER_WORLD = 5_000
"""The default states per world: the planner's default budget, 20,000, over
its 4 worlds."""


def line_key(world: GameState, line: Sequence[Action]) -> Hashable:
    """`line`, planned in `world`, up to the order of its plays: the mover's
    view before its last move, up to pile order, and that move."""
    state = world
    for action in line[:-1]:
        state = rules.advance_legal(state, action)
    return seen(observe(state, world.current)), line[-1]


@dataclass(frozen=True, slots=True)
class Comparison:
    """One position planned from each count of worlds."""

    counts: tuple[int, ...]
    keys: dict[int, Hashable]
    """Each count's line, by `line_key`."""
    fresh: Hashable
    """The base count's line from the next, disjoint, worlds."""
    seconds: dict[int, float]
    """Each count's time to plan."""
    plans: dict[int, Plan]

    @property
    def base(self) -> int:
        return self.counts[0]

    def changed(self, count: int) -> bool:
        """Whether the line from `count` worlds differs from the base's."""
        return self.keys[count] != self.keys[self.base]

    @property
    def changed_fresh(self) -> bool:
        """Whether the base count's line from fresh worlds differs from it."""
        return self.fresh != self.keys[self.base]


def needed(counts: Sequence[int]) -> int:
    """The worlds `compare` takes for `counts`: the largest count, and at
    least twice the base, for the fresh sample."""
    return max(*counts, 2 * counts[0])


def compare(
    worlds: Sequence[GameState],
    counts: Sequence[int] = COUNTS,
    per_world: int = PER_WORLD,
    evaluator: Evaluator = LINEAR,
    backend: Backend = "python",
) -> Comparison:
    """The plans from the first `count` of `worlds` (sampled worlds of one
    view) for each of `counts`, the first being the base, at `per_world`
    states per world, and the base's from the next `counts[0]` worlds."""
    counts = tuple(counts)
    if len(worlds) < needed(counts):
        msg = f"comparing {counts} needs {needed(counts)} worlds, not {len(worlds)}"
        raise ValueError(msg)

    def planned(chosen: Sequence[GameState]) -> tuple[Plan, float]:
        start = time.perf_counter()
        made = plan(chosen, per_world * len(chosen), evaluator, backend)
        return made, time.perf_counter() - start

    plans: dict[int, Plan] = {}
    keys: dict[int, Hashable] = {}
    seconds: dict[int, float] = {}
    for count in counts:
        plans[count], seconds[count] = planned(worlds[:count])
        keys[count] = line_key(worlds[0], plans[count].line)
    base = counts[0]
    other = worlds[base : 2 * base]
    fresh = line_key(other[0], planned(other)[0].line)
    return Comparison(counts, keys, fresh, seconds, plans)


class Probe(PlannerBot):
    """The planner at the base count of `counts` worlds and `per_world`
    states per world, which at each plan also compares every count
    (`compare`) and records it, with the turn, in `compared`. It plays the
    base count's line: the planner's own game, though its rng is drawn on
    for more worlds per plan than a plain planner's, so its games aren't a
    plain planner's from the same seed."""

    def __init__(
        self,
        seed: int,
        counts: Sequence[int] = COUNTS,
        per_world: int = PER_WORLD,
        evaluator: Evaluator = LINEAR,
        backend: Backend = "python",
    ) -> None:
        self.counts = tuple(counts)
        base = self.counts[0]
        super().__init__(seed, base, per_world * base, evaluator, backend)
        self.per_world = per_world
        self.compared: list[tuple[int, Comparison]] = []

    def _plan(self, view: View) -> tuple[GameState, Plan]:
        worlds = [sample_world(view, self._rng) for _ in range(needed(self.counts))]
        found = compare(
            worlds, self.counts, self.per_world, self.evaluator, self.backend
        )
        self.compared.append((view.turn, found))
        return worlds[0], found.plans[found.base]


@dataclass(frozen=True, slots=True)
class Record:
    """What one plan's comparison came to."""

    seed: int
    """The game's seed."""
    turn: int
    changed: dict[int, bool]
    """For each count but the base, whether its line differs from the base's."""
    fresh: bool
    """Whether the base count's line from fresh worlds differs from it."""
    seconds: dict[int, float]
    """Each count's time to plan: measured, so it varies by run."""


@cache
def _evaluator(model: str | None) -> tuple[Evaluator, Backend]:
    """The committed model `model` with the Rust enumeration, loaded once a
    process, or the linear evaluation with the Python one if None."""
    if model is None:
        return LINEAR, "python"
    return value.ValueEvaluator(value.load(model)), "rust"


def _records(
    seed: int,
    counts: tuple[int, ...],
    per_world: int,
    model: str | None,
    max_turns: int,
) -> list[Record]:
    """The comparisons of every plan both players make in game `seed`, each
    player a `Probe` of `model`, in the order they were made."""
    evaluator, backend = _evaluator(model)
    probes = (
        Probe(seed, counts, per_world, evaluator, backend),
        Probe(seed, counts, per_world, evaluator, backend),
    )
    with suppress(MatchTooLong):
        play_match(probes, seed, max_turns=max_turns)
    # Each turn is one player's, so ordering by turn (stably) keeps the
    # order the plans were made in.
    found = sorted([*probes[0].compared, *probes[1].compared], key=itemgetter(0))
    return [
        Record(
            seed,
            turn,
            {count: c.changed(count) for count in counts[1:]},
            c.changed_fresh,
            dict(c.seconds),
        )
        for turn, c in found
    ]


def measure(
    games: int,
    base_seed: int = 0,
    counts: Sequence[int] = COUNTS,
    per_world: int = PER_WORLD,
    model: str | None = None,
    max_turns: int = MAX_TURNS,
    jobs: int = 1,
) -> list[Record]:
    """Every plan's comparison in `games` games from seeds `base_seed`, ...,
    the champion (`model`, a committed learned value, on the Rust
    enumeration; the linear evaluation in Python if None) playing itself,
    both players probing (`Probe`), each game stopped after `max_turns`
    turns. Over `jobs` processes, game by game, with the same records."""
    run = partial(
        _records,
        counts=tuple(counts),
        per_world=per_world,
        model=model,
        max_turns=max_turns,
    )
    seeds = range(base_seed, base_seed + games)
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as pool:
            return [r for found in pool.map(run, seeds) for r in found]
    return [r for seed in seeds for r in run(seed)]


BAND = 5
"""Turns per row of the report's table by turn number."""


def _share(flags: Sequence[bool]) -> str:
    return f"{100 * sum(flags) / len(flags):.1f}%" if flags else "-"


def report(
    records: Sequence[Record], counts: Sequence[int] = COUNTS, band: int = BAND
) -> list[str]:
    """The measurement as plain-text lines: the share of plans whose line
    changes, from fresh worlds of the base count and at each larger count,
    over all plans and by turn number (both players' turns counted, as the
    game numbers them), then the mean time per plan at each count."""
    base, *more = counts
    games = len({r.seed for r in records})
    lines = [
        f"{len(records)} plans in {games} game{'s' if games != 1 else ''}:"
        f" the share whose line differs from {base} worlds' line, from"
        f" {base} fresh worlds and from {', '.join(map(str, more))} worlds"
        f" including the {base}"
    ]

    def row(label: str, part: Sequence[Record]) -> tuple[str, ...]:
        return (
            label,
            str(len(part)),
            _share([r.fresh for r in part]),
            *(_share([r.changed[count] for r in part]) for count in more),
        )

    rows = [
        ("Turns", "Plans", f"{base}-fresh", *map(str, more)),
        row("All", records),
    ]
    last = max((r.turn for r in records), default=0)
    for first in range(1, last + 1, band):
        part = [r for r in records if first <= r.turn < first + band]
        if part:
            rows.append(row(f"{first}-{first + band - 1}", part))
    lines += table(rows)
    lines.append("")

    def ms(count: int) -> str:
        seconds = [r.seconds[count] for r in records]
        return f"{1000 * sum(seconds) / len(seconds):.1f}" if seconds else "-"

    lines += table(
        [
            ("Worlds", *map(str, counts)),
            ("ms per plan", *map(ms, counts)),
        ]
    )
    return lines
