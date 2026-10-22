"""How it chooses, the first test: does examining more change the champion's
choice where its budget runs out (docs/star-realms.md, How it chooses:
lookahead adds a little, and tactics within the turn are calculated)?

At each plan the champion makes, a plan whose budget ran out before every
world listed every end of the turn (`planner.Plan.complete` false) is made
again from the same worlds with `factor` times the budget. Its line
*changes* when it differs from the plan's, compared up to the order of its
plays (`sampled_worlds.line_key`), and the chosen end's judgement *rises*
by the difference in the two plans' values. The same worlds are planned
both times, so a change is the examination's alone, not sampling's.
"""

import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from operator import itemgetter

from minimax_games.games.star_realms.bench import table
from minimax_games.games.star_realms.evaluate import LINEAR, Evaluator
from minimax_games.games.star_realms.match import MatchTooLong, play_match
from minimax_games.games.star_realms.planner import (
    BUDGET,
    WORLDS,
    Backend,
    Plan,
    PlannerBot,
    plan,
)
from minimax_games.games.star_realms.rollout import SPAN
from minimax_games.games.star_realms.sampled_worlds import MAX_TURNS, champion, line_key
from minimax_games.games.star_realms.state import GameState
from minimax_games.games.star_realms.view import View, sample_world

FACTOR = 4
"""The default multiple of the budget a budget-bound plan is made again with."""


@dataclass(frozen=True, slots=True)
class Examined:
    """One plan, and, if its budget ran out, the same worlds planned again
    with more."""

    base: Plan
    more: Plan | None
    """The plan with `factor` times the budget, or None if the base was
    complete."""
    seconds: float
    """The base plan's time."""
    more_seconds: float | None
    changed: bool | None
    """Whether the line with more budget differs from the base's, up to the
    order of its plays (None if the base was complete)."""

    @property
    def rise(self) -> float | None:
        """How much the chosen line's value rises with more budget, in the
        evaluator's units (None if the base was complete or either plan
        found no end)."""
        if self.more is None or self.more.value is None or self.base.value is None:
            return None
        return self.more.value - self.base.value


class Probe(PlannerBot):
    """The planner (`worlds` worlds, `budget` states per plan), which at each
    plan whose budget runs out also plans the same worlds with `factor`
    times the budget, and records both, with the turn, in `examined`. It
    plays the base plan, and samples its worlds as a plain planner does, so
    its games are a plain planner's from the same seed."""

    def __init__(
        self,
        seed: int,
        worlds: int = WORLDS,
        budget: int = BUDGET,
        factor: int = FACTOR,
        evaluator: Evaluator = LINEAR,
        backend: Backend = "python",
    ) -> None:
        super().__init__(seed, worlds, budget, evaluator, backend)
        self.factor = factor
        self.examined: list[tuple[int, Examined]] = []

    def _timed(self, worlds: Sequence[GameState], budget: int) -> tuple[Plan, float]:
        start = time.perf_counter()
        made = plan(worlds, budget, self.evaluator, self.backend)
        return made, time.perf_counter() - start

    def _plan(self, view: View) -> tuple[GameState, Plan]:
        worlds = [sample_world(view, self._rng) for _ in range(self.worlds)]
        base, seconds = self._timed(worlds, self.budget)
        more, more_seconds, changed = None, None, None
        if not base.complete:
            more, more_seconds = self._timed(worlds, self.factor * self.budget)
            changed = line_key(worlds[0], more.line) != line_key(worlds[0], base.line)
        found = Examined(base, more, seconds, more_seconds, changed)
        self.examined.append((view.turn, found))
        return worlds[0], base


def chance(value: float) -> float:
    """A learned value's score (`value.ValueEvaluator`) as the judged chance
    of winning, 0 to 1: a plan's value is a mean over worlds, so one that
    wins in any world is past `SPAN` and counts as a sure win."""
    return min(1.0, max(0.0, (value / SPAN + 1) / 2))


@dataclass(frozen=True, slots=True)
class Record:
    """What one plan came to."""

    seed: int
    """The game's seed."""
    turn: int
    bound: bool
    """Whether its budget ran out (the plan wasn't complete)."""
    changed: bool | None
    """Whether more budget changed the line (None if not bound)."""
    rise: float | None
    """How much the chosen end's judgement rose with more budget: in judged
    chance (`chance`) for a learned value, in the evaluator's units for the
    linear one (None if not bound, or no end was found)."""
    more_complete: bool | None
    """Whether the plan with more budget was complete (None if not bound)."""
    seconds: float
    """The plan's time: measured, so it varies by run."""
    more_seconds: float | None
    """The time with more budget (None if not bound)."""


def _records(
    seed: int,
    worlds: int,
    budget: int,
    factor: int,
    model: str | None,
    max_turns: int,
) -> list[Record]:
    """The record of every plan both players make in game `seed`, each
    player a `Probe` of `model`, in the order they were made."""
    evaluator, backend = champion(model)
    probes = (
        Probe(seed, worlds, budget, factor, evaluator, backend),
        Probe(seed, worlds, budget, factor, evaluator, backend),
    )
    with suppress(MatchTooLong):
        play_match(probes, seed, max_turns=max_turns)
    found = sorted([*probes[0].examined, *probes[1].examined], key=itemgetter(0))

    def rise(e: Examined) -> float | None:
        if model is None or e.rise is None:
            return e.rise
        assert e.more is not None and e.more.value is not None
        assert e.base.value is not None
        return chance(e.more.value) - chance(e.base.value)

    return [
        Record(
            seed,
            turn,
            e.more is not None,
            e.changed,
            rise(e),
            None if e.more is None else e.more.complete,
            e.seconds,
            e.more_seconds,
        )
        for turn, e in found
    ]


def measure(
    games: int,
    base_seed: int = 0,
    budget: int = BUDGET,
    factor: int = FACTOR,
    worlds: int = WORLDS,
    model: str | None = None,
    max_turns: int = MAX_TURNS,
    jobs: int = 1,
) -> list[Record]:
    """Every plan's record in `games` games from seeds `base_seed`, ..., the
    champion (`model`, a committed learned value, on the Rust enumeration;
    the linear evaluation in Python if None) playing itself at `worlds`
    worlds and `budget` states per plan, both players probing (`Probe`) with
    `factor` times the budget, each game stopped after `max_turns` turns.
    Over `jobs` processes, game by game, with the same records."""
    run = partial(
        _records,
        worlds=worlds,
        budget=budget,
        factor=factor,
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


def _share(flags: Sequence[bool | None]) -> str:
    return f"{100 * sum(map(bool, flags)) / len(flags):.1f}%" if flags else "-"


def _mean(values: Sequence[float], scale: float) -> str:
    return f"{scale * sum(values) / len(values):+.2f}" if values else "-"


def _ms(seconds: Sequence[float | None]) -> str:
    known = [s for s in seconds if s is not None]
    return f"{1000 * sum(known) / len(known):.1f}" if known else "-"


def report(
    records: Sequence[Record],
    factor: int = FACTOR,
    band: int = BAND,
    points: bool = True,
) -> list[str]:
    """The measurement as plain-text lines. By turn number (both players'
    turns, as the game numbers them): the plans, how many of them were
    budget-bound and their share, and of those, the share whose line changes
    with `factor` times the budget, the mean rise in the chosen end's
    judgement (in points of judged chance if `points`, else in the
    evaluator's units), the mean rise among the changed lines, and the share
    examined in full with the larger budget. Then the mean time per plan:
    every plan, the budget-bound ones, and those again with more budget."""
    games = len({r.seed for r in records})
    bound = [r for r in records if r.bound]
    lines = [
        f"{len(records)} plans in {games} game{'s' if games != 1 else ''},"
        f" {len(bound)} budget-bound ({_share([r.bound for r in records])}),"
        f" each planned again from the same worlds with {factor}x the budget;"
        " rise in the chosen end's judgement"
        + (" in points of judged chance" if points else " in the evaluator's units")
    ]
    scale = 100.0 if points else 1.0

    def row(label: str, part: Sequence[Record]) -> tuple[str, ...]:
        held = [r for r in part if r.bound]
        rises = [r.rise for r in held if r.rise is not None]
        changed = [r.rise for r in held if r.changed and r.rise is not None]
        return (
            label,
            str(len(part)),
            str(len(held)),
            _share([r.bound for r in part]),
            _share([r.changed for r in held]),
            _mean(rises, scale),
            _mean(changed, scale),
            _share([r.more_complete for r in held]),
        )

    rows = [
        (
            "Turns",
            "Plans",
            "Bound",
            "Share",
            "Changed",
            "Rise",
            "If-changed",
            f"Full-at-{factor}x",
        ),
        row("All", records),
    ]
    last = max((r.turn for r in records), default=0)
    for first in range(1, last + 1, band):
        part = [r for r in records if first <= r.turn < first + band]
        if part:
            rows.append(row(f"{first}-{first + band - 1}", part))
    lines += table(rows)
    lines.append("")
    lines += table(
        [
            ("", "All plans", "Bound", f"Bound at {factor}x"),
            (
                "ms per plan",
                _ms([r.seconds for r in records]),
                _ms([r.seconds for r in bound]),
                _ms([r.more_seconds for r in bound]),
            ),
        ]
    )
    return lines
