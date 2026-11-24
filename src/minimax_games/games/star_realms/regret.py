"""Regret by kind of decision (docs/star-realms.md, Learned values: regret by
kind of decision, #229): on the champion's own turns, how much win chance
its choice loses against the best of its other candidate turns, rolled out
to the game's end with the champion playing both sides.

A decision is the start of one of the champion's turns, a true state with
the champion to move, sampled from its games against itself (`decisions`).
Its options are the planner's candidates (a line of decisions to the turn's
end or its first chance event, after which the champion plays on), one per
distinct turn end (`shortlist`): the one the champion chooses, the next
best by its judgement within a margin of it, and a few drawn from further
down its ranking by judged gap (`Strata`, `widened`, #232). Each option is
rolled out from worlds sampled from the mover's view (`roll`; the Rust
engine plays them), every option on the same seeds, so on the same worlds
and, as far as the options leave it the same, the same future randomness
(keyed shuffles, `Settings.keyed`, and optionally keyed planner worlds,
`Settings.keyed_worlds`). Sequential halving narrows the options to the
best, which is re-estimated against the chosen on fresh rollouts
(`measure`); the regret is the difference. `Run.loss` turns the sampled
turns into win chance lost per game, and `report` breaks it down by phase,
by how decided the position was, by the judged gap of the best option and
by the kind of sub-choice that differs (`differ`, `group`). `pairing`
measures the cost of a rollout and how much shared randomness narrows a
paired difference, before a run is sized.
"""

import csv
import itertools
import math
import random
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from minimax_games._native import star_realms as native
from minimax_games.games.star_realms.bench import table
from minimax_games.games.star_realms.gap import PHASES, phase
from minimax_games.games.star_realms.planner import BUDGET, WORLDS
from minimax_games.games.star_realms.selfplay import Planner, native_model, play
from minimax_games.games.star_realms.value import Model

type Line = Sequence[tuple[Any, ...]]
"""An option: decisions as the Rust engine's action tuples."""

MAX_TURNS = 300
"""A rollout past it stops unfinished, and counts as half a win."""

SEEDS = 24_300_000
"""The run's first game seed: clear of every earlier range (#229's run used
24,100,000-24,100,049 and its measurements 24,000,000-24,050,099; #232's
measurements before its run use 24,200,000 up)."""

ROLLOUT_WORLDS = WORLDS
"""The planner's worlds in the run's rollouts: the champion's own 4."""

ROLLOUT_BUDGET = BUDGET
"""The planner's states per plan in the run's rollouts: the champion's own
20,000 (#232). #229 rolled out at 5,000, which plays the champion 43.5%;
on a quiet machine on 4 threads the champion's own settings cost about
0.55 s a rollout, and 10,000 states would save only a third of that
(results, Milestone 10: regret beyond the near ties)."""

CANDIDATES = 2
"""The chosen and its nearest options: the chosen and the next 1 within
`MARGIN` (#229 rolled out the next 2, and found no loss among them on 100
turns; #232 spends the third option's rollouts further down)."""

MARGIN = 0.1
"""The near options: judged within this of the chosen, the nearest
`CANDIDATES - 1` rolled out."""

BANDS_BEYOND = (0.3,)
"""The edges of judged gap past `MARGIN`: options are drawn from 0.1-0.3
and beyond 0.3 (`Strata`)."""

PER_BAND = 1
"""Options drawn at random from each band past `MARGIN`."""


@dataclass(frozen=True, slots=True)
class Settings:
    """The champion as it plays its games and chooses at a decision (the
    planner at `worlds` worlds and `budget` states per plan), and how
    rollouts are played: the same planner on both sides at `rollout_worlds`
    and `rollout_budget` (the champion's own settings if None), keyed
    shuffles or the engine's, the planners' sampled worlds keyed by the
    rollout's seed or drawn from one stream each (`keyed_worlds`), on
    `threads` threads, stopping unfinished past `max_turns`."""

    worlds: int = WORLDS
    budget: int = BUDGET
    rollout_worlds: int | None = None
    rollout_budget: int | None = None
    keyed: bool = True
    keyed_worlds: bool = False
    threads: int = 1
    max_turns: int = MAX_TURNS


@dataclass(frozen=True, slots=True, eq=False)
class Rollouts:
    """Each option's rollouts: `won[i, j]` is the mover's result from option
    `i` with seed `j` (1 won, 0 lost, 0.5 unfinished), `turns[i, j]` the
    turn the game stopped on."""

    won: np.ndarray
    turns: np.ndarray


@dataclass(frozen=True, slots=True)
class Option:
    """A candidate turn: its `line` and its `judged` chance, the mean over
    the planner's worlds of its best end's judged chance for the mover (1
    in a world where it wins, so 1 only if it wins in every world)."""

    line: Line
    judged: float


def shortlist(
    model: Model, state: dict[str, Any], settings: Settings, seed: int
) -> list[Option]:
    """The distinct turns the planner with `model` (`settings`' worlds and
    budget, its worlds sampled from `seed` as its bot samples them) compares
    at `state`, best first by its ranking (a win in any world first): the
    first is the turn it plays."""
    found = native.shortlist(
        native_model(model), state, settings.worlds, settings.budget, seed
    )
    return [
        Option(line=tuple(tuple(a) for a in line), judged=chance)
        for line, _, chance in found
    ]


KINDS: dict[str, str] = {
    "Acquire": "buy",
    "AcquireOnTop": "buy",
    "AcquireFreeShip": "buy",
    "ScrapTradeRowCard": "Trade Row scrap",
    "AttackPlayer": "attack",
    "AttackBase": "attack",
    "DestroyTarget": "attack",
    "Choose": "or choice",
}
"""The kind of sub-choice each kind of action makes; every other action
(playing, scrapping from hand or discard pile, ally and base abilities,
discards, copies, recycling) is "other"."""


def _kind(action: tuple[Any, ...]) -> str:
    return KINDS.get(action[0], "other")


def differ(a: Line, b: Line) -> frozenset[str]:
    """The kinds of sub-choice in which two turns differ: those whose
    decisions, as a multiset, aren't the same in both. Order within a turn
    doesn't count, nor does `EndTurn`."""
    kinds: set[str] = set()
    for kind in {*map(_kind, a), *map(_kind, b)}:
        mine = Counter(x for x in a if _kind(x) == kind and x[0] != "EndTurn")
        theirs = Counter(x for x in b if _kind(x) == kind and x[0] != "EndTurn")
        if mine != theirs:
            kinds.add(kind)
    return frozenset(kinds)


def group(kinds: frozenset[str]) -> str:
    """A turn's group by the kinds it differs in: the one kind, "mixed"
    for several, "order only" for none."""
    if not kinds:
        return "order only"
    if len(kinds) > 1:
        return "mixed"
    (kind,) = kinds
    return kind


@dataclass(frozen=True, slots=True)
class Budget:
    """Rollouts per decision: sequential halving gives every option left
    `per_round` rollouts a round, on seeds shared by them all, and keeps the
    better half (by each option's mean over all its rollouts so far) until
    one is left; then that one and the chosen are re-estimated on `fresh`
    rollouts each, on seeds of their own, also shared."""

    per_round: int = 8
    fresh: int = 16


class Roll(Protocol):
    def __call__(
        self,
        model: Model,
        state: dict[str, Any],
        lines: Sequence[Line],
        seeds: Sequence[int],
        settings: Settings,
    ) -> Rollouts: ...


@dataclass(frozen=True, slots=True, eq=False)
class Regret:
    """One decision's regret: its `options` (the chosen first), the `best`
    by sequential halving (an index into them) and every option's rollouts
    there (`screened`: each option's results, a list per option, as long
    as the rounds it lasted), and the best's and the chosen's results on
    fresh rollouts (`fresh`, rows best then chosen; empty if the chosen came
    out best). `regret` is the mean of the fresh paired differences, best
    less chosen, 0 if the chosen came out best."""

    options: Sequence[Option]
    best: int
    screened: list[np.ndarray]
    fresh: np.ndarray

    @property
    def regret(self) -> float:
        if self.best == 0:
            return 0.0
        return float(np.mean(self.fresh[0] - self.fresh[1]))


SEEDS_PER_DECISION = 1 << 20
"""Each decision's rollouts take seeds from `seed * SEEDS_PER_DECISION`: the
screen's from there up, the fresh ones from half way."""


def measure(
    model: Model,
    state: dict[str, Any],
    options: Sequence[Option],
    settings: Settings,
    budget: Budget,
    seed: int,
    roll: Roll | None = None,
) -> Regret:
    """The regret of the first of `options` (the chosen) at `state`:
    sequential halving among them all, then the best and the chosen
    re-estimated on fresh rollouts, every rollout's seed taken from
    `seed`'s range (`SEEDS_PER_DECISION`)."""
    roll = roll or _roll
    base = seed * SEEDS_PER_DECISION
    left = list(range(len(options)))
    results: list[list[float]] = [[] for _ in options]
    used = 0
    while len(left) > 1:
        seeds = range(base + used, base + used + budget.per_round)
        used += budget.per_round
        found = roll(model, state, [options[i].line for i in left], seeds, settings)
        for row, i in enumerate(left):
            results[i].extend(found.won[row].tolist())
        # stable: ties keep the judgement's order
        left.sort(key=lambda i: -float(np.mean(results[i])))
        left = sorted(left[: (len(left) + 1) // 2])
    best = left[0]
    screened = [np.array(r) for r in results]
    if best == 0:
        return Regret(options, best, screened, np.empty((2, 0)))
    half = base + SEEDS_PER_DECISION // 2
    fresh = roll(
        model,
        state,
        [options[best].line, options[0].line],
        range(half, half + budget.fresh),
        settings,
    )
    return Regret(options, best, screened, fresh.won)


@dataclass(frozen=True, slots=True, eq=False)
class Decision:
    """The start of a turn in one of the champion's games: the game (an
    index) and its `seed`, the game's turn number, the `seat` to move, and
    the true `state` at its first decision (after what plays itself)."""

    game: int
    seed: int
    turn: int
    seat: int
    state: dict[str, Any]


def decisions(
    model: Model,
    seeds: Sequence[int],
    settings: Settings,
    per_game: int,
    seed: int,
) -> tuple[list[Decision], dict[int, int]]:
    """The champion's games against itself on `seeds` (the planner with
    `model` at `settings`' worlds and budget on both sides, in one Rust
    batch), and `per_game` turns of each finished game drawn at random
    (`random.Random(seed)`; every turn if it has fewer), each as the state
    at its first decision, replayed in the Rust engine: the decisions, in
    game and turn order, and each finished game's turns."""
    player = Planner(model, settings.worlds, settings.budget)
    played = play(
        [(s, player, player) for s in seeds],
        threads=settings.threads,
        max_turns=settings.max_turns,
        record=True,
    )
    assert played.actions is not None
    rng = random.Random(seed)
    found: list[Decision] = []
    game_turns: dict[int, int] = {}
    for game, (s, actions, winner) in enumerate(
        zip(seeds, played.actions, played.winners, strict=True)
    ):
        if winner is None:
            continue
        starts: dict[int, dict[str, Any]] = {}
        replay = native.Game(s)
        for action in actions:
            state = replay.state()
            starts.setdefault(state["turn"], state)
            replay.advance(action)
        game_turns[game] = len(starts)
        chosen = sorted(rng.sample(sorted(starts), min(per_game, len(starts))))
        found += [
            Decision(game, s, turn, starts[turn]["current"], starts[turn])
            for turn in chosen
        ]
    return found, game_turns


@dataclass(frozen=True, slots=True)
class Strata:
    """Which options are rolled out at a decision, by their judged gap (the
    chosen option's judged chance less theirs): the `near` nearest within
    `edges[0]` of the chosen, then `per_band` drawn at random from each band
    between consecutive edges and beyond the last."""

    near: int = CANDIDATES - 1
    edges: tuple[float, ...] = (MARGIN, *BANDS_BEYOND)
    per_band: int = PER_BAND


def widened(
    options: Sequence[Option], strata: Strata, rng: random.Random
) -> list[Option]:
    """The options worth rolling out at a decision (`Strata`), in the
    judgement's order, the chosen first: so a turn the judgement ranks well
    below its choice can still reach the rollouts. Only the chosen if it
    wins (judged 1)."""
    chosen, *rest = options
    if chosen.judged >= 1.0:
        return [chosen]
    gaps = [chosen.judged - o.judged for o in rest]
    picked = [i for i, g in enumerate(gaps) if g <= strata.edges[0]][: strata.near]
    for low, high in itertools.pairwise((*strata.edges, math.inf)):
        band = [i for i, g in enumerate(gaps) if low < g <= high]
        picked += rng.sample(band, min(strata.per_band, len(band)))
    return [chosen, *(rest[i] for i in sorted(picked))]


@dataclass(frozen=True, slots=True)
class Turn:
    """One sampled turn of the champion's: its game (an index), the game's
    turn number and the seat to move, the chosen option's `judged` chance,
    how many `options` were rolled out (1 if the turn was left out, having
    one sensible end), its `regret` (0 if left out) and, if the best
    wasn't the chosen, the `group` of sub-choices they differ in and the
    `detail` of the decisions that differ and its judged `gap` (the
    chosen's judged chance less the best's); and the `rollouts` it took."""

    game: int
    turn: int
    seat: int
    judged: float
    options: int
    regret: float
    group: str | None
    rollouts: int = 0
    detail: str = ""
    gap: float | None = None


def _name(action: tuple[Any, ...]) -> str:
    return " ".join(str(x) for x in action)


def _detail(chosen: Line, best: Line) -> str:
    """The decisions in one of two turns and not the other (as multisets,
    `EndTurn` left out): "chosen: ...; best: ..."."""
    a = Counter(x for x in chosen if x[0] != "EndTurn")
    b = Counter(x for x in best if x[0] != "EndTurn")
    only = [
        ", ".join(_name(x) for x in (p - q).elements()) for p, q in ((a, b), (b, a))
    ]
    return f"chosen: {only[0] or '-'}; best: {only[1] or '-'}"


@dataclass(frozen=True, slots=True, eq=False)
class Run:
    """The sampled turns, and every game's number of turns (both players'
    counted), which the sample stands for."""

    turns: Sequence[Turn]
    game_turns: dict[int, int]

    def loss(self, where: Callable[[Turn], bool] | None = None) -> tuple[float, float]:
        """Win chance lost per game by one player, on the turns `where`
        picks (all by default), and its standard error counted by game.
        Each sampled turn stands for its game's turns over the turns
        sampled from it, and the two players' losses are halved into one
        player's, so the parts of a partition add up to the whole."""
        sampled = Counter(t.game for t in self.turns)
        per_game = dict.fromkeys(self.game_turns, 0.0)
        for t in self.turns:
            if where is None or where(t):
                weight = self.game_turns[t.game] / sampled[t.game]
                per_game[t.game] += weight * t.regret / 2
        values = np.array(list(per_game.values()))
        se = (
            float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        )
        return float(values.mean()), se


MODES = ("independent", "engine shuffles", "keyed shuffles", "keyed worlds")
"""How two options' rollouts share randomness in `pairing`: not at all
(each its own seeds), the same seeds with the engine's shuffles, the same
seeds with keyed shuffles (`Settings.keyed`), or the same seeds with keyed
shuffles and the planners' worlds keyed too (`Settings.keyed_worlds`)."""


@dataclass(frozen=True, slots=True, eq=False)
class Pairing:
    """Two options rolled out on the same number of seeds in each of
    `MODES`: `won[mode]` the results (rows the two options), and the
    `seconds` all the rollouts took."""

    won: dict[str, np.ndarray]
    seconds: float
    rollouts: int

    def discordant(self, mode: str) -> float:
        """The share of seeds on which the two options' results differ: the
        variance of a paired difference of wins and losses."""
        a, b = self.won[mode]
        return float(np.mean((a - b) ** 2))


def pairing(
    model: Model,
    state: dict[str, Any],
    options: Sequence[Option],
    settings: Settings,
    seeds: Sequence[int],
) -> Pairing:
    """The first two of `options` rolled out on `seeds` in each of `MODES`
    (the independent mode's second option on seeds past them)."""
    lines = [options[0].line, options[1].line]
    engine = replace(settings, keyed=False, keyed_worlds=False)
    keyed = replace(settings, keyed=True, keyed_worlds=False)
    worlds = replace(settings, keyed=True, keyed_worlds=True)
    start = time.perf_counter()
    won = {
        "engine shuffles": roll(model, state, lines, seeds, engine).won,
        "keyed shuffles": roll(model, state, lines, seeds, keyed).won,
        "keyed worlds": roll(model, state, lines, seeds, worlds).won,
    }
    offset = max(seeds) + 1 - min(seeds)
    won["independent"] = np.stack(
        [
            won["engine shuffles"][0],
            roll(model, state, lines[1:], [s + offset for s in seeds], engine).won[0],
        ]
    )
    return Pairing(won, time.perf_counter() - start, 7 * len(seeds))


def pairing_report(pairs: Sequence[tuple[bool, Pairing]]) -> list[str]:
    """The measurements before a run, as plain-text lines: for decisions
    whose two options buy differently (`True`) or alike, and all, each
    mode's mean share of discordant seeds (the variance of a paired
    difference) and the keyed modes' ratios to the independent mode's; and
    the time per rollout."""
    keyed = ("keyed shuffles", "keyed worlds")
    rows = [("Decisions", "Count", *MODES, *(f"{m} / independent" for m in keyed))]
    for name, keep in (
        ("buy differently", lambda b: b),
        ("buy alike", lambda b: not b),
        ("all", lambda b: True),
    ):
        here = [p for b, p in pairs if keep(b)]
        if not here:
            rows.append((name, "0", *["-"] * (len(MODES) + len(keyed))))
            continue
        means = {m: float(np.mean([p.discordant(m) for p in here])) for m in MODES}
        ratios = [
            means[m] / means["independent"] if means["independent"] else math.nan
            for m in keyed
        ]
        rows.append(
            (
                name,
                str(len(here)),
                *[f"{means[m]:.3f}" for m in MODES],
                *[f"{r:.2f}" for r in ratios],
            )
        )
    seconds = sum(p.seconds for _, p in pairs)
    rollouts = sum(p.rollouts for _, p in pairs)
    buying = sum(b for b, _ in pairs)
    return [
        f"Options buy differently: {buying} of {len(pairs)} decisions.",
        "Share of seeds on which the two options' results differ (the variance of"
        " their paired difference):",
        *table(rows),
        f"{1000 * seconds / max(rollouts, 1):.0f} ms per rollout (wall, every thread"
        f" together), {rollouts} rollouts.",
    ]


def decision_seed(decision: Decision) -> int:
    """The decision's own seed: its planner's worlds, and (times
    `SEEDS_PER_DECISION`) its rollouts' range."""
    return decision.seed * 1000 + decision.turn


def read(
    model: Model,
    found: Sequence[Decision],
    settings: Settings,
    budget: Budget,
    strata: Strata,
    progress: Callable[[int, Turn], None] | None = None,
) -> list[Turn]:
    """Each decision's regret: the planner's candidates there (`shortlist`,
    from the decision's seed), those `strata` picks (`widened`, drawn from
    the decision's seed) measured (`measure`), a turn with one left out at
    no regret. `progress` is told each turn as it is read."""
    turns = []
    for i, d in enumerate(found):
        seed = decision_seed(d)
        options = widened(
            shortlist(model, d.state, settings, seed), strata, random.Random(seed)
        )
        judged = options[0].judged
        if len(options) == 1:
            turn = Turn(d.game, d.turn, d.seat, judged, 1, 0.0, None)
        else:
            found_regret = measure(model, d.state, options, settings, budget, seed)
            best = found_regret.best
            kinds = (
                None
                if best == 0
                else group(differ(options[0].line, options[best].line))
            )
            turn = Turn(
                d.game,
                d.turn,
                d.seat,
                judged,
                len(options),
                found_regret.regret,
                kinds,
                rollouts=sum(len(r) for r in found_regret.screened)
                + found_regret.fresh.size,
                detail=""
                if best == 0
                else _detail(options[0].line, options[best].line),
                gap=None if best == 0 else judged - options[best].judged,
            )
        turns.append(turn)
        if progress is not None:
            progress(i, turn)
    return turns


def write(run: Run, path: Path) -> None:
    """Each sampled turn of `run` as a CSV row, with its game's turns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        out = csv.writer(f)
        out.writerow(
            [
                "game",
                "game_turns",
                "turn",
                "seat",
                "judged",
                "options",
                "regret",
                "group",
                "rollouts",
                "gap",
                "detail",
            ]
        )
        for t in run.turns:
            out.writerow(
                [
                    t.game,
                    run.game_turns[t.game],
                    t.turn,
                    t.seat,
                    f"{t.judged:.6f}",
                    t.options,
                    f"{t.regret:.6f}",
                    t.group or "",
                    t.rollouts,
                    "" if t.gap is None else f"{t.gap:.6f}",
                    t.detail,
                ]
            )


GAPS = (MARGIN, *BANDS_BEYOND)
"""Bands of the best option's judged gap below the chosen, in the report."""

BANDS = (0.0, 0.2, 0.4, 0.6, 0.8, math.inf)
"""How decided a position is: bands of the chosen option's judged chance."""

GROUPS = (
    "buy",
    "Trade Row scrap",
    "attack",
    "or choice",
    "other",
    "order only",
    "mixed",
)
"""The groups of turns by which kinds of sub-choice the chosen and the best
differ in (`group`)."""


def _breakdown(
    run: Run, title: str, parts: Sequence[tuple[str, Callable[[Turn], bool]]]
) -> list[str]:
    """A table of `run`'s loss over `parts` of its turns: each part's
    sampled turns, those left out and those whose best wasn't the chosen,
    the mean regret of those, and the loss per game with its error and
    share of the whole."""
    total, _ = run.loss()
    rows = [
        (
            title,
            "Turns",
            "Left out",
            "Best not chosen",
            "Mean regret then",
            "Loss per game",
            "se",
            "Share of loss",
        )
    ]
    for name, where in parts:
        here = [t for t in run.turns if where(t)]
        changed = [t.regret for t in here if t.group is not None]
        loss, se = run.loss(where)
        rows.append(
            (
                name,
                str(len(here)),
                str(sum(t.options == 1 for t in here)),
                str(len(changed)),
                f"{np.mean(changed):+.3f}" if changed else "-",
                f"{loss:+.4f}",
                f"{se:.4f}",
                f"{100 * loss / total:.0f}%" if total else "-",
            )
        )
    return table(rows)


def report(run: Run) -> list[str]:
    """The run as plain-text lines: the win chance lost per game by one
    player, overall and by game phase, by how decided the position was (the
    chosen option's judged chance) and by the kind of sub-choice the chosen
    and the best differ in, each with its error counted by game."""
    turns = run.turns
    loss, se = run.loss()
    left = sum(t.options == 1 for t in turns)
    changed = sum(t.group is not None for t in turns)
    per_player = float(np.mean(list(run.game_turns.values()))) / 2
    lines = [
        f"{len(run.game_turns)} games, {len(turns)} turns sampled;"
        f" {per_player:.1f} turns per player per game.",
        f"Left out (one sensible end): {left}. Measured: {len(turns) - left};"
        f" the best wasn't the chosen on {changed};"
        f" {sum(t.rollouts for t in turns)} rollouts.",
        f"Win chance lost per game by one player: {loss:+.4f} (se {se:.4f},"
        f" 95% {loss - 1.96 * se:+.4f} to {loss + 1.96 * se:+.4f});"
        f" per turn {loss / per_player:+.5f}.",
        "",
    ]
    phases = [
        (phase(low, high), lambda t, low=low, high=high: low <= t.turn < high)
        for low, high in itertools.pairwise(PHASES)
    ]
    lines += ["By game phase (the game's turn, both players' counted):"]
    lines += [*_breakdown(run, "Phase", phases), ""]
    bands = [
        (
            f"{low:.1f}-{min(high, 1.0):.1f}",
            lambda t, low=low, high=high: low <= t.judged < high,
        )
        for low, high in itertools.pairwise(BANDS)
    ]
    lines += ["By judged chance (the chosen option's, for the player to move):"]
    lines += [*_breakdown(run, "Judged", bands), ""]
    edges = (0.0, *GAPS, math.inf)
    gaps = [("chosen best", lambda t: t.gap is None)] + [
        (
            f"{low:.1f}-{high:.1f}" if high < math.inf else f"{low:.1f}+",
            lambda t, low=low, high=high: (
                t.gap is not None and (low == 0 or low < t.gap) and t.gap <= high
            ),
        )
        for low, high in itertools.pairwise(edges)
    ]
    lines += ["By judged gap of the best option (the chosen's less the best's):"]
    lines += [*_breakdown(run, "Gap", gaps), ""]
    kinds = [(g, lambda t, g=g: t.group == g) for g in GROUPS]
    lines += ["By kind of sub-choice the chosen and the best differ in:"]
    lines += _breakdown(run, "Kind", kinds)
    return lines


def roll(
    model: Model,
    state: dict[str, Any],
    lines: Sequence[Line],
    seeds: Sequence[int],
    settings: Settings,
) -> Rollouts:
    """Every line of `lines` rolled out from `state` with every seed of
    `seeds`, the planner with `model` playing both sides (`Settings`)."""
    found = native.rollouts(
        native_model(model),
        state,
        [list(line) for line in lines],
        list(seeds),
        settings.worlds if settings.rollout_worlds is None else settings.rollout_worlds,
        settings.budget if settings.rollout_budget is None else settings.rollout_budget,
        settings.keyed,
        settings.threads,
        settings.max_turns,
        keyed_worlds=settings.keyed_worlds,
    )
    winner = found["winner"]
    mover = state["current"]
    won = np.where(winner < 0, 0.5, (winner == mover).astype(np.float64))
    return Rollouts(won=won, turns=found["turns"])


_roll = roll
"""`roll` under a name `measure`'s parameter doesn't hide."""
