"""The exploitation gap (docs/star-realms.md, What it learns from; CONTEXT.md):
how much a player's judgement of the positions it chose exceeds how often it
went on to win from them.

A row is one position: the judged chance that the player who had just ended
their turn goes on to win, whether they did, and the game it is from.
"""

import itertools
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from minimax_games.games.star_realms.iterate import HEURISTIC, mix
from minimax_games.games.star_realms.planner import BUDGET, WORLDS
from minimax_games.games.star_realms.selfplay import (
    Pairing,
    Planner,
    Player,
    SelfPlay,
    play,
)
from minimax_games.games.star_realms.value import load


@dataclass(frozen=True, slots=True)
class Gap:
    """Judgement against outcome over some positions."""

    rows: int
    games: int
    """The games the positions come from."""
    judged: float
    """The mean judged chance of winning."""
    won: float
    """The share of positions whose player went on to win."""
    error: float
    """The gap's standard error, counting each game's positions together:
    they share the game's result, so they aren't independent draws."""

    @property
    def gap(self) -> float:
        """How much the judgement exceeds the outcome."""
        return self.judged - self.won


def gap(judged: np.ndarray, won: np.ndarray, game: np.ndarray) -> Gap:
    """The gap over the positions judged `judged` (chances), `won` (1.0 if
    the player who had just ended won) in `game`.

    The error is cluster-robust by game: the square root of the sum over
    games of each game's summed misses (judged less won less the gap),
    squared, over the positions."""
    rows = len(judged)
    if rows == 0:
        return Gap(rows=0, games=0, judged=np.nan, won=np.nan, error=np.nan)
    judged_mean = float(np.mean(judged))
    won_mean = float(np.mean(won))
    misses = judged - won - (judged_mean - won_mean)
    games, at = np.unique(game, return_inverse=True)
    per_game = np.bincount(at, weights=misses)
    return Gap(
        rows=rows,
        games=len(games),
        judged=judged_mean,
        won=won_mean,
        error=float(np.sqrt(np.sum(per_game**2)) / rows),
    )


@dataclass(frozen=True, slots=True)
class Bin:
    """The gap over the positions whose key is in `[low, high)` (the last
    bin includes `high`)."""

    low: float
    high: float
    gap: Gap


def breakdown(
    judged: np.ndarray,
    won: np.ndarray,
    game: np.ndarray,
    by: np.ndarray,
    edges: tuple[float, ...],
) -> list[Bin]:
    """The gap in each bin of `by` between consecutive `edges`; positions
    outside them are left out. By the judged chance itself, this is a
    calibration table; by the turn, the gap by game phase."""
    bins = []
    for i, (low, high) in enumerate(itertools.pairwise(edges)):
        last = i == len(edges) - 2
        inside = (by >= low) & ((by <= high) if last else (by < high))
        bins.append(Bin(low, high, gap(judged[inside], won[inside], game[inside])))
    return bins


@dataclass(frozen=True, slots=True, eq=False)
class Positions:
    """Positions with the champion's judgement of each: the judged chance
    that the player who had just ended wins, whether they did, the game
    (its index in the batch) and the game's turn (1 for the first turn's
    end, counting both players' turns)."""

    judged: np.ndarray
    won: np.ndarray
    game: np.ndarray
    turn: np.ndarray

    def gap(self) -> Gap:
        return gap(self.judged, self.won, self.game)

    def breakdown(self, by: np.ndarray, edges: tuple[float, ...]) -> list[Bin]:
        return breakdown(self.judged, self.won, self.game, by, edges)


@dataclass(frozen=True, slots=True, eq=False)
class Split:
    chosen: Positions
    """The ends the champion played, with the judgement it recorded as it
    played them: what its choices selected."""
    held_out: Positions
    """The ends its opponents played (never the champion), judged by the
    champion: positions its own choice didn't select."""


def positions(played: SelfPlay, games: Sequence[Pairing], champion: Planner) -> Split:
    """`played`'s positions (`games`, played with `judge`), split into those
    `champion` (the very player in `games`) chose and those its opponents
    did."""
    if played.judgement is None:
        msg = "the games were played without judging (play(..., judge=True))"
        raise ValueError(msg)
    mine = np.array(
        [
            games[g][1 + s] is champion
            for g, s in zip(played.game.tolist(), played.seat.tolist(), strict=True)
        ],
        dtype=bool,
    )
    theirs = ~mine

    def at(rows: np.ndarray, judged: np.ndarray) -> Positions:
        return Positions(
            judged=judged,
            won=played.y[rows],
            game=played.game[rows],
            turn=played.position[rows] + 1,
        )

    return Split(
        chosen=at(mine, played.judgement[mine]),
        held_out=at(
            theirs, champion.model.predict(played.x[theirs].astype(np.float64))
        ),
    )


CALIBRATION = tuple(i / 10 for i in range(11))
"""The calibration table's bins of judged chance: tenths."""
PHASES = (1, 7, 13, 19, 25, math.inf)
"""The game phases' bins of the game's turn (both players' turns counted):
about a quarter of a typical game each, the last open-ended."""
SEEDS = 18_400_000
"""The measurement's first seed: clear of every training, held-out, gating
and ladder range (`iterate.Settings`, all below 4,000,000) and of every
earlier experiment's (all below 10,000,000), so no champion was fitted to
these games."""


@dataclass(frozen=True, slots=True, eq=False)
class Measurement:
    champion: str
    pool: tuple[str, ...]
    seeds: range
    worlds: int
    budget: int
    split: Split


def measure(
    champion: str,
    pool: Sequence[str],
    seeds: range,
    threads: int = 1,
    worlds: int = WORLDS,
    budget: int = BUDGET,
) -> Measurement:
    """Play `champion`'s games on `seeds` as a policy-iteration round mixes
    them (`iterate.mix`: half against itself, a quarter against the
    heuristic, a quarter against a champion drawn from `pool`), judging
    every end, and split the positions into those it chose and those its
    opponents did."""
    players: dict[str, Player] = {
        name: Planner(load(name), worlds, budget) for name in sorted({*pool, champion})
    }
    players[HEURISTIC] = HEURISTIC
    named = mix(champion, list(pool), seeds, random.Random(seeds.start))
    games = [(seed, players[a], players[b]) for seed, a, b in named]
    played = play(games, threads=threads, judge=True)
    me = players[champion]
    assert isinstance(me, Planner)
    return Measurement(
        champion, tuple(pool), seeds, worlds, budget, positions(played, games, me)
    )


def _line(found: Gap) -> str:
    if found.rows == 0:
        return f"{0:>6}"
    return (
        f"{found.rows:>6} {found.judged:6.3f} {found.won:6.3f}"
        f" {found.gap:+7.3f} ± {found.error:.3f}"
    )


def _phase(low: float, high: float) -> str:
    if math.isinf(high):
        return f"turns {low:g}+"
    return f"turns {low:g}-{high - 1:g}"


def _chance(low: float, high: float) -> str:
    return f"{low:.1f}-{high:.1f}"


def report(measured: Measurement) -> str:
    """The gap overall, the calibration table and the gap by game phase, for
    the positions the champion chose and, beside them, those it didn't."""
    seeds = measured.seeds
    groups = {"Chosen": measured.split.chosen, "Held out": measured.split.held_out}
    lines = [
        f"{len(seeds)} games (seeds {seeds.start}-{seeds.stop - 1}):"
        f" {measured.champion} against itself, the heuristic and the pool"
        f" ({', '.join(measured.pool)}), as a policy-iteration round mixes"
        f" them; {measured.worlds} worlds, budget {measured.budget}.",
        "Chosen: the ends the champion played, judged as it played them."
        " Held out: the ends its opponents played, judged by the champion.",
        "",
    ]
    for name, group in groups.items():
        found = group.gap()
        lines.append(
            f"{name}: {found.rows} positions in {found.games} games:"
            f" judged {found.judged:.3f}, won {found.won:.3f},"
            f" gap {found.gap:+.3f} ± {found.error:.3f}"
        )
    columns = "     n judged    won     gap ± error"
    for title, key, edges, label in (
        ("Calibration, by judged chance", "judged", CALIBRATION, _chance),
        ("By game phase", "turn", PHASES, _phase),
    ):
        lines += ["", title, f"{'':12} {'Chosen':<38} Held out"]
        lines.append(f"{'':12} {columns:<38} {columns}")
        tables = [
            group.breakdown(getattr(group, key), edges) for group in groups.values()
        ]
        for bins in zip(*tables, strict=True):
            cells = " ".join(f"{_line(b.gap):<38}" for b in bins).rstrip()
            lines.append(f"{label(bins[0].low, bins[0].high):<12} {cells}")
    return "\n".join(lines)
