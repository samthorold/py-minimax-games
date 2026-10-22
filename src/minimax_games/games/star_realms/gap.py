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

from minimax_games.games.star_realms import features
from minimax_games.games.star_realms.catalogue import CATALOGUE
from minimax_games.games.star_realms.iterate import HEURISTIC, mix
from minimax_games.games.star_realms.planner import BUDGET, WORLDS
from minimax_games.games.star_realms.selfplay import (
    Examined,
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
    squared: float = np.nan
    """The mean squared miss (judged less won, squared; the Brier score):
    the judgement's error whichever way it misses, which a gap near zero
    can hide."""
    squared_error: float = np.nan
    """Its standard error, counted by game as `error` is."""

    @property
    def gap(self) -> float:
        """How much the judgement exceeds the outcome."""
        return self.judged - self.won


def gap(judged: np.ndarray, won: np.ndarray, game: np.ndarray) -> Gap:
    """The gap over the positions judged `judged` (chances), `won` (1.0 if
    the player who had just ended won) in `game`.

    The error is cluster-robust by game: the square root of the sum over
    games of each game's summed misses (judged less won less the gap),
    squared, over the positions; the squared error's likewise, about the
    mean squared miss."""
    rows = len(judged)
    if rows == 0:
        return Gap(rows=0, games=0, judged=np.nan, won=np.nan, error=np.nan)
    judged_mean = float(np.mean(judged))
    won_mean = float(np.mean(won))
    games, at = np.unique(game, return_inverse=True)

    def error(misses: np.ndarray) -> float:
        per_game = np.bincount(at, weights=misses - np.mean(misses))
        return float(np.sqrt(np.sum(per_game**2)) / rows)

    squares = (judged - won) ** 2
    return Gap(
        rows=rows,
        games=len(games),
        judged=judged_mean,
        won=won_mean,
        error=error(judged - won),
        squared=float(np.mean(squares)),
        squared_error=error(squares),
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


RARE: frozenset[str] = frozenset(
    name for name, card in CATALOGUE.items() if card.faction and card.copies == 1
)
"""The cards the Trade Deck has one copy of."""

_HELD = [
    i
    for i, name in enumerate(features.NAMES)
    if name.startswith(("ended: ", "next: "))
    and name.split(": ", 1)[1].startswith(("pool ", "base "))
    and name.split(" ", 2)[2] in RARE
]
"""The columns of a row counting a rare card either player holds: in their
draw pool or in play as a base."""


def rare(x: np.ndarray) -> np.ndarray:
    """Whether each row of `x` (`features.NAMES`) is a position in which
    either player holds a rare card (`RARE`)."""
    return np.asarray(x)[:, _HELD].sum(axis=1) > 0


@dataclass(frozen=True, slots=True, eq=False)
class Positions:
    """Positions with the champion's judgement of each: the judged chance
    that the player who had just ended wins, whether they did, the game
    (its index in the batch) and the game's turn (1 for the first turn's
    end, counting both players' turns). From diagnosed games
    (`play(..., diagnose=True)`), also each position's cards left to draw
    before a reshuffle (`SelfPlay.left`: the player who had just ended,
    then the next) and whether either player holds a rare card (`rare`);
    `None` otherwise."""

    judged: np.ndarray
    won: np.ndarray
    game: np.ndarray
    turn: np.ndarray
    left: np.ndarray | None = None
    rare: np.ndarray | None = None

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
        diagnosed = played.left is not None
        return Positions(
            judged=judged,
            won=played.y[rows],
            game=played.game[rows],
            turn=played.position[rows] + 1,
            left=played.left[rows] if diagnosed else None,
            rare=rare(played.x[rows]) if diagnosed else None,
        )

    return Split(
        chosen=at(mine, played.judgement[mine]),
        held_out=at(
            theirs, champion.model.predict(played.x[theirs].astype(np.float64))
        ),
    )


@dataclass(frozen=True, slots=True, eq=False)
class Turns:
    """A player's turns, in game order: each one's game, turn (the game's,
    both players' counted) and whether it was examined in full, every plan
    made in it having examined every end of the turn in every world."""

    game: np.ndarray
    turn: np.ndarray
    full: np.ndarray


def turns(plans: Examined, games: Sequence[Pairing], player: Planner) -> Turns:
    """`player`'s turns in `games` (played as `plans` records), whether
    each was examined in full."""
    mine = np.array(
        [
            games[g][1 + s] is player
            for g, s in zip(plans.game.tolist(), plans.seat.tolist(), strict=True)
        ],
        dtype=bool,
    )
    keys = np.stack([plans.game[mine], plans.turn[mine]], axis=1)
    if len(keys) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return Turns(game=empty, turn=empty, full=np.zeros(0, dtype=bool))
    found, at = np.unique(keys, axis=0, return_inverse=True)
    incomplete = np.bincount(at.ravel(), weights=~plans.complete[mine])
    return Turns(game=found[:, 0], turn=found[:, 1], full=incomplete == 0)


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
    every: int = 1
    """Every how manyth game of the mix was played."""
    turns: Turns | None = None
    """The champion's turns, and whether each was examined in full."""

    @property
    def games(self) -> int:
        """The games played: every `every`th of the mix over `seeds`."""
        return len(range(0, len(self.seeds), self.every))


def measure(
    champion: str,
    pool: Sequence[str],
    seeds: range,
    threads: int = 1,
    worlds: int = WORLDS,
    budget: int = BUDGET,
    every: int = 1,
) -> Measurement:
    """Play `champion`'s games on `seeds` as a policy-iteration round mixes
    them (`iterate.mix`: half against itself, a quarter against the
    heuristic, a quarter against a champion drawn from `pool`), judging
    and diagnosing every end, and split the positions into those it chose
    and those its opponents did. With `every`, only every `every`th game
    of that mix is played (the 1st, then the `every + 1`th, ...): the same
    games, fewer of them, in the mix's proportions."""
    players: dict[str, Player] = {
        name: Planner(load(name), worlds, budget) for name in sorted({*pool, champion})
    }
    players[HEURISTIC] = HEURISTIC
    named = mix(champion, list(pool), seeds, random.Random(seeds.start))
    games = [(seed, players[a], players[b]) for seed, a, b in named[::every]]
    played = play(games, threads=threads, judge=True, diagnose=True)
    me = players[champion]
    assert isinstance(me, Planner)
    assert played.examined is not None
    return Measurement(
        champion,
        tuple(pool),
        seeds,
        worlds,
        budget,
        positions(played, games, me),
        every=every,
        turns=turns(played.examined, games, me),
    )


def _line(found: Gap) -> str:
    if found.rows == 0:
        return f"{0:>6}"
    return (
        f"{found.rows:>6} {found.judged:6.3f} {found.won:6.3f}"
        f" {found.gap:+7.3f} ± {found.error:.3f}"
    )


def phase(low: float, high: float) -> str:
    if math.isinf(high):
        return f"turns {low:g}+"
    return f"turns {low:g}-{high - 1:g}"


def _chance(low: float, high: float) -> str:
    return f"{low:.1f}-{high:.1f}"


def described(measured: Measurement) -> list[str]:
    """What games `measured` is of, and what its two groups of positions
    are: a report's opening lines."""
    seeds = measured.seeds
    span = f"seeds {seeds.start}-{seeds.stop - 1}"
    games = (
        f"{len(seeds)} games ({span})"
        if measured.every == 1
        else f"{measured.games} of {len(seeds)} games"
        f" (every {measured.every} of the mix on {span})"
    )
    return [
        f"{games}: {measured.champion} against itself, the heuristic and the"
        f" pool ({', '.join(measured.pool)}), as a policy-iteration round mixes"
        f" them; {measured.worlds} worlds, budget {measured.budget}.",
        "Chosen: the ends the champion played, judged as it played them."
        " Held out: the ends its opponents played, judged by the champion.",
    ]


def report(measured: Measurement) -> str:
    """The gap overall, the calibration table and the gap by game phase, for
    the positions the champion chose and, beside them, those it didn't."""
    groups = {"Chosen": measured.split.chosen, "Held out": measured.split.held_out}
    lines = [*described(measured), ""]
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
        ("By game phase", "turn", PHASES, phase),
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
