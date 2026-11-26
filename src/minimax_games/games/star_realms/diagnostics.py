"""The diagnostics beside the exploitation gap (docs/star-realms.md, How
progress is measured), on the champion's games as `gap.measure` plays them:
the judgement's error on positions by how many cards each player has left
to draw before their next reshuffle, and on positions holding a card the
Trade Deck has one copy of (`gap.rare`), against those without; and the
share of the champion's turns whose outcomes weren't all examined within
its budget, by turn.
"""

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from minimax_games.games.star_realms.gap import (
    PHASES,
    Gap,
    Measurement,
    Positions,
    Turns,
    described,
    gap,
    phase,
)


@dataclass(frozen=True, slots=True)
class Share:
    """Of the turns in `[low, high)`, how many weren't examined in full."""

    low: float
    high: float
    turns: int
    not_full: int

    @property
    def share(self) -> float:
        return self.not_full / self.turns if self.turns else math.nan


def not_full(found: Turns, edges: tuple[float, ...]) -> list[Share]:
    """The share of `found`'s turns not examined in full, in each bin of
    turn between consecutive `edges`."""
    return [
        Share(
            low,
            high,
            int(np.sum(inside := (found.turn >= low) & (found.turn < high))),
            int(np.sum(inside & ~found.full)),
        )
        for low, high in itertools.pairwise(edges)
    ]


LEFT = (0, 5, 10, 15, math.inf)
"""The bins of cards left to draw before a reshuffle."""
NEAR = 5
"""Fewer cards left than this is **near** a reshuffle: the player's next
draw of a hand reshuffles their discard pile, so what they buy on their
coming turn is in the new deck at once; from here up it waits at least a
hand longer (**far**)."""


def _label(low: float, high: float) -> str:
    return f"{low:g}+" if math.isinf(high) else f"{low:g}-{high - 1:g}"


def _cell(found: Gap) -> str:
    if found.rows == 0:
        return f"{0:>6} {0:>5}"
    return (
        f"{found.rows:>6} {found.games:>5} {found.gap:+7.3f} ± {found.error:.3f}"
        f" {found.squared:7.3f} ± {found.squared_error:.3f}"
    )


_COLUMNS = "     n games     gap ± error squared ± error"
_WIDTH = len(_COLUMNS) + 2

type Select = Callable[[Positions], Gap]


def _table(groups: dict[str, Positions], rows: dict[str, Select]) -> list[str]:
    lines = [f"{'':12} " + "".join(f"{name:<{_WIDTH}}" for name in groups).rstrip()]
    lines.append(f"{'':12} " + "".join(f"{_COLUMNS:<{_WIDTH}}" for _ in groups))
    for label, select in rows.items():
        cells = "".join(f"{_cell(select(g)):<{_WIDTH}}" for g in groups.values())
        lines.append(f"{label:<12} {cells}".rstrip())
    return lines


def _where(group: Positions, inside: np.ndarray) -> Gap:
    return gap(group.judged[inside], group.won[inside], group.game[inside])


_UNDIAGNOSED = "the games were played without diagnosing (play(..., diagnose=True))"


def _left(group: Positions, column: int) -> np.ndarray:
    if group.left is None:
        raise ValueError(_UNDIAGNOSED)
    return group.left[:, column]


def _rare(group: Positions) -> np.ndarray:
    if group.rare is None:
        raise ValueError(_UNDIAGNOSED)
    return group.rare


def _by_left(column: int) -> dict[str, Select]:
    def within(low: float, high: float) -> Select:
        return lambda g: _where(
            g, (_left(g, column) >= low) & (_left(g, column) < high)
        )

    rows = {
        _label(low, high): within(low, high) for low, high in itertools.pairwise(LEFT)
    }
    rows[f"far {NEAR}+"] = within(NEAR, math.inf)
    return rows


@dataclass(frozen=True, slots=True)
class Difference:
    """One side's mean squared miss less another's, over the same games."""

    difference: float
    error: float
    """Its standard error, counting each game's positions together, on
    both sides at once: a game's positions share one result, and a game can
    hold positions on both sides."""


def difference(group: Positions, first: np.ndarray, second: np.ndarray) -> Difference:
    """The mean squared miss of `group`'s positions in `first` less that of
    those in `second` (disjoint masks).

    The error is cluster-robust by game: each position contributes its
    squared miss less its side's mean, over its side's count (negated for
    `second`), and the error is the square root of the sum over games of
    each game's contributions, summed, squared."""
    rows_first, rows_second = int(np.sum(first)), int(np.sum(second))
    if rows_first == 0 or rows_second == 0:
        return Difference(math.nan, math.nan)
    squares = (group.judged - group.won) ** 2
    mean_first = float(np.mean(squares[first]))
    mean_second = float(np.mean(squares[second]))
    each = np.zeros(len(squares))
    each[first] = (squares[first] - mean_first) / rows_first
    each[second] = -(squares[second] - mean_second) / rows_second
    _, at = np.unique(group.game, return_inverse=True)
    per_game = np.bincount(at, weights=each)
    return Difference(mean_first - mean_second, float(np.sqrt(np.sum(per_game**2))))


def _within_phase(group: Positions) -> list[str]:
    """Near against far (the player who just ended) and a rare card against
    none, phase by phase: the squared miss of each, with its count, then
    near less far and with less without, each with its error by game
    (`difference`), which the End state's reading rules compare. The phase
    confounds both: decks grow and rare cards are bought as a game goes on,
    while early positions are the hardest to call."""
    near = _left(group, 0) < NEAR
    held = _rare(group)
    subsets = {"near": near, "far": ~near, "with": held, "without": ~held}
    lines = [
        "(a, b) Within each game phase, chosen positions: n and squared miss",
        f"{'':12} " + " ".join(f"{name:>13}" for name in subsets),
    ]
    for low, high in itertools.pairwise(PHASES):
        in_phase = (group.turn >= low) & (group.turn < high)
        cells = []
        for inside in subsets.values():
            found = _where(group, in_phase & inside)
            squared = "-" if found.rows == 0 else f"{found.squared:.3f}"
            cells.append(f"{found.rows:>6} {squared:>6}")
        lines.append(f"{phase(low, high):<12} " + " ".join(cells))
    lines += [
        "Within each game phase, the differences in squared miss, ± error"
        " counted by game",
        f"{'':12} {'near - far':>16} {'with - without':>16}",
    ]
    for low, high in itertools.pairwise(PHASES):
        in_phase = (group.turn >= low) & (group.turn < high)
        cells = []
        for first, second in ((near, ~near), (held, ~held)):
            found = difference(group, in_phase & first, in_phase & second)
            cells.append(
                "-"
                if math.isnan(found.difference)
                else f"{found.difference:+.3f} ± {found.error:.3f}"
            )
        lines.append(f"{phase(low, high):<12} " + " ".join(f"{c:>16}" for c in cells))
    return lines


def report(measured: Measurement) -> str:
    """The three diagnostics, for the positions the champion chose and,
    beside them, those its opponents did, each bin with its positions and
    games so small ones show: (a) the judgement's error by each player's
    cards left before a reshuffle, (b) with a rare card held and without,
    (c) the share of the champion's turns not examined in full, by turn."""
    if measured.turns is None:
        raise ValueError(_UNDIAGNOSED)
    groups = {"Chosen": measured.split.chosen, "Held out": measured.split.held_out}
    lines = [
        *described(measured),
        "gap: mean judged chance less share won; squared: mean squared miss"
        " (Brier); errors counted by game.",
        "",
        f"(a) By cards left to draw before a reshuffle (near: 0-{NEAR - 1})",
    ]
    for side, column in (("the player who just ended", 0), ("the next player", 1)):
        lines += [f"By {side}", *_table(groups, _by_left(column))]
    lines += [
        "",
        "(b) With a card the Trade Deck has one copy of, held by either player",
        *_table(
            groups,
            {
                "with": lambda g: _where(g, _rare(g)),
                "without": lambda g: _where(g, ~_rare(g)),
            },
        ),
        "",
        *_within_phase(measured.split.chosen),
        "",
        "(c) The champion's turns not examined in full within the budget",
        f"{'':12} {'turns':>6} {'not':>6} {'share':>6}",
    ]
    turns = measured.turns
    shares = [*not_full(turns, PHASES), *not_full(turns, (-math.inf, math.inf))]
    for found in shares:
        label = "all" if math.isinf(found.low) else phase(found.low, found.high)
        share = "-" if found.turns == 0 else f"{found.share:.3f}"
        lines.append(f"{label:<12} {found.turns:>6} {found.not_full:>6} {share:>6}")
    return "\n".join(lines)
