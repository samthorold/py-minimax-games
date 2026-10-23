"""How it chooses, more sampled worlds settled: is the champion at 16
worlds stronger than itself at 4 (docs/star-realms.md, Milestones, 10.3)?

A sequential probability ratio test (`sprt.in_batches`): `planner:<model>`
at 16 worlds (80,000 states) against itself at 4 (20,000), in batches of
seeds, every seed played twice, once with each first, a whole batch in one
Rust batch; after each batch the record so far is tested, and the match
stops when a bound is crossed or at the cap. The test counts games (a
win or a loss each, unfinished games neither), the unit its hypotheses
are stated in; the seeds won both, split and lost both are reported beside
it, with the sign z.
"""

import itertools
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from minimax_games.games.star_realms.bench import table
from minimax_games.games.star_realms.reply import sign_z
from minimax_games.games.star_realms.selfplay import Planner, play
from minimax_games.games.star_realms.sprt import (
    Batch,
    Check,
    Hypothesis,
    Sprt,
    in_batches,
    wilson,
)
from minimax_games.games.star_realms.value import load

SEEDS = 19_900_000
"""The match's first seed: clear of every earlier range (the training,
held-out, gating and ladder ones all below 4,000,000, the Milestone 9
measurements' from 18,400,000 to 19,880,149)."""

BATCH = 50
"""Seeds per batch: 100 games, each seed in both seats."""

CAP = 1_200
"""The most games the match plays: 12 batches."""

MAX_TURNS = 300
"""A game past it stops unfinished, and counts in neither player's wins."""


@dataclass(frozen=True, slots=True)
class Tally:
    """Games of the player with more worlds (A) against the one with fewer,
    each seed twice, A first in the first of the two."""

    games: int
    wins: int
    """A's wins."""
    losses: int
    first: tuple[int, int]
    """A's wins and finished games moving first."""
    second: tuple[int, int]
    pairs: tuple[int, int, int]
    """The seeds whose two games both finished that A won both, one and
    neither of."""
    plans: tuple[int, int]
    """Each player's plans (A's first), in every game."""
    seconds: tuple[float, float]
    """Each player's seconds planning, in every game: measured, so it
    varies by run."""
    noted: tuple[int, int]
    """Each player's plans in the finished games, whose states are known."""
    states: tuple[int, int]
    """Each player's states visited over those plans."""
    turns: int
    """The games' turns, summed."""

    def __add__(self, other: "Tally") -> "Tally":
        """The two batches' games together."""
        a, b = self, other
        return Tally(
            games=a.games + b.games,
            wins=a.wins + b.wins,
            losses=a.losses + b.losses,
            first=_sum(a.first, b.first),
            second=_sum(a.second, b.second),
            pairs=(
                a.pairs[0] + b.pairs[0],
                a.pairs[1] + b.pairs[1],
                a.pairs[2] + b.pairs[2],
            ),
            plans=_sum(a.plans, b.plans),
            seconds=_sum(a.seconds, b.seconds),
            noted=_sum(a.noted, b.noted),
            states=_sum(a.states, b.states),
            turns=a.turns + b.turns,
        )

    @property
    def finished(self) -> int:
        return self.wins + self.losses

    def ms_per_plan(self, player: int) -> float:
        """Player `player`'s (0 for A) mean time per plan."""
        return 1000 * self.seconds[player] / max(self.plans[player], 1)

    def states_per_plan(self, player: int) -> float:
        """Player `player`'s (0 for A) mean states per plan, in the
        finished games."""
        return self.states[player] / max(self.noted[player], 1)


def play_batch(
    model: str,
    seeds: range,
    *,
    worlds: tuple[int, int] = (16, 4),
    budgets: tuple[int, int] = (80_000, 20_000),
    threads: int = 1,
    max_turns: int = MAX_TURNS,
) -> Tally:
    """`model`'s planner at `worlds[0]` worlds and `budgets[0]` states
    against itself at `worlds[1]` and `budgets[1]`, each of `seeds` played
    twice, once with each first, in one batch on `threads` threads."""
    value = load(model)
    a = Planner(value, worlds[0], budgets[0])
    b = Planner(value, worlds[1], budgets[1])
    games = [game for seed in seeds for game in ((seed, a, b), (seed, b, a))]
    played = play(games, threads=threads, max_turns=max_turns, replies=True)
    # each game: whether A won, None if unfinished
    won = [None if w is None else w == i % 2 for i, w in enumerate(played.winners)]
    pairs = [0, 0, 0]
    for x, y in zip(won[::2], won[1::2], strict=True):
        if x is not None and y is not None:
            pairs[2 - (x + y)] += 1

    def record(part: list[bool | None]) -> tuple[int, int]:
        return sum(w is True for w in part), sum(w is not None for w in part)

    # each game's seat of A, and of the other player
    seat_of = np.arange(len(games)) % 2
    seats = (seat_of, 1 - seat_of)
    rows = np.arange(len(games))
    notes = played.replies
    seconds = played.seconds
    assert notes is not None and seconds is not None
    mine = notes.seat == seat_of[notes.game]
    noted = (mine, ~mine)
    return Tally(
        games=len(games),
        wins=sum(w is True for w in won),
        losses=sum(w is False for w in won),
        first=record(won[::2]),
        second=record(won[1::2]),
        pairs=(pairs[0], pairs[1], pairs[2]),
        plans=_two(int(played.plans[rows, s].sum()) for s in seats),
        seconds=_two(float(seconds[rows, s].sum()) for s in seats),
        noted=_two(int(n.sum()) for n in noted),
        states=_two(int(notes.states[n].sum()) for n in noted),
        turns=sum(played.turns),
    )


def sequential(
    model: str,
    test: Sprt,
    *,
    base_seed: int = SEEDS,
    batch: int = BATCH,
    cap: int = CAP,
    worlds: tuple[int, int] = (16, 4),
    budgets: tuple[int, int] = (80_000, 20_000),
    threads: int = 1,
    max_turns: int = MAX_TURNS,
) -> Iterator[tuple[Tally, Check]]:
    """The SPRT match (`play_batch`'s players), `batch` seeds at a time
    from `base_seed` on: after each batch, the games so far and the test's
    check, the last once a bound is crossed or `cap` games are played."""
    tallies: list[Tally] = []

    def batches() -> Iterator[Batch]:
        for start in itertools.count(base_seed, batch):
            found = play_batch(
                model,
                range(start, start + batch),
                worlds=worlds,
                budgets=budgets,
                threads=threads,
                max_turns=max_turns,
            )
            tallies.append(tallies[-1] + found if tallies else found)
            yield Batch(found.wins, found.losses, found.games)

    for check in in_batches(test, batches(), cap):
        yield tallies[-1], check


def reading(decision: Hypothesis | None) -> str:
    """The rule set before the run (#210)."""
    if decision is Hypothesis.H1:
        return (
            "H1 accepted: more worlds is worth about 5 points or more; the belief"
            " holds only for the choice, and more worlds is the one use of extra"
            " budget measured to pay (changing the default is its own decision)"
        )
    if decision is Hypothesis.H0:
        return (
            "H0 accepted: more worlds is worth less than about 5 points; the"
            " belief holds, and 4 worlds stays"
        )
    return (
        "undecided at the cap: the gain is small, within the band between the"
        " hypotheses; read as H0 for the default, and 4 worlds stays"
    )


def progress(tally: Tally, check: Check, test: Sprt) -> str:
    """One line after a batch: A's record, win rate and interval, the
    log-likelihood ratio against its bounds, the seeds won both, split and
    lost both by A with their sign z, and each player's ms and states per
    plan (A's first)."""
    low, high = wilson(check.wins, check.wins + check.losses)
    both, split, neither = tally.pairs
    return (
        f"{check.games} games: {check.wins} wins, {check.losses} losses,"
        f" {_share(check.wins, check.wins + check.losses)}"
        f" ({100 * low:.1f}-{100 * high:.1f}%);"
        f" LLR {check.llr:+.2f} in [{test.lower:+.2f}, {test.upper:+.2f}];"
        f" pairs {both} / {split} / {neither} (sign z {sign_z(both, neither):.2f});"
        f" ms per plan {tally.ms_per_plan(0):.1f} / {tally.ms_per_plan(1):.1f},"
        f" states per plan {tally.states_per_plan(0):,.0f}"
        f" / {tally.states_per_plan(1):,.0f}"
    )


def report(tally: Tally, check: Check, test: Sprt, names: tuple[str, str]) -> list[str]:
    """The match at its end as plain-text lines: the test, each player's
    wins, win rate and interval, seats, ms and states per plan, the seeds
    won both, split and lost both by A with their sign z, and the reading."""
    finished = tally.finished
    rows = [
        (
            "Bot",
            "Wins",
            "Win%",
            "95% interval",
            "As 1st",
            "As 2nd",
            "ms per plan",
            "States per plan",
        )
    ]
    for player, name in enumerate(names):
        first, second = tally.first, tally.second
        if player:
            # the other player's seats are A's swapped, its wins A's losses
            first = (tally.second[1] - tally.second[0], tally.second[1])
            second = (tally.first[1] - tally.first[0], tally.first[1])
        wins = first[0] + second[0]
        low, high = wilson(wins, finished)
        rows.append(
            (
                name,
                f"{wins}/{finished}",
                _share(wins, finished),
                f"{100 * low:.1f}-{100 * high:.1f}%",
                f"{first[0]}/{first[1]}",
                f"{second[0]}/{second[1]}",
                f"{tally.ms_per_plan(player):.1f}",
                f"{tally.states_per_plan(player):,.0f}",
            )
        )
    both, split, neither = tally.pairs
    outcome = "undecided" if check.decision is None else check.decision.value
    return [
        f"SPRT: H0 {names[0]} wins {test.p0:.1%}, H1 {test.p1:.1%};"
        f" alpha {test.alpha}, beta {test.beta}: {outcome} after {check.games}"
        f" games (LLR {check.llr:+.2f}, bounds {test.lower:+.2f}"
        f" and {test.upper:+.2f})",
        "",
        *table(rows),
        "",
        f"Seeds won both / split / lost both by {names[0]}: {both} / {split} /"
        f" {neither} (sign z {sign_z(both, neither):.2f})",
        f"Unfinished games: {tally.games - finished};"
        f" mean {tally.turns / max(tally.games, 1):.1f} turns.",
        "",
        f"Reading: {reading(check.decision)}",
    ]


def _share(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "-"


def _two[T](values: Iterable[T]) -> tuple[T, T]:
    a, b = values
    return a, b


def _sum[N: (int, float)](x: tuple[N, N], y: tuple[N, N]) -> tuple[N, N]:
    return x[0] + y[0], x[1] + y[1]
