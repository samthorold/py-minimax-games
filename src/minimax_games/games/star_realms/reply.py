"""How it chooses, the first test's second half: does looking through the
opponent's reply change the champion's choice, and add strength at equal
cost (docs/star-realms.md, Learned values: looking through the reply)?

The paired match: the champion looking through the opponent's reply (the
Rust planner's reply option, `selfplay.Reply`; at its defaults a plan
costs at most 4 worlds' 20,000 states and 12 replies of 5,000, 80,000 in
all) against the same champion planning over 16 worlds at 5,000 states
each, the same 80,000. Every seed is played twice, once with each first,
all in one Rust batch, and every plan of the reply player records whether
the reply changed its choice (`selfplay.Replies`), so one match gives both
the strength and the changed choices.
"""

import itertools
import math
from dataclasses import dataclass

import numpy as np

from minimax_games.games.star_realms.bench import table
from minimax_games.games.star_realms.gap import PHASES, phase
from minimax_games.games.star_realms.selfplay import (
    Planner,
    Reply,
    SelfPlay,
    play,
)
from minimax_games.games.star_realms.sprt import wilson
from minimax_games.games.star_realms.value import load

SEEDS = 19_880_000
"""The match's first seed: clear of every earlier range (the training,
held-out, gating and ladder ones all below 4,000,000, the Milestone 9
measurements' from 18,400,000 to 19,879,999)."""

REPLY = Reply()
"""The design's reply: the best 3 candidates, 2 sub-worlds, 5,000 states."""

MAX_TURNS = 300
"""A game past it stops unfinished, and counts in neither player's wins."""

NOISE = 0.113
"""The share of plans whose line a fresh 4-world sample changes (results,
Milestone 9: more sampled worlds): the yardstick for changed choices."""


def sign_z(both: int, neither: int) -> float:
    """The sign test's z over the seeds whose two games one player won both
    of (`both`) or neither of (`neither`), 0 if there are none."""
    decided = both + neither
    return (both - neither) / math.sqrt(decided) if decided else 0.0


def reading(z: float, changed: float) -> str:
    """The design's rule: the belief holds if the reply player doesn't beat
    the rival (sign z below 2) and its choice changes on no more plans than
    the noise of a fresh sample (`NOISE`); it fails if the reply player wins
    with sign z of 2 or more; otherwise neither."""
    if z >= 2:
        return "fails: looking through the reply beats the same budget spent on worlds"
    if changed <= NOISE:
        return (
            "holds: no strength from the reply, and its changed choices are"
            " within the noise of a fresh sample"
        )
    return (
        "neither: no strength from the reply, but it changes more choices than"
        " a fresh sample does (the diagnostic moving without strength)"
    )


@dataclass(frozen=True, slots=True, eq=False)
class Measurement:
    """The paired match: game `2i` is seed `seeds[i]` with the reply player
    first, game `2i + 1` the same seed with it second."""

    model: str
    seeds: range
    worlds: int
    budget: int
    reply: Reply
    rival_worlds: int
    rival_budget: int
    played: SelfPlay

    @property
    def games(self) -> int:
        return 2 * len(self.seeds)

    @property
    def replier(self) -> np.ndarray:
        """Each game's seat of the reply player."""
        return np.arange(self.games) % 2

    def _won(self) -> list[bool | None]:
        """Each game: whether the reply player won, None if unfinished."""
        return [
            None if w is None else w == seat
            for w, seat in zip(self.played.winners, self.replier.tolist(), strict=True)
        ]

    @property
    def finished(self) -> int:
        return sum(w is not None for w in self.played.winners)

    @property
    def wins(self) -> int:
        """The reply player's wins."""
        return sum(won is True for won in self._won())

    def wins_as(self, first: bool) -> tuple[int, int]:
        """The reply player's wins and finished games moving first (or
        second)."""
        won = self._won()[0 if first else 1 :: 2]
        return sum(w is True for w in won), sum(w is not None for w in won)

    @property
    def pairs(self) -> tuple[int, int, int]:
        """The seeds whose two games both finished that the reply player
        won both, one and neither of."""
        won = self._won()
        counts = [0, 0, 0]
        for a, b in itertools.batched(won, 2, strict=True):
            if a is not None and b is not None:
                counts[2 - (a + b)] += 1
        return counts[0], counts[1], counts[2]

    @property
    def seeds_finished(self) -> int:
        return sum(self.pairs)

    @property
    def z(self) -> float:
        both, _, neither = self.pairs
        return sign_z(both, neither)

    def _seat(self, rival: bool) -> np.ndarray:
        """Each game's seat of the reply player, or of the rival."""
        return 1 - self.replier if rival else self.replier

    def ms_per_plan(self, rival: bool) -> float:
        """The mean time per plan of the reply player (or the rival), over
        every game: measured, so it varies by run."""
        assert self.played.seconds is not None
        rows = np.arange(self.games)
        seat = self._seat(rival)
        plans = self.played.plans[rows, seat].sum()
        return 1000 * float(self.played.seconds[rows, seat].sum()) / max(plans, 1)

    def _notes_of(self, rival: bool) -> np.ndarray:
        notes = self.played.replies
        assert notes is not None
        return notes.seat == self._seat(rival)[notes.game]

    @property
    def mine(self) -> np.ndarray:
        """Which of the finished games' plans are the reply player's."""
        return self._notes_of(rival=False)

    def states_per_plan(self, rival: bool) -> float:
        """The mean states per plan of the reply player (or the rival), in
        the finished games, its replies' included."""
        notes = self.played.replies
        assert notes is not None
        states = notes.states[self._notes_of(rival)]
        return float(states.mean()) if len(states) else 0.0

    @property
    def changed(self) -> float:
        """The share of the reply player's plans whose choice the reply
        changed (its pick wasn't the judgement's)."""
        notes = self.played.replies
        assert notes is not None
        ranks = notes.rank[self.mine]
        return float(np.mean(ranks > 1)) if len(ranks) else 0.0


def measure(
    model: str,
    seeds: range,
    *,
    worlds: int = 4,
    budget: int = 20_000,
    reply: Reply = REPLY,
    rival_worlds: int = 16,
    rival_budget: int = 80_000,
    threads: int = 1,
    max_turns: int = MAX_TURNS,
) -> Measurement:
    """`model`'s planner over `worlds` worlds at `budget` states, looking
    through the replies (`reply`), against the same model over
    `rival_worlds` worlds at `rival_budget` states, each of `seeds` played
    twice, once with each first, in one batch on `threads` threads."""
    value = load(model)
    replier = Planner(value, worlds, budget, reply)
    rival = Planner(value, rival_worlds, rival_budget)
    games = [
        game
        for seed in seeds
        for game in ((seed, replier, rival), (seed, rival, replier))
    ]
    played = play(games, threads=threads, max_turns=max_turns, replies=True)
    return Measurement(
        model, seeds, worlds, budget, reply, rival_worlds, rival_budget, played
    )


def _share(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "-"


def report(found: Measurement) -> list[str]:
    """The match as plain-text lines: the strength (each player's wins and
    interval, seats, time and states per plan; the seeds won both, split and
    lost both by the reply player and their sign z), then the changed
    choices (overall and by game phase: the share of the reply player's
    plans the reply changed, of those that looked through 2 or more
    candidates, and of those whose replies were all examined in full; the
    pick's rank), and the design's reading."""
    r = found.reply
    seeds = found.seeds
    names = (
        f"replies ({found.worlds} worlds, {found.budget:,} states,"
        f" {r.candidates} candidates x {r.worlds} sub-worlds x {r.budget:,})",
        f"{found.rival_worlds} worlds ({found.rival_budget:,} states)",
    )
    lines = [
        f"planner:{found.model} looking through the reply against"
        f" planner:{found.model} at {found.rival_worlds} worlds:"
        f" {len(seeds)} seeds in both seats ({seeds.start}-{seeds.stop - 1})",
        "",
    ]
    finished = found.finished
    rows = [
        (
            "Bot",
            "Wins",
            "Win%",
            "95% interval",
            "As 1st",
            "As 2nd",
            "ms per plan",
            "states per plan",
        )
    ]
    for rival, name in ((False, names[0]), (True, names[1])):
        wins = found.wins if not rival else finished - found.wins
        low, high = wilson(wins, finished)
        first = found.wins_as(first=not rival)
        second = found.wins_as(first=rival)
        if rival:
            first = (first[1] - first[0], first[1])
            second = (second[1] - second[0], second[1])
        rows.append(
            (
                name,
                f"{wins}/{finished}",
                _share(wins, finished),
                f"{100 * low:.1f}-{100 * high:.1f}%",
                f"{first[0]}/{first[1]}",
                f"{second[0]}/{second[1]}",
                f"{found.ms_per_plan(rival):.1f}",
                f"{found.states_per_plan(rival):,.0f}",
            )
        )
    lines += table(rows)
    both, split, neither = found.pairs
    unfinished = found.games - finished
    turns = np.mean(found.played.turns)
    lines += [
        "",
        f"Seeds won both / split / lost both by the reply player: {both} /"
        f" {split} / {neither} (sign z {found.z:.2f})",
        f"Unfinished games: {unfinished}; mean {turns:.1f} turns.",
        "",
    ]
    notes = found.played.replies
    assert notes is not None
    mine = found.mine
    looked = notes.looked >= 2
    rows = [
        (
            "Turns",
            "Plans",
            "Looked through",
            "Changed",
            "Changed of looked",
            "Replies examined in full",
        )
    ]
    bands = [(-math.inf, math.inf), *itertools.pairwise(PHASES)]
    for low, high in bands:
        part = mine & (notes.turn >= low) & (notes.turn < high)
        here = part & looked
        rows.append(
            (
                "All" if math.isinf(low) else phase(low, high),
                str(int(part.sum())),
                _share(int(here.sum()), int(part.sum())),
                _share(int((notes.rank[part] > 1).sum()), int(part.sum())),
                _share(int((notes.rank[here] > 1).sum()), int(here.sum())),
                _share(int(notes.complete[here].sum()), int(here.sum())),
            )
        )
    lines.append(
        "The reply player's plans in the finished games: the share that looked"
        " through 2 or more candidates, the share whose choice the reply"
        " changed, and of those that looked, the share changed and the share"
        " whose replies were all examined in full"
    )
    lines += table(rows)
    ranked = notes.rank[mine & looked]
    counts = [int((ranked == k).sum()) for k in range(1, r.candidates + 1)]
    lines += [
        "",
        "The pick's rank by the judgement among the candidates looked through: "
        + ", ".join(
            f"{k} {n} ({_share(n, len(ranked))})" for k, n in enumerate(counts, 1)
        ),
        "",
        f"Reading (sign z {found.z:.2f}, changed {100 * found.changed:.1f}%"
        f" against {100 * NOISE:.1f}%): {reading(found.z, found.changed)}",
    ]
    return lines
