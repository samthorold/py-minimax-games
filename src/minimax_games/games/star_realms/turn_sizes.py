"""How many distinct ends a turn has: the whole-turn planner's workload.

Each turn start of seeded heuristic-vs-heuristic games is enumerated to
every distinct state reachable at the end of the mover's turn, orderings
merged as `bench.turn_tree` merges them, counting the states within the turn
visited to list them. A turn past `max_states` states is stopped there and
counted as capped, so the measurement always finishes. The counts are the
same on every run; only the timings vary.
"""

import math
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.action import EndTurn
from minimax_games.games.star_realms.bench import table
from minimax_games.games.star_realms.bots import HeuristicBot
from minimax_games.games.star_realms.card import Card
from minimax_games.games.star_realms.state import GameState, PlayerState
from minimax_games.games.star_realms.view import observe

MAX_STATES = 100_000
"""The default cap on the states one turn's enumeration visits."""

BUDGETS = (1_000, 5_000, 20_000, 100_000)
"""The candidate state budgets the report gives the share of turns over."""

BAND = 5
"""Turns per row of the report's table by turn number."""


def turn_starts(seed: int) -> Iterator[GameState]:
    """The first decision of each turn of the heuristic-vs-heuristic game
    from `seed`, in order, until someone wins."""
    bot = HeuristicBot()
    state = rules.auto_play(rules.new_game(seed))
    turn = 0
    while rules.winner(state) is None:
        if state.turn != turn:
            turn = state.turn
            yield state
        actions = rules.legal_actions(state)
        state = rules.advance(state, bot.choose(observe(state, state.current), actions))


def _sorted(cards: tuple[Card, ...]) -> tuple[Card, ...]:
    return tuple(sorted(cards, key=lambda card: card.name))


def _unordered_player(player: PlayerState) -> PlayerState:
    return player.replace(
        hand=_sorted(player.hand),
        in_play=_sorted(player.in_play),
        discard=_sorted(player.discard),
        bases=_sorted(player.bases),
    )


def unordered(state: GameState) -> GameState:
    """`state` with every pile but the decks sorted by card name.

    Ends of a turn that differ only in the order of a hand, a discard pile,
    the bases, the Trade Row or the scrap heap are one position to a value
    that counts cards by zone (design, Learned values). They are still
    distinct states: a discard pile's order decides the deck it is shuffled
    into.
    """
    first, second = state.players
    return state.replace(
        players=(_unordered_player(first), _unordered_player(second)),
        trade_row=_sorted(state.trade_row),
        scrap_heap=_sorted(state.scrap_heap),
        allies_used=_sorted(state.allies_used),
        bases_used=_sorted(state.bases_used),
    )


@dataclass(frozen=True, slots=True)
class TurnEnds:
    """The distinct ends of one turn, and the states visited to list them."""

    ends: int
    """Distinct end-of-turn states: the states after `EndTurn`, or after a
    play that wins the game. A lower bound if not `complete`."""
    unordered: int
    """Distinct ends up to the order of the piles (`unordered`)."""
    states: int
    """Distinct states within the turn, the start included, each expanded
    once."""
    complete: bool
    """False if the enumeration passed its cap and was stopped."""

    @property
    def visited(self) -> int:
        """Every distinct state visited: what a planner's state budget pays
        for. `bench.turn_tree`'s `states` for a complete turn."""
        return self.ends + self.states


def turn_ends(state: GameState, max_states: int | None = None) -> TurnEnds:
    """Enumerate the distinct ends of the turn from `state`, depth first,
    each state expanded once, stopping once more than `max_states` states
    would be visited."""
    within = {state}
    ends: set[GameState] = set()
    stack = [state]

    def found(complete: bool) -> TurnEnds:
        merged = len({unordered(end) for end in ends})
        return TurnEnds(len(ends), merged, len(within), complete)

    while stack:
        current = stack.pop()
        for action in rules.legal_actions(current):
            child = rules.advance_legal(current, action)
            end = isinstance(action, EndTurn) or rules.winner(child) is not None
            if child in (ends if end else within):
                continue
            if max_states is not None and len(ends) + len(within) >= max_states:
                return found(complete=False)
            if end:
                ends.add(child)
            else:
                within.add(child)
                stack.append(child)
    return found(complete=True)


@dataclass(frozen=True, slots=True)
class TurnSize:
    """One turn's enumeration: its counts are reproducible, its time measured."""

    seed: int
    turn: int
    ends: TurnEnds
    seconds: float


def game_turn_sizes(seed: int, max_states: int = MAX_STATES) -> list[TurnSize]:
    """Enumerate each turn of the heuristic-vs-heuristic game from `seed`."""
    sizes = []
    for start in turn_starts(seed):
        begun = time.perf_counter()
        ends = turn_ends(start, max_states)
        sizes.append(TurnSize(seed, start.turn, ends, time.perf_counter() - begun))
    return sizes


def turn_sizes(
    seeds: Sequence[int], max_states: int = MAX_STATES, jobs: int = 1
) -> list[TurnSize]:
    """`game_turn_sizes` for each of `seeds`, in order, over `jobs`
    processes: the counts are the same whatever `jobs` is."""
    measure = partial(game_turn_sizes, max_states=max_states)
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            games = list(pool.map(measure, seeds))
    else:
        games = [measure(seed) for seed in seeds]
    return [size for game in games for size in game]


def quantile(values: Sequence[float | None], q: float) -> float | None:
    """The nearest-rank `q` quantile of `values`, None (capped) ranking above
    every value."""
    if not values:
        msg = "no values"
        raise ValueError(msg)
    ranked = sorted(values, key=lambda v: math.inf if v is None else v)
    return ranked[max(0, math.ceil(q * len(ranked)) - 1)]


QUANTILES = (("q1", 0.25), ("median", 0.5), ("q3", 0.75), ("p90", 0.9))
QUANTILES += (("p99", 0.99), ("max", 1.0))


def _cell(value: float | None) -> str:
    if value is None:
        return "capped"
    return f"{value:.0f}" if isinstance(value, int) else f"{value:.1f}"


def _over(sizes: Sequence[TurnSize], budget: int) -> str:
    """The share of `sizes` that visit more than `budget` states."""
    over = sum(not s.ends.complete or s.ends.visited > budget for s in sizes)
    return f"{100 * over / len(sizes):.0f}%"


def report(
    sizes: Sequence[TurnSize],
    max_states: int = MAX_STATES,
    budgets: Sequence[int] = BUDGETS,
    band: int = BAND,
) -> list[str]:
    """The measurement as plain-text lines: the distribution of each count,
    the share of turns over each budget, then both by turn number. A capped
    turn's counts rank above every complete turn's and read `capped`.
    Budgets past `max_states` are left out: which turns pass them is unknown.
    """
    budgets = [b for b in budgets if b <= max_states]

    def counts(pick: str) -> list[int | None]:
        return [getattr(s.ends, pick) if s.ends.complete else None for s in sizes]

    seeds = sorted({s.seed for s in sizes})
    games = len(seeds)
    capped = sum(not s.ends.complete for s in sizes)
    rows = [("Count", *(name for name, _ in QUANTILES))]
    for label, values in (
        ("End states", counts("ends")),
        ("Unordered ends", counts("unordered")),
        ("Within-turn states", counts("states")),
        ("States visited", counts("visited")),
    ):
        rows.append((label, *(_cell(quantile(values, q)) for _, q in QUANTILES)))
    ms = [1000 * s.seconds for s in sizes]
    rows.append(("ms", *(_cell(quantile(ms, q)) for _, q in QUANTILES)))
    done = [s.ends for s in sizes if s.ends.complete]
    ends = sum(e.ends for e in done)
    band_rows = [("Turns", "Count", "median", "p90", *(f">{b}" for b in budgets))]
    last = max(s.turn for s in sizes)
    for first in range(1, last + 1, band):
        part = [s for s in sizes if first <= s.turn < first + band]
        if not part:
            continue
        visited = [s.ends.visited if s.ends.complete else None for s in part]
        band_rows.append(
            (
                f"{first}-{first + band - 1}",
                str(len(part)),
                _cell(quantile(visited, 0.5)),
                _cell(quantile(visited, 0.9)),
                *(_over(part, b) for b in budgets),
            )
        )
    seconds = sum(s.seconds for s in sizes if s.ends.complete)
    rate = sum(e.visited for e in done) / seconds if seconds else 0.0
    return [
        f"{len(sizes)} turns of {games} heuristic-vs-heuristic"
        f" game{'s' if games != 1 else ''} (seeds {seeds[0]}-{seeds[-1]}),"
        f" each enumerated up to {max_states} states; {capped} capped",
        "",
        *table(rows),
        "",
        "Over budget (states visited; capped turns count as over): "
        + ", ".join(f"{b} {_over(sizes, b)}" for b in budgets),
        f"Complete turns: {ends} ends, {sum(e.unordered for e in done)} up to"
        f" pile order, {sum(e.states for e in done)} within-turn states;"
        f" {rate:.0f} states/s",
        "",
        "By turn number (states visited):",
        *table(band_rows),
    ]
