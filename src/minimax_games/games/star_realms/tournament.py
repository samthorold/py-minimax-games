"""The tournament: seeded games between two bots, alternating seats."""

import time
from collections import deque
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import closing
from dataclasses import dataclass, replace

from minimax_games.games.star_realms.action import Action
from minimax_games.games.star_realms.bots import AnyBot, BenchmarkBot, Bot, Planning
from minimax_games.games.star_realms.match import MatchTooLong, play_match
from minimax_games.games.star_realms.sprt import Hypothesis, Sprt
from minimax_games.games.star_realms.state import GameState
from minimax_games.games.star_realms.view import View

MAX_TURNS = 300
"""The default turn cap. Games with greedy end in a few dozen turns; random
against random can take several hundred, so raise the cap for those."""

type BotFactory = Callable[[int], AnyBot]
"""Builds a bot for one game from a seed (which only seeded bots use)."""


@dataclass(frozen=True, slots=True)
class Game:
    """One tournament game. Bots are numbered 0 (A) and 1 (B)."""

    seed: int
    first: int
    """The bot that moved first."""
    winner: int | None
    """The winning bot, or None if the game was unfinished (passed the cap)."""
    turns: int
    """Turns played; for an unfinished game, the cap."""


@dataclass(frozen=True, slots=True)
class TournamentResult:
    """What the games came to: identical for the same seeds and bots."""

    games: tuple[Game, ...]

    @property
    def unfinished(self) -> tuple[int, ...]:
        """The seeds of the games that passed the turn cap."""
        return tuple(game.seed for game in self.games if game.winner is None)

    @property
    def finished(self) -> int:
        """How many games had a winner: the denominator of every win rate."""
        return len(self.games) - len(self.unfinished)

    def wins(self, bot: int) -> int:
        return sum(game.winner == bot for game in self.games)

    def record(self, bot: int, seat: int) -> tuple[int, int]:
        """`bot`'s wins and finished games in `seat` (0 moves first)."""
        seated = [
            game
            for game in self.games
            if game.winner is not None and (game.first == bot) == (seat == 0)
        ]
        return sum(game.winner == bot for game in seated), len(seated)

    @property
    def mean_turns(self) -> float:
        """The mean length of the finished games (0 if there are none)."""
        turns = [game.turns for game in self.games if game.winner is not None]
        return sum(turns) / len(turns) if turns else 0.0


@dataclass(frozen=True, slots=True)
class Timing:
    """Time spent inside each bot's `choose`: measured, so it varies by run."""

    decisions: tuple[int, int]
    seconds: tuple[float, float]
    wall_seconds: float
    """The whole tournament, rules included."""
    plans: tuple[int, int] = (0, 0)
    """Plans each bot made, if it plans (`bots.Planning`), else 0."""
    turns: tuple[int, int] = (0, 0)
    """Turns each planning bot played, else 0."""

    @property
    def ms_per_move(self) -> tuple[float, float]:
        """Each bot's mean milliseconds per decision (0 if it made none)."""
        return (self._ms_per_move(0), self._ms_per_move(1))

    def _ms_per_move(self, bot: int) -> float:
        n = self.decisions[bot]
        return 1000 * self.seconds[bot] / n if n else 0.0

    def ms_per_plan(self, bot: int) -> float:
        """A planning bot's mean milliseconds per plan: all its time per
        decision, since a decision it doesn't plan for only plays the plan's
        next move (0 if it made no plan)."""
        n = self.plans[bot]
        return 1000 * self.seconds[bot] / n if n else 0.0

    def plans_per_turn(self, bot: int) -> float:
        """A planning bot's mean plans per turn played (0 if it played none)."""
        n = self.turns[bot]
        return self.plans[bot] / n if n else 0.0


@dataclass(frozen=True, slots=True)
class Tournament:
    result: TournamentResult
    timing: Timing


class _Clock:
    """One bot's decisions so far, and the time spent making them."""

    def __init__(self) -> None:
        self.decisions = 0
        self.seconds = 0.0
        self.plans = 0
        self.turns = 0

    def count_plans(self, bot: AnyBot) -> None:
        """Add the plans and turns of `bot`, which has played its game, if it
        plans."""
        if isinstance(bot, Planning):
            self.plans += bot.plans
            self.turns += bot.turns


class _Timed:
    """Wraps a bot: times each of its decisions on `clock`."""

    def __init__(self, bot: Bot, clock: _Clock) -> None:
        self._bot = bot
        self._clock = clock

    def choose(self, view: View, actions: tuple[Action, ...]) -> Action:
        start = time.perf_counter()
        action = self._bot.choose(view, actions)
        self._clock.seconds += time.perf_counter() - start
        self._clock.decisions += 1
        return action


class _TimedBenchmark:
    """Wraps a benchmark-only bot: times each of its decisions on `clock`,
    and stays a `BenchmarkBot`, so it is still given the true state."""

    def __init__(self, bot: BenchmarkBot, clock: _Clock) -> None:
        self._bot = bot
        self._clock = clock

    def choose_from_state(
        self, state: GameState, actions: tuple[Action, ...]
    ) -> Action:
        start = time.perf_counter()
        action = self._bot.choose_from_state(state, actions)
        self._clock.seconds += time.perf_counter() - start
        self._clock.decisions += 1
        return action


def _timed(bot: AnyBot, clock: _Clock) -> AnyBot:
    """`bot`, timed on `clock`: a benchmark-only bot stays one, and every
    other bot still gets views only."""
    if isinstance(bot, BenchmarkBot):
        return _TimedBenchmark(bot, clock)
    return _Timed(bot, clock)


def play_tournament(
    bots: tuple[BotFactory, BotFactory],
    games: int,
    base_seed: int = 0,
    max_turns: int = MAX_TURNS,
    jobs: int = 1,
) -> Tournament:
    """Play `games` games from seeds `base_seed`, `base_seed + 1`, ...

    Bot A (`bots[0]`) moves first in even-numbered games, bot B in odd ones.
    A game that passes `max_turns` is unfinished: it has no winner and counts
    in no win rate.

    With `jobs` above 1 the games are split into up to `jobs` runs of
    consecutive seeds, played in separate processes (so `bots` must pickle),
    and combined: the result is the same as one run's. Each bot's time per
    decision is measured in the process that made the decision; the wall
    time is the whole parallel run's.
    """
    if jobs < 1:
        msg = f"jobs must be at least 1, not {jobs}"
        raise ValueError(msg)
    if jobs > 1 and games > 0:
        return _play_parallel(bots, games, base_seed, max_turns, jobs)
    return _play_games(bots, 0, games, base_seed, max_turns)


def _play_games(
    bots: tuple[BotFactory, BotFactory],
    start: int,
    stop: int,
    base_seed: int,
    max_turns: int,
) -> Tournament:
    """Games `start` to `stop - 1` of a tournament from `base_seed`, in one
    process: game `i` from seed `base_seed + i`, bot A first if `i` is even."""
    began = time.perf_counter()
    clocks = (_Clock(), _Clock())
    played = []
    for i in range(start, stop):
        seed = base_seed + i
        first = i % 2
        players = (bots[first](seed), bots[1 - first](seed + 1))
        seated = (clocks[first], clocks[1 - first])
        seats = (_timed(players[0], seated[0]), _timed(players[1], seated[1]))
        try:
            match = play_match(seats, seed, max_turns=max_turns)
        except MatchTooLong:
            winner, turns = None, max_turns
        else:
            winner = first if match.winner == 0 else 1 - first
            turns = match.turns
        for player, clock in zip(players, seated, strict=True):
            clock.count_plans(player)
        played.append(Game(seed=seed, first=first, winner=winner, turns=turns))
    timing = Timing(
        decisions=(clocks[0].decisions, clocks[1].decisions),
        seconds=(clocks[0].seconds, clocks[1].seconds),
        wall_seconds=time.perf_counter() - began,
        plans=(clocks[0].plans, clocks[1].plans),
        turns=(clocks[0].turns, clocks[1].turns),
    )
    return Tournament(result=TournamentResult(games=tuple(played)), timing=timing)


def _play_parallel(
    bots: tuple[BotFactory, BotFactory],
    games: int,
    base_seed: int,
    max_turns: int,
    jobs: int,
) -> Tournament:
    """`play_tournament`'s games, in runs of an even number of games each, so
    every run starts with bot A first, as its games do in one long run."""
    start = time.perf_counter()
    size = -(-games // jobs)
    size += size % 2
    offsets = range(0, games, size)
    with ProcessPoolExecutor(max_workers=len(offsets)) as pool:
        parts = list(
            pool.map(
                play_tournament,
                [bots] * len(offsets),
                [min(size, games - offset) for offset in offsets],
                [base_seed + offset for offset in offsets],
                [max_turns] * len(offsets),
            )
        )
    combined = combine(parts)
    timing = replace(combined.timing, wall_seconds=time.perf_counter() - start)
    return Tournament(result=combined.result, timing=timing)


def combine(parts: Sequence[Tournament]) -> Tournament:
    """The tournament that `parts`, run on consecutive seeds, add up to.

    Each part after the first must start an even number of games after the
    first part's base seed, so its seats alternate as in one long run; then
    the result is exactly that run's. Decisions, plans, turns and time add up, and
    so does the wall time (the time the parts took one after another).
    """
    timings = [part.timing for part in parts]
    return Tournament(
        result=TournamentResult(
            games=tuple(game for part in parts for game in part.result.games)
        ),
        timing=Timing(
            decisions=(
                sum(t.decisions[0] for t in timings),
                sum(t.decisions[1] for t in timings),
            ),
            seconds=(
                sum(t.seconds[0] for t in timings),
                sum(t.seconds[1] for t in timings),
            ),
            wall_seconds=sum(t.wall_seconds for t in timings),
            plans=(sum(t.plans[0] for t in timings), sum(t.plans[1] for t in timings)),
            turns=(sum(t.turns[0] for t in timings), sum(t.turns[1] for t in timings)),
        ),
    )


@dataclass(frozen=True, slots=True)
class SprtTournament:
    """A tournament stopped by a sequential probability ratio test."""

    tournament: Tournament
    """The games up to the one that decided (or the limit), in seed order."""
    test: Sprt
    verdict: Hypothesis | None
    """The hypothesis accepted, or None if the game limit came first."""


def play_sprt(
    bots: tuple[BotFactory, BotFactory],
    test: Sprt,
    max_games: int,
    base_seed: int = 0,
    max_turns: int = MAX_TURNS,
    jobs: int = 1,
) -> SprtTournament:
    """`play_tournament`'s games, stopped as soon as `test` decides on bot
    A's wins and losses (unfinished games count as neither), or after
    `max_games`.

    With `jobs` above 1 games are played ahead in up to `jobs` processes,
    but the decision is taken on completed games in seed order and the
    games past the deciding one are dropped, so the result is one
    process's. Their decisions are dropped from the timing too; the wall
    time includes them.
    """
    if jobs < 1:
        msg = f"jobs must be at least 1, not {jobs}"
        raise ValueError(msg)
    start = time.perf_counter()
    parts = []
    wins = [0, 0]
    verdict = None
    with closing(_one_by_one(bots, max_games, base_seed, max_turns, jobs)) as games:
        for part in games:
            parts.append(part)
            winner = part.result.games[0].winner
            if winner is not None:
                wins[winner] += 1
            verdict = test.decide(*wins)
            if verdict is not None:
                break
    combined = combine(parts)
    timing = replace(combined.timing, wall_seconds=time.perf_counter() - start)
    return SprtTournament(
        tournament=Tournament(result=combined.result, timing=timing),
        test=test,
        verdict=verdict,
    )


def _one_by_one(
    bots: tuple[BotFactory, BotFactory],
    games: int,
    base_seed: int,
    max_turns: int,
    jobs: int,
) -> Generator[Tournament]:
    """Each of `games` games as a tournament of its own, in seed order.
    With `jobs` above 1, up to twice `jobs` games are played ahead in
    separate processes; those still waiting are cancelled on close."""
    if jobs == 1:
        for i in range(games):
            yield _play_games(bots, i, i + 1, base_seed, max_turns)
        return
    pool = ProcessPoolExecutor(max_workers=jobs)
    try:
        ahead: deque[Future[Tournament]] = deque()
        submitted = 0
        while submitted < games or ahead:
            while submitted < games and len(ahead) < 2 * jobs:
                ahead.append(
                    pool.submit(
                        _play_games,
                        bots,
                        submitted,
                        submitted + 1,
                        base_seed,
                        max_turns,
                    )
                )
                submitted += 1
            yield ahead.popleft().result()
    finally:
        pool.shutdown(cancel_futures=True)


@dataclass(frozen=True, slots=True)
class PairedTournament:
    """Every seed played twice, once with each bot first."""

    tournament: Tournament
    """The games of both seatings, bot A numbered 0 in all of them."""

    @property
    def pairs(self) -> dict[int, int]:
        """How many seeds bot A won both, one and neither games of, counting
        only seeds whose two games both finished."""
        by_seed: dict[int, list[int | None]] = {}
        for game in self.tournament.result.games:
            by_seed.setdefault(game.seed, []).append(game.winner)
        pairs = dict.fromkeys((2, 1, 0), 0)
        for winners in by_seed.values():
            if None not in winners:
                pairs[sum(winner == 0 for winner in winners)] += 1
        return pairs


def _swapped(played: Tournament) -> Tournament:
    """`played` with its bots renumbered, A as B and B as A."""

    def swap[T](pair: tuple[T, T]) -> tuple[T, T]:
        return pair[1], pair[0]

    timing = played.timing
    return Tournament(
        result=TournamentResult(
            games=tuple(
                replace(
                    game,
                    first=1 - game.first,
                    winner=None if game.winner is None else 1 - game.winner,
                )
                for game in played.result.games
            )
        ),
        timing=replace(
            timing,
            decisions=swap(timing.decisions),
            seconds=swap(timing.seconds),
            plans=swap(timing.plans),
            turns=swap(timing.turns),
        ),
    )


def play_paired(
    bots: tuple[BotFactory, BotFactory],
    seeds: int,
    base_seed: int = 0,
    max_turns: int = MAX_TURNS,
    jobs: int = 1,
) -> PairedTournament:
    """Each of `seeds` seeds from `base_seed` played twice, once with each
    bot first, as two tournaments (`play_tournament`), the second with the
    bots' seats swapped: comparing the two games of a seed takes the deal's
    luck out of the comparison. The wall time is both tournaments'."""
    first = play_tournament(bots, seeds, base_seed, max_turns, jobs)
    second = play_tournament((bots[1], bots[0]), seeds, base_seed, max_turns, jobs)
    return PairedTournament(combine([first, _swapped(second)]))
