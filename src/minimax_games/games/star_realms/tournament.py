"""The tournament: seeded games between two bots, alternating seats."""

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from minimax_games.games.star_realms.action import Action
from minimax_games.games.star_realms.bots import AnyBot, BenchmarkBot, Bot
from minimax_games.games.star_realms.match import MatchTooLong, play_match
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

    @property
    def ms_per_move(self) -> tuple[float, float]:
        """Each bot's mean milliseconds per decision (0 if it made none)."""
        return (self._ms_per_move(0), self._ms_per_move(1))

    def _ms_per_move(self, bot: int) -> float:
        n = self.decisions[bot]
        return 1000 * self.seconds[bot] / n if n else 0.0


@dataclass(frozen=True, slots=True)
class Tournament:
    result: TournamentResult
    timing: Timing


class _Clock:
    """One bot's decisions so far, and the time spent making them."""

    def __init__(self) -> None:
        self.decisions = 0
        self.seconds = 0.0


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
    start = time.perf_counter()
    clocks = (_Clock(), _Clock())
    played = []
    for i in range(games):
        seed = base_seed + i
        first = i % 2
        seats = (
            _timed(bots[first](seed), clocks[first]),
            _timed(bots[1 - first](seed + 1), clocks[1 - first]),
        )
        try:
            match = play_match(seats, seed, max_turns=max_turns)
        except MatchTooLong:
            winner, turns = None, max_turns
        else:
            winner = first if match.winner == 0 else 1 - first
            turns = match.turns
        played.append(Game(seed=seed, first=first, winner=winner, turns=turns))
    timing = Timing(
        decisions=(clocks[0].decisions, clocks[1].decisions),
        seconds=(clocks[0].seconds, clocks[1].seconds),
        wall_seconds=time.perf_counter() - start,
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
    timing = Timing(
        decisions=combined.timing.decisions,
        seconds=combined.timing.seconds,
        wall_seconds=time.perf_counter() - start,
    )
    return Tournament(result=combined.result, timing=timing)


def combine(parts: Sequence[Tournament]) -> Tournament:
    """The tournament that `parts`, run on consecutive seeds, add up to.

    Each part after the first must start an even number of games after the
    first part's base seed, so its seats alternate as in one long run; then
    the result is exactly that run's. Decisions and their time add up, and
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
        ),
    )
