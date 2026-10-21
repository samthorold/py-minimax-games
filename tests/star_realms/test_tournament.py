import pytest

from minimax_games.games.star_realms import transcript
from minimax_games.games.star_realms.action import Action, EndTurn
from minimax_games.games.star_realms.bots import GreedyBot, RandomBot
from minimax_games.games.star_realms.match import play_match
from minimax_games.games.star_realms.tournament import (
    TournamentResult,
    combine,
    play_tournament,
)
from minimax_games.games.star_realms.view import View


def greedy(seed: int) -> GreedyBot:
    return GreedyBot()


def test_seats_alternate_evenly():
    result = play_tournament((greedy, RandomBot), games=4).result
    assert [game.seed for game in result.games] == [0, 1, 2, 3]
    assert [game.first for game in result.games] == [0, 1, 0, 1]


class Passive:
    """Never attacks: two of them never finish a game. It ends every turn,
    after a discard it must make."""

    def choose(self, view: View, actions: tuple[Action, ...]) -> Action:
        return EndTurn() if EndTurn() in actions else actions[0]


def passive(seed: int) -> Passive:
    return Passive()


def test_games_past_the_turn_cap_are_unfinished_not_errors():
    result = play_tournament((passive, passive), games=2, max_turns=5).result
    assert [game.winner for game in result.games] == [None, None]
    assert result.unfinished == (0, 1)
    assert result.finished == 0


def test_the_same_seeds_and_bots_give_the_same_result():
    def run(base_seed: int) -> TournamentResult:
        return play_tournament((RandomBot, RandomBot), 3, base_seed=base_seed).result

    assert run(20) == run(20)
    assert run(20) != run(30)


def test_win_rates_count_wins_overall_and_by_seat():
    result = play_tournament((greedy, passive), games=4).result
    assert result.wins(0) == 4
    assert result.wins(1) == 0
    assert result.record(0, seat=0) == (2, 2)
    assert result.record(0, seat=1) == (2, 2)
    assert result.record(1, seat=0) == (0, 2)
    assert result.mean_turns > 0


def test_the_game_log_is_off(monkeypatch: pytest.MonkeyPatch):
    def fail(*args: object) -> str:
        raise AssertionError("formatted a log line during a tournament")

    for formatter in ("turn_header", "describe", "result"):
        monkeypatch.setattr(transcript, formatter, fail)

    assert play_tournament((greedy, RandomBot), games=2).result.finished == 2


def test_timing_measures_each_bots_decisions():
    timing = play_tournament((greedy, passive), games=2).timing
    assert all(n > 0 for n in timing.decisions)
    assert all(ms >= 0 for ms in timing.ms_per_move)


def test_each_game_replays_as_a_single_match():
    game = play_tournament((greedy, RandomBot), games=2).result.games[1]
    match = play_match((RandomBot(1), GreedyBot()), seed=1)
    assert (game.first, game.turns) == (1, match.turns)
    assert game.winner == 1 - match.winner


def test_chunks_from_even_offsets_combine_into_the_whole_run():
    whole = play_tournament((greedy, RandomBot), games=6)
    chunks = [
        play_tournament((greedy, RandomBot), games=4),
        play_tournament((greedy, RandomBot), games=2, base_seed=4),
    ]
    combined = combine(chunks)
    assert combined.result == whole.result
    assert combined.timing.decisions == whole.timing.decisions
    assert combined.timing.wall_seconds == sum(
        chunk.timing.wall_seconds for chunk in chunks
    )


def test_games_in_parallel_give_the_serial_result():
    serial = play_tournament((RandomBot, RandomBot), games=7, base_seed=3)
    parallel = play_tournament((RandomBot, RandomBot), games=7, base_seed=3, jobs=3)
    assert parallel.result == serial.result
    assert parallel.timing.decisions == serial.timing.decisions


def test_jobs_must_be_positive():
    with pytest.raises(ValueError, match="jobs"):
        play_tournament((RandomBot, RandomBot), games=2, jobs=0)
