import pytest

from minimax_games.games.star_realms.action import Action
from minimax_games.games.star_realms.bots import GreedyBot
from minimax_games.games.star_realms.cheat import CheatBot
from minimax_games.games.star_realms.cli import BOTS, main
from minimax_games.games.star_realms.state import GameState
from minimax_games.games.star_realms.view import observe


def test_prints_the_result_of_a_seeded_game(capsys):
    main(["--seed", "7"])
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1
    assert "wins on turn" in out[0]


def test_with_log_prints_the_transcript(capsys):
    main(["--seed", "7", "--log"])
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("Turn 1: Player 1 to play.")
    assert "wins on turn" in out[-1]


def test_bots_can_be_picked_by_name(capsys):
    main(["--seed", "7", "--bots", "greedy", "random", "--log"])
    out = capsys.readouterr().out.splitlines()
    assert out[-1].startswith("Player 1 wins on turn")
    assert any(line.startswith("Player 1 acquires ") for line in out)


def test_tournament_reports_win_rates_by_seat_and_time_per_move(capsys):
    main(["tournament", "--bots", "greedy", "random", "--games", "4"])
    out = capsys.readouterr().out
    assert "greedy vs random: 4 games, seeds 0-3, max 300 turns" in out
    assert "Unfinished: 0" in out
    lines = out.splitlines()
    table = lines[lines.index("") + 1 :][:3]
    header, greedy, random_ = (line.split() for line in table)
    assert header == ["Bot", "Wins", "Win%", "As", "1st", "As", "2nd", "ms/move"]
    assert greedy[1] == "4/4"
    assert greedy[2] == "100.0%"
    assert random_[1] == "0/4"


def test_tournament_lists_unfinished_games_and_tells_same_bots_apart(capsys):
    argv = ["tournament", "--bots", "random", "random", "--games", "2"]
    main([*argv, "--max-turns", "5"])
    out = capsys.readouterr().out.splitlines()
    assert "Unfinished: 2 (seeds 0, 1)" in out
    assert "Win rates are of the 0 finished games." in out
    assert any(line.startswith("random-A  0/0") for line in out)
    assert any(line.startswith("random-B  0/0") for line in out)


def test_tournament_plays_the_cheat(capsys):
    # The cheat at its default horizon is slow past the opening: stop early.
    argv = ["tournament", "--bots", "cheat", "greedy", "--games", "2"]
    main([*argv, "--max-turns", "4"])
    out = capsys.readouterr().out.splitlines()
    assert "Unfinished: 2 (seeds 0, 1)" in out
    cheat = next(line for line in out if line.split()[:2] == ["cheat", "0/0"])
    assert float(cheat.split()[-1]) > 0  # it made timed decisions


def test_the_cheat_is_marked_benchmark_only_in_the_help(capsys):
    for argv in (["--help"], ["tournament", "--help"]):
        with pytest.raises(SystemExit):
            main(argv)
        help_text = " ".join(capsys.readouterr().out.split())
        assert "cheat (benchmark only: sees the whole state)" in help_text


class Peeker:
    """A fast stand-in for the cheat: benchmark-only, plays greedily."""

    def __init__(self) -> None:
        self.calls = 0

    def choose_from_state(
        self, state: GameState, actions: tuple[Action, ...]
    ) -> Action:
        self.calls += 1
        return GreedyBot().choose(observe(state, state.current), actions)


def test_a_single_game_plays_the_cheat(capsys, monkeypatch):
    # The real cheat is too slow for a whole game: check it is the one
    # registered, then play a fast benchmark-only stand-in under its name.
    assert isinstance(BOTS["cheat"](0), CheatBot)
    peeker = Peeker()
    monkeypatch.setitem(BOTS, "cheat", lambda seed: peeker)
    main(["--seed", "7", "--bots", "cheat", "greedy"])
    assert "wins on turn" in capsys.readouterr().out
    assert peeker.calls > 0
