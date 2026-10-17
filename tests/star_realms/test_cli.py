from minimax_games.games.star_realms.cli import main


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
