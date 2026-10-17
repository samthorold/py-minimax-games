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
