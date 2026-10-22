"""The reference population (`population`, docs/star-realms.md, How
progress is measured): a player against every member on fixed seeds, seats
alternating; per-seed results written and paired with a recorded run's;
the report."""

from pathlib import Path

import pytest

from minimax_games.games.star_realms import population
from minimax_games.games.star_realms.cli import main
from minimax_games.games.star_realms.sprt import wilson

TINY = {"worlds": 1, "budget": 60, "threads": 2}
"""Plans of a few states, so a tiny match takes well under a second."""


def test_a_player_plays_each_opponent_on_the_same_seeds_seats_alternating():
    matches = population.play("greedy", opponents=("heuristic", "mlp"), games=4, **TINY)
    assert [m.opponent for m in matches] == ["heuristic", "mlp"]
    for match in matches:
        assert [(g.seed, g.first, g.second) for g in match.games] == [
            (0, "greedy", match.opponent),
            (1, match.opponent, "greedy"),
            (2, "greedy", match.opponent),
            (3, match.opponent, "greedy"),
        ]
        assert all(g.winner in ("greedy", match.opponent) for g in match.games)
        assert match.wins == sum(g.winner == "greedy" for g in match.games)


def test_a_player_is_not_played_against_itself():
    matches = population.play("mlp", opponents=("heuristic", "mlp"), games=2, **TINY)
    assert [m.opponent for m in matches] == ["heuristic"]


def test_results_are_the_same_whatever_the_number_of_threads():
    def run(threads: int) -> tuple[population.Match, ...]:
        return population.play(
            "mlp",
            opponents=("heuristic", "pi03"),
            games=6,
            worlds=1,
            budget=60,
            threads=threads,
        )

    assert run(1) == run(3)


def test_per_seed_results_are_written_one_csv_per_opponent_as_gating_matches_are(
    tmp_path: Path,
):
    matches = population.play(
        "greedy", opponents=("heuristic", "mlp"), games=4, max_turns=12, **TINY
    )
    population.write(matches, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["heuristic.csv", "mlp.csv"]
    lines = (tmp_path / "heuristic.csv").read_text().splitlines()
    assert lines[0] == "seed,first,second,winner,turns"
    assert lines[1].startswith("0,greedy,heuristic,")
    for match in matches:
        assert population.read(tmp_path / f"{match.opponent}.csv") == match.games


def test_unfinished_games_are_recorded_with_no_winner(tmp_path: Path):
    (match,) = population.play(
        "greedy", opponents=("heuristic",), games=2, max_turns=3, **TINY
    )
    assert [g.winner for g in match.games] == [None, None]
    population.write([match], tmp_path)
    row = (tmp_path / "heuristic.csv").read_text().splitlines()[1]
    assert row.startswith("0,greedy,heuristic,,")
    assert population.read(tmp_path / "heuristic.csv") == match.games


def _games(player: str, opponent: str, winners: str) -> tuple[population.Game, ...]:
    """Games on seeds 0 up, seats alternating, from a string of who won each:
    `p` the player, `o` the opponent, `-` unfinished."""
    return tuple(
        population.Game(
            seed,
            *((player, opponent) if seed % 2 == 0 else (opponent, player)),
            {"p": player, "o": opponent, "-": None}[w],
            20,
        )
        for seed, w in enumerate(winners)
    )


def test_a_run_pairs_with_a_recorded_run_seed_by_seed():
    match = population.Match("cand", "mlp", _games("cand", "mlp", "ppoop-o"))
    recorded = _games("champ", "mlp", "poppoop")
    paired = population.pair(match, recorded)
    assert paired.recorded == "champ"
    assert paired.seeds == 6  # seed 5 is unfinished in the candidate's run
    assert paired.only_player == (1, 4)
    assert paired.only_recorded == (2, 3, 6)
    assert paired.z == pytest.approx((2 - 3) / 5**0.5)


def test_pairing_uses_only_the_seeds_both_runs_played():
    match = population.Match("cand", "mlp", _games("cand", "mlp", "pppp"))
    paired = population.pair(match, _games("champ", "mlp", "oo"))
    assert paired.seeds == 2
    assert paired.only_player == (0, 1)


def test_pairing_refuses_runs_against_another_opponent_or_in_other_seats():
    match = population.Match("cand", "mlp", _games("cand", "mlp", "pp"))
    with pytest.raises(ValueError, match="opponent"):
        population.pair(match, _games("champ", "pi03", "pp"))
    swapped = tuple(
        population.Game(g.seed, g.second, g.first, g.winner, g.turns)
        for g in _games("champ", "mlp", "pp")
    )
    with pytest.raises(ValueError, match="seat"):
        population.pair(match, swapped)


def test_the_report_gives_each_opponents_win_rate_with_its_interval_and_their_mean():
    matches = (
        population.Match("cand", "heuristic", _games("cand", "heuristic", "ppppoopp")),
        population.Match("cand", "mlp", _games("cand", "mlp", "ppoo-")),
    )
    assert matches[0].rate == 0.75
    assert matches[0].interval == pytest.approx(wilson(6, 8))
    assert matches[1].finished == 4
    assert population.mean(matches) == pytest.approx(0.625)
    lines = population.report(matches)
    text = "\n".join(lines)
    heuristic = next(line for line in lines if line.startswith("heuristic"))
    assert "6/8" in heuristic
    assert "75.0%" in heuristic
    low, high = wilson(6, 8)
    assert f"{low:.1%}-{high:.1%}" in heuristic
    assert "3/4" in heuristic  # as first player
    mlp = next(line for line in lines if line.startswith("mlp"))
    assert "2/4" in mlp
    assert "Mean" in text
    assert "62.5%" in text
    assert "Unfinished: 1 (mlp: seed 4)" in text


def test_the_report_gives_the_seeds_won_only_by_each_player_when_paired():
    match = population.Match("cand", "mlp", _games("cand", "mlp", "ppoop-o"))
    paired = population.pair(match, _games("champ", "mlp", "poppoop"))
    text = "\n".join(population.report([match], [paired]))
    assert "against mlp, paired with champ on 6 seeds:" in text
    assert "won only by cand 2, only by champ 3" in text
    assert f"sign z {paired.z:.2f}" in text


def test_one_command_plays_the_population_and_pairs_with_a_recorded_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    tiny = ["--games", "4", "--worlds", "1", "--budget", "60", "--threads", "2"]
    champion = tmp_path / "mlp"
    main(["population", "mlp", "--opponents", "heuristic", "greedy", "mlp",
          "--out", str(champion), *tiny])  # fmt: skip
    out = capsys.readouterr().out
    assert sorted(p.name for p in champion.iterdir()) == ["greedy.csv", "heuristic.csv"]
    assert "Mean" in out
    candidate = tmp_path / "pi03"
    main(["population", "pi03", "--opponents", "heuristic", "greedy", "mlp",
          "--out", str(candidate), "--paired", str(champion), *tiny])  # fmt: skip
    out = capsys.readouterr().out
    assert "pi03 against heuristic, paired with mlp on" in out
    assert "pi03 against greedy, paired with mlp on" in out
    assert "against mlp: no recorded results" in out


def test_the_champions_per_seed_baseline_is_recorded_against_every_other_member():
    recorded = Path(__file__).parents[2] / "docs" / "population" / "pf01"
    others = [o for o in population.POPULATION if o != "pf01"]
    assert sorted(p.stem for p in recorded.glob("*.csv")) == sorted(others)
    for opponent in others:
        games = population.read(recorded / f"{opponent}.csv")
        expected = [
            (s, "pf01", opponent) if s % 2 == 0 else (s, opponent, "pf01")
            for s in range(population.GAMES)
        ]
        assert [(g.seed, g.first, g.second) for g in games] == expected
