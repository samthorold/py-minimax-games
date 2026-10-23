"""The sequential match of the planner at more sampled worlds against
itself at fewer (`more_worlds`)."""

import functools

from minimax_games.games.star_realms import more_worlds
from minimax_games.games.star_realms.sprt import Check, Hypothesis, Sprt


def _match(seeds: range) -> more_worlds.Tally:
    # tiny budgets, so a batch takes about a second
    return more_worlds.play_batch(
        "mlp", seeds, worlds=(4, 2), budgets=(400, 200), threads=2
    )


@functools.cache
def _small() -> more_worlds.Tally:
    return _match(range(50, 54))


def test_every_seed_is_played_in_both_seats_and_counted_by_pair():
    found = _small()
    assert found.games == 8
    both, split, neither = found.pairs
    assert both + split + neither == 4
    assert 2 * both + split == found.wins
    assert found.wins + found.losses == found.finished == 8
    first, second = found.first, found.second
    assert first[0] + second[0] == found.wins
    assert first[1] == second[1] == 4


def test_batches_add_up_to_the_match_played_at_once():
    whole = _small()
    added = _match(range(50, 52)) + _match(range(52, 54))
    assert (added.games, added.wins, added.losses) == (
        whole.games,
        whole.wins,
        whole.losses,
    )
    assert added.pairs == whole.pairs
    assert (added.first, added.second) == (whole.first, whole.second)
    assert added.plans == whole.plans
    assert added.states == whole.states


def test_each_player_s_time_and_states_per_plan_are_reported():
    found = _small()
    for player, budget in ((0, 400), (1, 200)):
        assert 0 < found.states_per_plan(player) <= budget
        assert found.ms_per_plan(player) > 0


def _sequential(test: Sprt, cap: int) -> list[tuple[more_worlds.Tally, Check]]:
    return list(
        more_worlds.sequential(
            "mlp",
            test,
            base_seed=50,
            batch=2,
            cap=cap,
            worlds=(4, 2),
            budgets=(400, 200),
            threads=2,
        )
    )


def test_the_match_plays_consecutive_batches_until_the_cap():
    steps = _sequential(Sprt(), cap=8)
    # 4 games a batch; neither bound is near after 8 games
    assert [check.games for _, check in steps] == [4, 8]
    assert steps[-1][1].decision is None
    tally = steps[-1][0]
    assert tally.pairs == _small().pairs
    assert steps[-1][1].wins == tally.wins == _small().wins


def test_the_match_stops_at_the_batch_that_decides_it():
    # H1 1% against H0 0.1%: two wins are strong evidence for H1
    steps = _sequential(Sprt(p0=0.001, p1=0.01), cap=1_000)
    assert steps[0][0].wins >= 2
    assert len(steps) == 1
    assert steps[0][1].decision is Hypothesis.H1


def test_the_reading_follows_the_rule_set_before_the_run():
    assert more_worlds.reading(Hypothesis.H1).startswith("H1 accepted")
    assert more_worlds.reading(Hypothesis.H0).startswith("H0 accepted")
    assert more_worlds.reading(None).startswith("undecided at the cap")
    assert "4 worlds stays" in more_worlds.reading(None)


def test_each_batch_reports_the_record_the_llr_and_the_pairs():
    tally, check = _sequential(Sprt(), cap=4)[-1]
    line = more_worlds.progress(tally, check, Sprt())
    assert line.startswith(f"{check.games} games: {check.wins} wins")
    assert "% (" in line  # the win rate and its interval
    assert f"LLR {check.llr:+.2f} in [-2.94, +2.94]" in line
    both, split, neither = tally.pairs
    assert f"pairs {both} / {split} / {neither}" in line
    assert "sign z" in line


def test_the_report_gives_both_players_time_and_states_and_the_reading():
    tally, check = _sequential(Sprt(), cap=4)[-1]
    lines = "\n".join(
        more_worlds.report(tally, check, Sprt(), ("4 worlds", "2 worlds"))
    )
    assert "4 worlds" in lines and "2 worlds" in lines
    assert "ms per plan" in lines and "States per plan" in lines
    assert "Seeds won both / split / lost both by 4 worlds" in lines
    assert "undecided at the cap" in lines


def test_the_stronger_player_s_wins_are_credited_to_it_in_either_order():
    # a planner with a real budget against one examining 2 states
    strong = more_worlds.play_batch(
        "mlp", range(60, 70), worlds=(2, 1), budgets=(2_000, 2), threads=2
    )
    weak = more_worlds.play_batch(
        "mlp", range(60, 70), worlds=(1, 2), budgets=(2, 2_000), threads=2
    )
    assert strong.wins >= 16
    assert weak.wins <= 4
