import pytest

from minimax_games.games.star_realms.evaluate import WEIGHTS, Weights
from minimax_games.games.star_realms.tune import (
    TrialCache,
    compare,
    decision_states,
    describe,
    neighbours,
    play,
    screen,
    search,
)


def test_a_paired_comparison_counts_extra_wins_and_discordant_seeds():
    candidate = (True, True, False, True)
    incumbent = (True, False, True, False)
    comparison = compare(candidate, incumbent)
    assert comparison.gain == 1
    assert comparison.discordant == 3


def test_a_gain_must_pass_the_noise_of_the_discordant_seeds():
    four_of_four = compare((True,) * 4, (False,) * 4)  # 4 > 1.64 * 2
    assert four_of_four.improves()
    three_of_four = compare((True,) * 3 + (False,), (False,) * 3 + (True,))
    assert not three_of_four.improves()  # +2 on 4 discordant
    assert not compare((True, False), (True, False)).improves()


def test_runs_on_different_seeds_dont_compare():
    with pytest.raises(ValueError, match="different seeds"):
        compare((True,), (True, False))


def test_a_weight_steps_up_and_down_by_the_factor():
    up, down = neighbours(WEIGHTS, "trade", 2)
    assert (up.trade, down.trade) == (2 * WEIGHTS.trade, WEIGHTS.trade / 2)
    assert up.combat == WEIGHTS.combat


def test_a_whole_number_weight_stays_whole_and_at_least_one():
    assert [w.ally_hand for w in neighbours(Weights(ally_hand=4), "ally_hand")] == [
        8,
        2,
    ]
    assert [w.ally_hand for w in neighbours(Weights(ally_hand=1), "ally_hand")] == [2]


def test_describe_lists_the_changed_weights():
    assert describe(WEIGHTS) == "unchanged"
    assert describe(Weights(trade=40, draw=15)) == "trade=40, draw=15"


def fake_play(weights: Weights) -> tuple[bool, ...]:
    """20 seeds: trade above 20 wins 6 more, below 20 loses 1; draw nothing."""
    wins = 10 + (6 if weights.trade > 20 else -1 if weights.trade < 20 else 0)
    return (True,) * wins + (False,) * (20 - wins)


def test_the_search_keeps_a_change_that_beats_the_noise_and_logs_each_step():
    lines: list[str] = []
    tuned = search(fake_play, ["trade", "draw"], log=lines.append)
    assert tuned == Weights(trade=40)
    assert lines[0] == "start: 10/20 wins"
    assert "  trade 20 -> 40: 16/20 wins, +6 on 6 discordant (needs > 4.0)" in lines
    assert "  keep trade = 40: 16/20 wins" in lines
    assert "  draw 30 -> 60: 16/20 wins, +0 on 0 discordant (needs > 0.0)" in lines
    assert lines[-1] == "end: 16/20 wins; trade=40"


def test_the_search_stops_after_a_pass_that_changes_nothing():
    lines: list[str] = []
    search(fake_play, ["draw"], passes=3, log=lines.append)
    assert lines.count("pass 1") == 1
    assert "pass 2" not in lines


def test_another_pass_starts_from_the_kept_weights():
    lines: list[str] = []
    tuned = search(fake_play, ["trade"], passes=2, log=lines.append)
    assert tuned == Weights(trade=40)  # 80 gains nothing over 40
    assert "  trade 40 -> 80: 16/20 wins, +0 on 0 discordant (needs > 0.0)" in lines


def test_the_cache_replays_trials_already_played(tmp_path):
    played: list[Weights] = []

    def counting(weights: Weights) -> tuple[bool, ...]:
        played.append(weights)
        return fake_play(weights)

    path = tmp_path / "trials.jsonl"
    settings = {"base_seed": 3000, "games": 20, "budget": 5000}
    first = TrialCache(path, settings)
    assert first.play(Weights(trade=40), counting) == fake_play(Weights(trade=40))
    again = TrialCache(path, settings)
    assert again.play(Weights(trade=40), counting) == fake_play(Weights(trade=40))
    assert played == [Weights(trade=40)]
    other = TrialCache(path, {**settings, "budget": 1000})
    other.play(Weights(trade=40), counting)
    assert len(played) == 2


def test_play_runs_the_cheat_against_the_heuristic_on_the_seeds():
    outcomes = play(WEIGHTS, base_seed=3000, games=2, budget=20)
    assert len(outcomes) == 2
    assert play(WEIGHTS, base_seed=3000, games=2, budget=20, jobs=2) == outcomes


def test_the_screen_measures_how_often_a_weight_changes_the_cheats_move():
    states = decision_states([3000], budget=20, every=10)
    assert states
    shares = screen(states, budget=20, fields=["trade", "draw"])
    assert set(shares) == {"trade", "draw"}
    assert all(0 <= share <= 1 for share in shares.values())
    assert list(shares.values()) == sorted(shares.values(), reverse=True)
