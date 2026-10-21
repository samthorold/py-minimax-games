import math

import pytest

from minimax_games.games.star_realms.sprt import Hypothesis, Sprt, wilson


def test_the_bounds_come_from_the_error_rates():
    test = Sprt(p0=0.5, p1=0.55, alpha=0.05, beta=0.1)
    assert test.lower == pytest.approx(math.log(0.1 / 0.95))
    assert test.upper == pytest.approx(math.log(0.9 / 0.05))


def test_the_llr_adds_a_term_per_win_and_per_loss():
    test = Sprt(p0=0.5, p1=0.55)
    assert test.llr(0, 0) == 0
    assert test.llr(3, 2) == pytest.approx(3 * math.log(1.1) + 2 * math.log(0.9))


def decisions(test: Sprt, outcomes: str) -> list[Hypothesis | None]:
    """The decision after each game of `outcomes`: W a win, L a loss."""
    wins = losses = 0
    made = []
    for outcome in outcomes:
        wins += outcome == "W"
        losses += outcome == "L"
        made.append(test.decide(wins, losses))
    return made


def test_wins_accept_h1_once_the_llr_passes_the_upper_bound():
    # Each win adds log(1.1); log(19) / log(1.1) is about 30.9.
    made = decisions(Sprt(p0=0.5, p1=0.55), "W" * 31)
    assert made[:30] == [None] * 30
    assert made[30] is Hypothesis.H1


def test_losses_accept_h0_once_the_llr_passes_the_lower_bound():
    # Each loss adds log(0.9); log(1 / 19) / log(0.9) is about 27.9.
    made = decisions(Sprt(p0=0.5, p1=0.55), "L" * 28)
    assert made[:27] == [None] * 27
    assert made[27] is Hypothesis.H0


def test_an_even_record_stays_undecided_for_a_while():
    assert set(decisions(Sprt(p0=0.5, p1=0.55), "WL" * 100)) == {None}


def test_wider_hypotheses_decide_sooner():
    # Each win adds log(1.4); log(19) / log(1.4) is about 8.8.
    made = decisions(Sprt(p0=0.5, p1=0.7), "W" * 9)
    assert made[-2:] == [None, Hypothesis.H1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p0": 0.55, "p1": 0.5},
        {"p0": 0.0, "p1": 0.5},
        {"p0": 0.5, "p1": 1.0},
        {"alpha": 0.0},
        {"beta": 1.0},
        {"alpha": 0.6, "beta": 0.5},
    ],
)
def test_hypotheses_and_error_rates_must_make_sense(kwargs: dict[str, float]):
    with pytest.raises(ValueError):
        Sprt(**kwargs)


def test_the_wilson_interval_brackets_the_win_rate():
    low, high = wilson(5, 10)
    assert low == pytest.approx(1 - high)
    assert low < 0.5 < high
    assert wilson(0, 10)[0] == 0
    assert wilson(10, 10)[1] == pytest.approx(1)
    assert wilson(0, 0) == (0.0, 1.0)
