"""Temporal-difference targets (`td`) for the learned value
(docs/star-realms.md, Learned values: Targets)."""

import numpy as np
import pytest

from minimax_games.games.star_realms import fit, td
from minimax_games.games.star_realms.value import load

# two games: game 0 of three states, won by the player who ended the last
# (so the labels alternate 1, 0, 1), game 1 of two states, lost by the player
# who ended the last
Y = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
GAME = np.array([0, 0, 0, 3, 3])
POSITION = np.array([0, 1, 2, 0, 1])
VALUE = np.array([0.6, 0.3, 0.8, 0.45, 0.9])
"""Each state's value for the player who just ended it."""


def test_lambda_one_gives_the_game_results():
    assert td.targets(Y, GAME, POSITION, VALUE, lam=1.0).tolist() == Y.tolist()


def test_lambda_one_gives_the_results_of_real_games():
    data = fit.dataset(range(4))
    value = load("mlp").predict(data.x)
    targets = td.targets(data.y, data.game, data.position, value, lam=1.0)
    np.testing.assert_allclose(targets, data.y, rtol=0, atol=1e-12)


def test_lambda_zero_bootstraps_from_the_next_state_of_the_same_game():
    targets = td.targets(Y, GAME, POSITION, VALUE, lam=0.0)
    # the next state was ended by the opponent: its value is theirs
    assert targets.tolist() == pytest.approx([1 - 0.3, 1 - 0.8, 1.0, 1 - 0.9, 0.0])


def test_the_last_state_of_a_game_takes_its_result_not_the_next_game_s_value():
    for lam in (0.0, 0.3, 0.7):
        targets = td.targets(Y, GAME, POSITION, VALUE, lam=lam)
        assert targets[2] == Y[2]
        assert targets[4] == Y[4]


def test_lambda_mixes_the_bootstrap_with_the_next_state_s_target():
    lam = 0.25
    last = Y[2]
    middle = (1 - lam) * (1 - VALUE[2]) + lam * (1 - last)
    first = (1 - lam) * (1 - VALUE[1]) + lam * (1 - middle)
    second_game = (1 - lam) * (1 - VALUE[4]) + lam * (1 - Y[4])
    targets = td.targets(Y, GAME, POSITION, VALUE, lam=lam)
    assert targets.tolist() == pytest.approx([first, middle, last, second_game, Y[4]])


def test_targets_stay_probabilities():
    rng = np.random.default_rng(0)
    for lam in (0.0, 0.5, 0.9):
        targets = td.targets(Y, GAME, POSITION, rng.uniform(size=5), lam=lam)
        assert ((targets >= 0) & (targets <= 1)).all()


@pytest.mark.parametrize(
    ("game", "position"),
    [
        (np.array([0, 0, 0, 3, 3]), np.array([0, 2, 1, 0, 1])),  # out of order
        (np.array([0, 0, 0, 3, 3]), np.array([0, 1, 3, 0, 1])),  # a gap
        (np.array([0, 0, 3, 0, 3]), np.array([0, 1, 0, 2, 1])),  # interleaved
        (np.array([0, 0, 0, 3, 3]), np.array([1, 2, 3, 0, 1])),  # no start
    ],
)
def test_rows_must_be_whole_games_in_order(game: np.ndarray, position: np.ndarray):
    with pytest.raises(ValueError, match="order"):
        td.targets(Y, game, position, VALUE, lam=0.5)


def test_lambda_must_be_between_zero_and_one():
    for lam in (-0.1, 1.1):
        with pytest.raises(ValueError, match="lam"):
            td.targets(Y, GAME, POSITION, VALUE, lam=lam)
