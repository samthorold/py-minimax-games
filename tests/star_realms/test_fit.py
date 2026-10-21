import numpy as np
import pytest

from minimax_games.games.star_realms import features
from minimax_games.games.star_realms.fit import (
    Settings,
    boundaries,
    dataset,
    report,
    run,
)
from minimax_games.games.star_realms.rollout import SCALE, win_probability


def test_a_game_gives_each_turn_boundary_and_who_won():
    game = boundaries(0)
    assert game.winner in (0, 1)
    assert len(game.states) > 10
    for i, state in enumerate(game.states):
        assert state.turn == i + 2
        ended = state.players[1 - state.current]
        assert ended.in_play == () and ended.hand  # just ended, and drew
        assert state.trade == state.combat == 0


def test_games_are_deterministic_given_the_seed_and_epsilon():
    assert boundaries(5) == boundaries(5)
    assert boundaries(5, epsilon=0.1) == boundaries(5, epsilon=0.1)
    assert boundaries(5, epsilon=0.1) != boundaries(5)


def test_a_dataset_labels_each_state_for_the_player_who_just_ended():
    data = dataset(range(3))
    games = [boundaries(seed) for seed in range(3)]
    states = [s for g in games for s in g.states]
    assert data.x.shape == (len(states), len(features.NAMES))
    assert np.array_equal(data.x, features.encode_all(states))
    labels = [float(g.winner == 1 - s.current) for g in games for s in g.states]
    assert data.y.tolist() == labels
    assert data.evaluate.tolist() == pytest.approx(
        [win_probability(s, 1 - s.current, SCALE) for s in states]
    )


def test_a_dataset_records_each_state_s_game_and_position_in_it():
    data = dataset(range(10, 13))
    lengths = [len(boundaries(seed).states) for seed in range(10, 13)]
    assert data.game.tolist() == [g for g, n in enumerate(lengths) for _ in range(n)]
    assert data.position.tolist() == [i for n in lengths for i in range(n)]


def test_a_dataset_is_the_same_in_any_number_of_processes():
    one, two = dataset(range(4)), dataset(range(4), jobs=2)
    assert np.array_equal(one.x, two.x)
    assert np.array_equal(one.y, two.y)
    assert np.array_equal(one.game, two.game)
    assert np.array_equal(one.position, two.position)


def test_a_run_is_deterministic_and_reports_every_log_loss():
    settings = Settings(train_games=6, test_games=3, epochs=3, hidden=4)
    first, second = run(settings), run(settings)
    assert first.losses == second.losses
    assert set(first.losses) == {"constant", "evaluate", "logistic", "mlp"}
    lines = report(first, settings)
    assert any("evaluate" in line for line in lines)
    assert any(f"{first.test_states}" in line for line in lines)
