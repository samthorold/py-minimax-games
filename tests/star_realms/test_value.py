from functools import partial

import numpy as np
import pytest

from minimax_games.games.star_realms import features, rules, value
from minimax_games.games.star_realms.evaluate import WIN
from minimax_games.games.star_realms.rollout import SPAN
from minimax_games.games.star_realms.state import PlayerState
from minimax_games.games.star_realms.value import (
    LOGISTIC,
    Form,
    Mlp,
    Model,
    Params,
    ValueEvaluator,
    fit,
    log_loss,
)


def toy(rows: int = 40, columns: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, columns))
    y = (rng.random(rows) < 0.5).astype(np.float64)
    return x, y


def numeric_gradient(
    form: Form, params: Params, x: np.ndarray, y: np.ndarray, l2: float
) -> Params:
    """Central finite differences of `form`'s loss, one parameter at a time."""
    step = 1e-6
    grads = {}
    for name, array in params.items():
        grad = np.zeros_like(array)
        for i in np.ndindex(array.shape):
            saved = array[i]
            array[i] = saved + step
            up, _ = form.loss_and_gradient(params, x, y, l2)
            array[i] = saved - step
            down, _ = form.loss_and_gradient(params, x, y, l2)
            array[i] = saved
            grad[i] = (up - down) / (2 * step)
        grads[name] = grad
    return grads


@pytest.mark.parametrize("form", [LOGISTIC, Mlp(hidden=5)], ids=["logistic", "mlp"])
@pytest.mark.parametrize("l2", [0.0, 0.1])
def test_analytic_gradients_match_finite_differences(form: Form, l2: float):
    x, y = toy()
    params = form.init(x.shape[1], np.random.default_rng(1))
    # Move off the initial point, so every parameter's gradient matters.
    for array in params.values():
        array += np.random.default_rng(2).normal(scale=0.3, size=array.shape)
    _, analytic = form.loss_and_gradient(params, x, y, l2)
    numeric = numeric_gradient(form, params, x, y, l2)
    assert analytic.keys() == numeric.keys()
    for name in analytic:
        np.testing.assert_allclose(analytic[name], numeric[name], rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("form", [LOGISTIC, Mlp(hidden=5)], ids=["logistic", "mlp"])
def test_the_loss_is_the_log_loss_of_the_predictions_without_l2(form: Form):
    x, y = toy()
    params = form.init(x.shape[1], np.random.default_rng(1))
    loss, _ = form.loss_and_gradient(params, x, y, 0.0)
    assert loss == pytest.approx(log_loss(form.predict(params, x), y))


def test_log_loss_of_a_coin_is_log_2():
    assert log_loss(np.full(4, 0.5), np.array([0, 1, 1, 0])) == pytest.approx(np.log(2))


@pytest.mark.parametrize("form", [LOGISTIC, Mlp(hidden=4)], ids=["logistic", "mlp"])
def test_fitting_is_deterministic_given_the_seed_and_lowers_the_loss(form: Form):
    x, y = toy(rows=200)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(np.float64)
    fitted = partial(fit, form, x, y, epochs=30, batch=32, learning_rate=0.05)
    first, second, other = fitted(seed=3), fitted(seed=3), fitted(seed=4)
    for name in first.params:
        assert np.array_equal(first.params[name], second.params[name])
    assert any(
        not np.array_equal(first.params[n], other.params[n]) for n in first.params
    )
    assert log_loss(first.predict(x), y) < 0.5 * np.log(2)


def test_a_model_saved_as_npz_loads_identically(tmp_path):
    x, y = toy()
    model = fit(Mlp(hidden=3), x, y, seed=0, epochs=2)
    path = tmp_path / "m.npz"
    model.save(path)
    loaded = Model.load(path)
    assert loaded.form == model.form
    assert loaded.names == model.names
    assert np.array_equal(loaded.predict(x), model.predict(x))
    # Plain arrays only: loading never needs pickle.
    with np.load(path, allow_pickle=False) as data:
        assert all(data[key].dtype != object for key in data.files)


def test_a_model_for_other_features_is_refused_by_the_evaluator():
    x, y = toy(columns=3)
    model = fit(LOGISTIC, x, y, seed=0, epochs=1)
    with pytest.raises(ValueError, match="features"):
        ValueEvaluator(model)


def fitted_on_states() -> Model:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, len(features.NAMES)))
    y = (rng.random(30) < 0.5).astype(np.float64)
    return fit(LOGISTIC, x, y, seed=0, epochs=3, names=features.NAMES)


def test_the_evaluator_scores_the_model_for_whoever_just_ended_and_its_complement():
    model = fitted_on_states()
    evaluator = ValueEvaluator(model)
    state = rules.new_game(0).replace(current=1)
    p = float(model.predict(features.encode(state)[None, :])[0])
    assert evaluator(state, 0) == round(SPAN * (2 * p - 1))
    assert evaluator(state, 1) == -evaluator(state, 0)
    assert "logistic" in evaluator.describe()


def test_the_evaluator_scores_a_finished_game_as_won_or_lost():
    evaluator = ValueEvaluator(fitted_on_states())
    state = rules.new_game(0)
    loser = state.players[1]
    over = state.replace(
        players=(state.players[0], PlayerState(0, loser.deck, loser.hand))
    )
    assert evaluator(over, 0) == WIN
    assert evaluator(over, 1) == -WIN


def test_a_missing_model_name_says_where_it_looked():
    with pytest.raises(FileNotFoundError, match="no-such-model"):
        value.load("no-such-model")
