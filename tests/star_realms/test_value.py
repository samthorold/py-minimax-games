from functools import partial

import numpy as np
import pytest

from minimax_games.games.star_realms import features, rules, value
from minimax_games.games.star_realms.action import EndTurn
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


PACKAGED = sorted(path.stem for path in value._models().glob("*.npz"))
"""Every committed model: the first fits and each rung of the ladder."""


def test_the_first_fits_are_packaged():
    assert {"logistic", "mlp"} <= set(PACKAGED)


@pytest.mark.parametrize("name", PACKAGED)
def test_the_packaged_models_load_by_name_for_the_current_features(name: str):
    evaluator = ValueEvaluator(value.load(name))
    state = rules.apply(rules.auto_play(rules.new_game(0)), EndTurn())
    assert -SPAN <= evaluator(state, 0) <= SPAN


def test_an_ensemble_s_value_is_the_mean_of_its_members_win_probabilities():
    x, y = toy()
    members = [fit(Mlp(hidden=3), x, y, seed=s, epochs=2) for s in (0, 1, 2)]
    model = value.ensemble(members)
    assert len(model.members) == 3
    assert model.form == "mlp"
    expected = sum(m.predict(x) for m in members) / 3
    assert np.array_equal(model.predict(x), expected)


def test_a_one_member_ensemble_is_the_member():
    x, y = toy()
    member = fit(LOGISTIC, x, y, seed=0, epochs=2)
    model = value.ensemble([member])
    assert np.array_equal(model.predict(x), member.predict(x))
    assert model.params is member.params


def test_only_fits_to_the_same_form_and_standardisation_make_an_ensemble():
    x, y = toy()
    a = fit(LOGISTIC, x, y, seed=0, epochs=1)
    with pytest.raises(ValueError, match="standardis"):
        value.ensemble([a, fit(LOGISTIC, x + 1, y, seed=1, epochs=1)])
    with pytest.raises(ValueError, match="form"):
        value.ensemble([a, fit(Mlp(hidden=2), x, y, seed=1, epochs=1)])
    with pytest.raises(ValueError, match="member"):
        value.ensemble([])


def test_a_model_of_several_members_has_no_single_params():
    x, y = toy()
    model = value.ensemble([fit(LOGISTIC, x, y, seed=s, epochs=1) for s in (0, 1)])
    with pytest.raises(ValueError, match="members"):
        _ = model.params


def test_an_ensemble_saved_as_npz_loads_identically(tmp_path):
    x, y = toy()
    members = [fit(Mlp(hidden=3), x, y, seed=s, epochs=2) for s in (0, 1)]
    model = value.ensemble(members)
    path = tmp_path / "e.npz"
    model.save(path)
    loaded = Model.load(path)
    assert len(loaded.members) == 2
    assert np.array_equal(loaded.predict(x), model.predict(x))
    with np.load(path, allow_pickle=False) as data:
        assert all(data[key].dtype != object for key in data.files)


def test_a_one_member_model_is_saved_as_before(tmp_path):
    x, y = toy()
    model = fit(Mlp(hidden=3), x, y, seed=0, epochs=2)
    model.save(tmp_path / "m.npz")
    with np.load(tmp_path / "m.npz", allow_pickle=False) as data:
        assert set(data.files) == {
            "form",
            "names",
            "mean",
            "std",
            "param_w1",
            "param_b1",
            "param_w2",
            "param_b2",
        }


def test_fit_ensemble_fits_one_member_per_seed():
    x, y = toy()
    model = value.fit_ensemble(LOGISTIC, x, y, seeds=(3, 4), epochs=2)
    for member, seed in zip(model.members, (3, 4), strict=True):
        alone = fit(LOGISTIC, x, y, seed=seed, epochs=2)
        assert all(np.array_equal(member[n], alone.params[n]) for n in member)


def state_rows(games: range = range(2)) -> tuple[np.ndarray, np.ndarray]:
    from minimax_games.games.star_realms.fit import dataset

    data = dataset(games)
    return data.x, data.y


def test_a_model_with_extra_features_reads_them_from_the_counts():
    x, y = state_rows()
    model = fit(
        Mlp(hidden=3),
        x,
        y,
        seed=0,
        epochs=2,
        names=features.NAMES,
        extra=features.EXTRA,
    )
    assert model.extra == features.EXTRA
    assert len(model.mean) == len(features.NAMES) + len(features.EXTRA)
    assert model.params["w1"].shape == (len(model.mean), 3)
    # the same as a model of the widened rows
    widened = np.hstack([x, features.extra(x).astype(x.dtype)])
    plain = Model(model.form, model.names, model.mean, model.std, model.members)
    np.testing.assert_array_equal(model.predict(x), plain.predict(widened))
    ValueEvaluator(model)  # still a value of features.NAMES


def test_a_model_with_extra_features_saves_and_loads_them(tmp_path):
    x, y = state_rows()
    model = value.fit_ensemble(
        LOGISTIC,
        x,
        y,
        seeds=(0, 1),
        epochs=1,
        names=features.NAMES,
        extra=features.EXTRA,
    )
    model.save(tmp_path / "m.npz")
    loaded = Model.load(tmp_path / "m.npz")
    assert loaded.extra == features.EXTRA
    assert np.array_equal(loaded.predict(x), model.predict(x))
    with np.load(tmp_path / "m.npz", allow_pickle=False) as data:
        assert all(data[key].dtype != object for key in data.files)


def test_a_model_without_extra_features_saves_none(tmp_path):
    x, y = toy()
    fit(LOGISTIC, x, y, seed=0, epochs=1).save(tmp_path / "m.npz")
    with np.load(tmp_path / "m.npz", allow_pickle=False) as data:
        assert "extra" not in data.files
    assert Model.load(tmp_path / "m.npz").extra == ()


def test_only_the_known_extra_features_are_taken():
    x, y = toy()
    with pytest.raises(ValueError, match="extra"):
        fit(LOGISTIC, x, y, seed=0, epochs=1, extra=("kill threat",))
