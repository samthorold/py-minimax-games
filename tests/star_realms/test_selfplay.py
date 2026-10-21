"""Self-play in Rust (`selfplay`, `minimax_games._native.star_realms.self_play`)
against the Python reference: the value's forward pass against numpy's, the
training rows against `fit`'s, and the Rust planner's games against the
Python planner's with the same model (docs/star-realms.md, Compiled engine:
self-play)."""

import random

import numpy as np
import pytest

from minimax_games._native import star_realms as native
from minimax_games.games.star_realms import fit
from minimax_games.games.star_realms.action import Action
from minimax_games.games.star_realms.bots import Bot, GreedyBot, HeuristicBot
from minimax_games.games.star_realms.match import play_match
from minimax_games.games.star_realms.planner import PlannerBot, native_state
from minimax_games.games.star_realms.rollout import SPAN
from minimax_games.games.star_realms.selfplay import (
    Pairing,
    Planner,
    alternating,
    native_model,
    play,
)
from minimax_games.games.star_realms.turn_sizes import turn_starts
from minimax_games.games.star_realms.value import (
    LOGISTIC,
    Mlp,
    Model,
    ValueEvaluator,
    load,
)
from minimax_games.games.star_realms.view import View, observe, sample_world
from tests.star_realms.test_native import native_action

TOLERANCE = 1e-12
"""How far the Rust probabilities may be from numpy's: the two sum the
products in different orders (numpy's matrix products are blocked), so they
may differ in the last bits, about 1e-16."""


def _random_model(form: str, columns: int, seed: int) -> Model:
    """A model with random parameters and standardisation, of `columns`
    features."""
    rng = np.random.default_rng(seed)
    params = (Mlp(hidden=7) if form == "mlp" else LOGISTIC).init(columns, rng)
    params = {
        name: a + rng.normal(scale=0.3, size=a.shape) for name, a in params.items()
    }
    names = tuple(f"x{i}" for i in range(columns))
    mean = rng.normal(size=columns)
    std = rng.uniform(0.5, 2, size=columns)
    return Model(form, names, mean, std, (params,))


@pytest.mark.parametrize("form", ["logistic", "mlp"])
def test_the_forward_pass_agrees_with_numpy_on_random_inputs(form: str):
    model = _random_model(form, 23, seed=3)
    x = np.random.default_rng(4).normal(scale=3, size=(500, 23))
    rust = native.predict(native_model(model), x)
    np.testing.assert_allclose(rust, model.predict(x), rtol=0, atol=TOLERANCE)


@pytest.mark.parametrize("form", ["logistic", "mlp"])
def test_an_ensemble_s_forward_pass_agrees_with_numpy(form: str):
    first = _random_model(form, 23, seed=3)
    # three members with the first's standardisation
    model = Model(
        first.form,
        first.names,
        first.mean,
        first.std,
        tuple(_random_model(form, 23, seed=s).params for s in (3, 4, 5)),
    )
    x = np.random.default_rng(4).normal(scale=3, size=(500, 23))
    rust = native.predict(native_model(model), x)
    np.testing.assert_allclose(rust, model.predict(x), rtol=0, atol=TOLERANCE)
    one = Model(model.form, model.names, model.mean, model.std, model.members[:1])
    assert not np.allclose(rust, native.predict(native_model(one), x))


def _real_rows() -> np.ndarray:
    """The feature rows of every end of turn the planner lists from the turn
    starts of a seeded heuristic game, in one world each."""
    rows = []
    for state in turn_starts(0):
        world = sample_world(observe(state, state.current), random.Random(0))
        rows.append(native.plan_ends([native_state(world)], 5_000)["features"])
    return np.concatenate(rows)


REAL_ROWS = _real_rows()


def _value(name: str) -> Model:
    """A packaged model, or `"ensemble"`: three members, `mlp`'s, `pi03`'s
    and `mlp`'s again, all under `mlp`'s standardisation."""
    if name != "ensemble":
        return load(name)
    mlp = load("mlp")
    members = (mlp.params, load("pi03").params, mlp.params)
    return Model(mlp.form, mlp.names, mlp.mean, mlp.std, members)


@pytest.mark.parametrize("name", ["logistic", "mlp", "ensemble"])
def test_the_forward_pass_agrees_with_numpy_on_real_rows(name: str):
    model = _value(name)
    assert len(REAL_ROWS) > 5_000
    rust = native.predict(native_model(model), REAL_ROWS)
    numpy = model.predict(REAL_ROWS)
    np.testing.assert_allclose(rust, numpy, rtol=0, atol=TOLERANCE)
    # and the scores the planner compares, rounded to whole points, are equal
    assert np.array_equal(
        np.round(SPAN * (2 * rust - 1)), np.round(SPAN * (2 * numpy - 1))
    )


def test_a_model_of_another_form_is_refused():
    model = native_model(load("logistic"))
    with pytest.raises(ValueError, match="form"):
        native.predict({**model, "form": "forest"}, REAL_ROWS[:2])


def test_heuristic_games_give_the_rows_fit_builds():
    seeds = range(6)
    data = fit.dataset(seeds)
    played = play([(seed, "heuristic", "heuristic") for seed in seeds])
    assert played.games == data.games == 6
    assert played.x.dtype == np.float32
    assert np.array_equal(played.x, data.x)
    assert np.array_equal(played.y, data.y)
    assert np.array_equal(played.game, data.game)
    assert np.array_equal(played.position, data.position)
    assert played.winners == tuple(fit.boundaries(seed).winner for seed in seeds)


def test_an_unfinished_game_gives_no_rows():
    played = play([(0, "heuristic", "heuristic"), (1, "greedy", "greedy")], max_turns=3)
    assert played.winners == (None, None)
    assert played.games == 0
    assert played.x.shape == (0, len(fit.features.NAMES))


def _batch() -> list[Pairing]:
    logistic = Planner(load("logistic"), worlds=2, budget=300)
    mlp = Planner(load("mlp"), worlds=2, budget=300)
    return [
        *alternating(logistic, "heuristic", range(100, 104)),
        (104, logistic, mlp),
        (105, mlp, "greedy"),
        (106, "heuristic", "greedy"),
    ]


def test_results_are_the_same_whatever_the_number_of_threads():
    games = _batch()
    one = play(games, threads=1, record=True)
    for threads in (2, 5):
        other = play(games, threads=threads, record=True)
        assert np.array_equal(other.x, one.x)
        assert np.array_equal(other.y, one.y)
        assert np.array_equal(other.game, one.game)
        assert np.array_equal(other.position, one.position)
        assert other.winners == one.winners
        assert other.turns == one.turns
        assert np.array_equal(other.plans, one.plans)
        assert np.array_equal(other.diverged, one.diverged)
        assert other.actions == one.actions
    # every game but the last has a planner, and it planned
    assert (one.plans.sum(axis=1) > 0).tolist() == [True] * 6 + [False]


class _Recorder:
    """Records every decision a bot makes, as the Rust engine names it."""

    def __init__(self, bot: Bot, decisions: list[native.NativeAction]) -> None:
        self.bot = bot
        self.decisions = decisions

    def choose(self, view: View, actions: tuple[Action, ...]) -> Action:
        action = self.bot.choose(view, actions)
        self.decisions.append(native_action(action))
        return action


@pytest.mark.parametrize(
    ("seed", "model"),
    [(0, "logistic"), (1, "mlp"), (2, "mlp"), (3, "ensemble"), (4, "ensemble")],
)
def test_the_rust_planner_plays_the_python_planner_s_games(seed: int, model: str):
    value = _value(model)
    opponent = "heuristic" if seed % 2 else "greedy"
    planner = Planner(value, worlds=3, budget=1_500)
    # the planner moves first on even seeds, second on odd ones
    seats = (planner, opponent) if seed % 2 == 0 else (opponent, planner)
    rust = play([(seed, *seats)], record=True)

    decisions: list[native.NativeAction] = []
    bot = PlannerBot(
        seed + (seed % 2),
        worlds=3,
        budget=1_500,
        evaluator=ValueEvaluator(value),
        backend="rust",
    )
    other = HeuristicBot() if opponent == "heuristic" else GreedyBot()
    pair = (bot, other) if seed % 2 == 0 else (other, bot)
    result = play_match(
        (_Recorder(pair[0], decisions), _Recorder(pair[1], decisions)), seed
    )
    assert rust.actions is not None
    assert rust.actions[0] == decisions
    assert rust.winners == (result.winner,)
    assert rust.turns == (result.turns,)
    me = 0 if seed % 2 == 0 else 1
    assert rust.plans[0, me] == bot.plans >= 20
    assert rust.diverged[0, me] == bot.diverged
    assert rust.plans[0, 1 - me] == 0
