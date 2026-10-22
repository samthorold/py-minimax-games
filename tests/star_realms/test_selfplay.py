"""Self-play in Rust (`selfplay`, `minimax_games._native.star_realms.self_play`)
against the Python reference: the value's forward pass against numpy's, the
training rows against `fit`'s, and the Rust planner's games against the
Python planner's with the same model (docs/star-realms.md, Compiled engine:
self-play)."""

import functools
import random

import numpy as np
import pytest

from minimax_games._native import star_realms as native
from minimax_games.games.star_realms import features, fit, value
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


@functools.cache
def _extra_model(more: bool = False) -> Model:
    """An ensemble of 2 small networks reading `features.EXTRA` too, and
    `features.MORE` if `more`, fitted to a few heuristic games."""
    data = fit.dataset(range(40))
    return value.fit_ensemble(
        Mlp(hidden=8),
        data.x,
        data.y,
        seeds=(0, 1),
        epochs=3,
        names=features.NAMES,
        extra=features.EXTRA + (features.MORE if more else ()),
    )


def _value(name: str) -> Model:
    """A packaged model, `"ensemble"`: three members, `mlp`'s, `pi03`'s and
    `mlp`'s again, all under `mlp`'s standardisation, or `"extra"`, a model
    reading the extra features (`_extra_model`), or `"more"`, one reading
    the more features too."""
    if name == "extra":
        return _extra_model()
    if name == "more":
        return _extra_model(more=True)
    if name != "ensemble":
        return load(name)
    mlp = load("mlp")
    members = (mlp.params, load("pi03").params, mlp.params)
    return Model(mlp.form, mlp.names, mlp.mean, mlp.std, members)


@pytest.mark.parametrize("name", ["logistic", "mlp", "ensemble", "extra", "more"])
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
    [
        (0, "logistic"),
        (1, "mlp"),
        (2, "mlp"),
        (3, "ensemble"),
        (4, "ensemble"),
        (5, "extra"),
        (6, "extra"),
        (13, "more"),
        (8, "more"),
    ],
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


def test_a_model_s_extra_features_cross_with_it():
    model = native_model(_extra_model())
    assert set(model["extra"]) == {"combat", "faction", "base_card"}
    assert "extra" not in native_model(load("mlp"))
    assert "more" not in native_model(load("pf01"))["extra"]
    more = native_model(_extra_model(more=True))
    assert set(more["extra"]["more"]) == {
        "trade",
        "authority",
        "draw",
        "scrap",
        "discard",
        "base",
        "outpost_defense",
    }
    # a model reading the more features needs their tables
    without = {**more, "extra": model["extra"]}
    with pytest.raises(ValueError, match="features"):
        native.predict(without, REAL_ROWS)
    # without them, its inputs don't fit the rows
    with pytest.raises(ValueError, match="features"):
        native.predict({k: v for k, v in model.items() if k != "extra"}, REAL_ROWS)


def test_each_played_end_is_recorded_with_the_judgement_of_whoever_played_it():
    games = _batch()
    played = play(games, judge=True)
    assert played.judgement is not None
    finished = [g for g, w in zip(games, played.winners, strict=True) if w is not None]
    assert len(finished) == played.games
    for row, (game, seat) in enumerate(zip(played.game, played.seat, strict=True)):
        player = games[game][1 + seat]
        if isinstance(player, Planner):
            expected = player.model.predict(played.x[row : row + 1].astype(np.float64))
            assert abs(played.judgement[row] - expected[0]) <= TOLERANCE
        else:
            assert np.isnan(played.judgement[row])
    # the first turn is seat 0's, then the seats alternate
    assert np.array_equal(played.seat, played.position % 2)
    assert not np.isnan(played.judgement).all()


def test_judging_and_diagnosing_change_no_game():
    games = _batch()
    plain = play(games, threads=2, record=True)
    judged = play(games, threads=2, record=True, judge=True, diagnose=True)
    assert plain.judgement is None
    assert plain.left is None
    assert plain.examined is None
    assert judged.judgement is not None
    assert judged.judgement.shape == judged.y.shape
    assert judged.left is not None
    assert judged.left.shape == (len(judged.y), 2)
    assert judged.examined is not None
    finished = [w is not None for w in judged.winners]
    assert len(judged.examined.game) == judged.plans[finished].sum()
    assert judged.actions == plain.actions
    assert judged.winners == plain.winners
    assert judged.turns == plain.turns
    assert np.array_equal(judged.x, plain.x)
    assert np.array_equal(judged.y, plain.y)
    assert np.array_equal(judged.plans, plain.plans)
    assert np.array_equal(judged.diverged, plain.diverged)


def test_each_end_records_the_cards_each_player_has_left_to_draw():
    # the cards left in each player's deck before their discard pile is
    # reshuffled, the player who ended first, as the Python engine has them
    seeds = range(4)
    played = play([(seed, "heuristic", "heuristic") for seed in seeds], diagnose=True)
    assert played.left is not None
    expected = [
        [len(s.players[1 - s.current].deck), len(s.players[s.current].deck)]
        for seed in seeds
        for s in fit.boundaries(seed).states
    ]
    assert played.left.tolist() == expected


class _Examining(PlannerBot):
    """The Python planner, noting each plan's turn and whether it examined
    every end of the turn in every world (`Plan.complete`)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # ty: ignore[invalid-argument-type]
        self.examined: list[tuple[int, bool]] = []

    def _plan(self, view: View):
        state, made = super()._plan(view)
        self.examined.append((view.turn, made.complete))
        return state, made


def test_each_plan_records_its_turn_and_whether_it_examined_every_end():
    # a budget small enough that some plans run out
    value = _value("mlp")
    planner = Planner(value, worlds=2, budget=150)
    games: list[Pairing] = [(21, planner, "heuristic"), (22, "heuristic", planner)]
    played = play(games, diagnose=True)
    assert played.examined is not None
    for game, (seed, *seats) in enumerate(games):
        me = seats.index(planner)
        bot = _Examining(
            seed + me,
            worlds=2,
            budget=150,
            evaluator=ValueEvaluator(value),
            backend="rust",
        )
        pair = (bot, HeuristicBot()) if me == 0 else (HeuristicBot(), bot)
        play_match(pair, seed)
        mine = played.examined.game == game
        assert np.all(played.examined.seat[mine] == me)
        found = list(
            zip(
                played.examined.turn[mine].tolist(),
                played.examined.complete[mine].tolist(),
                strict=True,
            )
        )
        assert found == bot.examined
        assert len(found) == played.plans[game, me]
    # some plans ran out and some didn't
    assert played.examined.complete.any()
    assert not played.examined.complete.all()
