"""Differential tests: the Rust planner's enumeration
(`minimax_games._native.star_realms.plan_ends`) against the Python planner,
the reference (docs/star-realms.md, Compiled engine: the planner).

Both list the ends of a turn in the same order, so the Python planner's
evaluator is called on exactly the Rust rows, in row order: the tests record
those calls and compare the states and their features (`features.encode`)
row for row, then compare whole plans and whole games with either backend.
"""

import itertools
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from minimax_games._native import star_realms as native
from minimax_games.games.star_realms import features
from minimax_games.games.star_realms.bots import GreedyBot, HeuristicBot
from minimax_games.games.star_realms.evaluate import evaluate
from minimax_games.games.star_realms.match import play_match
from minimax_games.games.star_realms.planner import (
    Backend,
    PlannerBot,
    native_state,
    plan,
)
from minimax_games.games.star_realms.state import GameState
from minimax_games.games.star_realms.turn_sizes import turn_starts
from minimax_games.games.star_realms.value import ValueEvaluator, load
from minimax_games.games.star_realms.view import observe, sample_world
from tests.star_realms.test_native import snapshot
from tests.star_realms.test_planner import with_a_draw


@dataclass
class Recorder:
    """The linear evaluation, recording every state it scores, in order."""

    states: list[GameState] = field(default_factory=list)

    def __call__(self, state: GameState, perspective: int) -> int:
        self.states.append(state)
        return evaluate(state, perspective)

    def describe(self) -> str:
        return "recorder"


def positions() -> list[tuple[list[GameState], int]]:
    """Worlds sampled from the turn starts of seeded heuristic games, and a
    budget for each: 1 to 4 worlds, budgets small enough to run out."""
    found = []
    budgets = itertools.cycle((40, 300, 3_000, 1_200))
    starts = [*turn_starts(0), *turn_starts(1)]
    for i, state in enumerate([with_a_draw(), *starts]):
        view = observe(state, state.current)
        worlds = [
            sample_world(view, random.Random(100 * i + w)) for w in range(i % 4 + 1)
        ]
        found.append((worlds, next(budgets)))
    return found


POSITIONS = positions()


def test_it_lists_the_same_ends_with_the_same_features():
    complete = 0
    for i, (worlds, budget) in enumerate(POSITIONS):
        recorder = Recorder()
        reference = plan(worlds, budget, recorder)
        complete += reference.complete
        found = native.plan_ends([native_state(w) for w in worlds], budget, ends=True)
        where = f"position {i}, {len(worlds)} worlds, budget {budget}"
        assert found["ends"] == [snapshot(s) for s in recorder.states], where
        expected = features.encode_all(recorder.states)
        assert found["features"].dtype == np.float64
        assert np.array_equal(found["features"], expected), where
        assert found["states"] == reference.states, where
        assert found["complete"] == reference.complete, where
    # the budgets both run out and suffice
    assert 0 < complete < len(POSITIONS)


@pytest.mark.parametrize("model", ["logistic", "mlp"])
def test_it_makes_the_same_plans_as_the_python_planner(model: str):
    evaluator = ValueEvaluator(load(model))
    for i, (worlds, budget) in enumerate(POSITIONS):
        rust = plan(worlds, budget, evaluator, backend="rust")
        assert rust == plan(worlds, budget, evaluator), f"position {i}"


def test_the_rust_backend_needs_an_evaluator_of_feature_rows():
    with pytest.raises(ValueError, match="feature rows"):
        plan([with_a_draw()], 100, backend="rust")
    with pytest.raises(ValueError, match="backend"):
        unknown: Any = "fortran"
        plan([with_a_draw()], 100, backend=unknown)


def test_a_value_scores_rows_as_it_scores_states():
    evaluator = ValueEvaluator(load("mlp"))
    worlds, budget = POSITIONS[5]
    found = native.plan_ends([native_state(w) for w in worlds], budget, ends=True)
    recorder = Recorder()
    plan(worlds, budget, recorder)
    for perspective in (0, 1):
        scores = evaluator.score_rows(
            found["features"], found["winner"], found["current"], perspective
        )
        assert scores.tolist() == [evaluator(s, perspective) for s in recorder.states]


@pytest.mark.parametrize(("seed", "model"), [(0, "logistic"), (1, "mlp")])
def test_the_bot_plays_the_same_game_with_either_backend(seed: int, model: str):
    evaluator = ValueEvaluator(load(model))

    def game(backend: Backend) -> tuple[list[str], int, int]:
        lines: list[str] = []
        bot = PlannerBot(
            seed, worlds=3, budget=1_500, evaluator=evaluator, backend=backend
        )
        opponent = HeuristicBot() if seed % 2 else GreedyBot()
        play_match((bot, opponent), seed, log=lines.append)
        return lines, bot.plans, bot.diverged

    python = game("python")
    assert python[1] >= 20
    assert game("rust") == python
