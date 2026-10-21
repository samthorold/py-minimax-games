import math
import random

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.bots import GreedyBot, HeuristicBot
from minimax_games.games.star_realms.evaluate import WIN, evaluate
from minimax_games.games.star_realms.rollout import (
    SCALE,
    SPAN,
    Rollout,
    greedy,
    heuristic,
    playout,
)
from minimax_games.games.star_realms.state import GameState
from minimax_games.games.star_realms.view import observe
from tests.star_realms.test_cheat import mid_turn


def random_states(seed: int, moves: int) -> list[GameState]:
    """The states of a random game from `seed`, at most `moves` decisions."""
    rng = random.Random(seed)
    state = rules.auto_play(rules.new_game(seed))
    states = []
    while rules.winner(state) is None and len(states) < moves:
        states.append(state)
        state = rules.advance_legal(state, rng.choice(rules.legal_actions(state)))
    return states


def test_the_greedy_policy_plays_as_the_greedy_bot_without_a_view():
    for seed in range(10):
        for state in random_states(seed, 200):
            actions = rules.legal_actions(state)
            view = observe(state, state.current)
            assert greedy(state, actions) == GreedyBot().choose(view, actions)


def test_a_rollout_score_depends_only_on_the_state_and_settings():
    rollout = Rollout(rollouts=3, turns=2, epsilon=0.25, seed=7)
    for state in random_states(1, 120)[::20]:
        score = rollout(state, 0)
        assert rollout(state, 0) == score
        assert Rollout(rollouts=3, turns=2, epsilon=0.25, seed=7)(state, 0) == score
        assert -SPAN <= score <= SPAN


def test_a_playout_that_wins_scores_a_sure_win():
    # Player 0 has the combat to finish the opponent, which greedy does.
    state = mid_turn(combat=5, trade=0, opponent_authority=5)
    assert rules.winner(state) is None
    rollout = Rollout(rollouts=2, turns=1, epsilon=0)
    assert rollout(state, 0) == SPAN
    assert rollout(state, 1) == -SPAN


def test_a_finished_game_scores_as_evaluate_does():
    state = mid_turn(combat=0, trade=0, opponent_authority=0)
    assert Rollout()(state, 0) == evaluate(state, 0) == WIN
    assert Rollout()(state, 1) == -WIN


def test_an_unfinished_playout_scores_its_win_probability():
    state = random_states(2, 30)[-1]
    rollout = Rollout(rollouts=1, turns=1, epsilon=0)
    end = playout(state, 1, 0, random.Random(0))
    assert rules.winner(end) is None
    assert end.turn == state.turn + 1
    p = 1 / (1 + math.exp(-evaluate(end, 0) / SCALE))
    assert rollout(state, 0) == round(SPAN * (2 * p - 1))


def test_the_heuristic_policy_plays_as_the_heuristic_bot_from_the_movers_view():
    for seed in range(5):
        for state in random_states(seed, 120)[::3]:
            actions = rules.legal_actions(state)
            view = observe(state, state.current)
            assert heuristic(state, actions) == HeuristicBot().choose(view, actions)


def test_a_playout_can_follow_another_policy():
    state = next(  # a state where the policies differ at once
        s
        for s in random_states(2, 200)
        if heuristic(s, rules.legal_actions(s)) != greedy(s, rules.legal_actions(s))
    )
    rng = random.Random(0)
    end = playout(state, 2, 0.0, rng, policy=heuristic)
    expected = state
    while expected.turn < state.turn + 2 and rules.winner(expected) is None:
        actions = rules.legal_actions(expected)
        expected = rules.advance_legal(expected, heuristic(expected, actions))
    assert end == expected
    assert end != playout(state, 2, 0.0, rng)  # greedy plays differently here
