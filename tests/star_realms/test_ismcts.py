import random
from dataclasses import replace

import pytest

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.action import AttackPlayer, EndTurn
from minimax_games.games.star_realms.bots import GreedyBot
from minimax_games.games.star_realms.catalogue import CATALOGUE, STARTING_DECK
from minimax_games.games.star_realms.ismcts import ITERATIONS, IsmctsBot, StarRealms
from minimax_games.games.star_realms.match import MatchTooLong, play_match
from minimax_games.games.star_realms.rollout import (
    SCALE,
    heuristic,
    playout,
    win_probability,
)
from minimax_games.games.star_realms.state import GameState, PlayerState
from minimax_games.games.star_realms.view import observe


def lethal() -> GameState:
    """Player 0 has the combat to finish the opponent, and trade to buy
    with instead: the sampler's obvious winning attack."""
    deck = STARTING_DECK + tuple(CATALOGUE[n] for n in ("Blob Fighter", "Cutter"))
    me = PlayerState(authority=50, deck=deck)
    them = PlayerState(authority=5, hand=deck[:5], deck=deck[5:])
    row = tuple(CATALOGUE[name] for name in ("Cutter", "Corvette", "Battle Blob"))
    trade_deck = tuple(CATALOGUE[n] for n in ("Blob Wheel", "Federation Shuttle"))
    return GameState(
        players=(me, them),
        rng_seed=0,
        trade_deck=trade_deck,
        trade_row=row,
        trade=6,
        combat=5,
    )


def test_it_takes_an_obvious_winning_attack_whatever_the_hidden_cards():
    state = lethal()
    actions = rules.legal_actions(state)
    assert AttackPlayer() in actions
    assert len(actions) > 2  # buys too: the attack is not the only option
    for seed in range(3):
        bot = IsmctsBot(seed, iterations=50)
        assert bot.choose(observe(state, 0), actions) == AttackPlayer()


class NoWorlds(IsmctsBot):
    """Fails if it searches at all."""

    def choose(self, view, actions):
        self.game = None  # type: ignore[assignment]
        return super().choose(view, actions)


def test_a_lone_legal_action_is_played_without_searching():
    state = replace(lethal(), trade=0, combat=0)
    view = observe(state, 0)
    assert NoWorlds(0).choose(view, (EndTurn(),)) == EndTurn()


def test_it_states_its_settings():
    bot = IsmctsBot(0, iterations=120, exploration=0.35, epsilon=0.25, extra_turns=1)
    assert bot.settings() == (
        "120 iterations per decision, exploration 0.35,"
        " leaf: epsilon-greedy playout to the end of the turn + 1 more turn"
        " (epsilon 0.25), scale 16"
    )
    assert IsmctsBot(0).settings() == (
        f"{ITERATIONS} iterations per decision, exploration 0.7,"
        " leaf: epsilon-greedy playout to the end of the turn (epsilon 0.1), scale 16"
    )


def test_a_leaf_is_player_0s_win_probability():
    game = StarRealms(turns=1, epsilon=0.1)
    rng = random.Random(0)
    state = lethal()
    won = rules.advance_legal(state, AttackPlayer())
    assert game.is_terminal(won)
    assert game.value(won, rng) == 1.0
    lost = replace(won, players=won.players[::-1])
    assert game.value(lost, rng) == 0.0
    early = rules.auto_play(rules.new_game(3))
    assert 0 < game.value(early, rng) < 1


def test_it_needs_an_iteration_and_no_negative_extra_turns():
    with pytest.raises(ValueError, match="at least 1 iteration"):
        IsmctsBot(0, iterations=0)
    with pytest.raises(ValueError, match="can't be negative"):
        IsmctsBot(0, extra_turns=-1)


def opening(seed: int, turns: int = 4) -> list[str]:
    """The transcript of the first `turns` turns of a seeded ISMCTS-vs-greedy
    game at a small budget. The match runner rejects illegal moves."""
    lines: list[str] = []
    bot = IsmctsBot(seed, iterations=10)
    with pytest.raises(MatchTooLong):
        play_match((bot, GreedyBot()), seed, max_turns=turns, log=lines.append)
    return lines


def test_a_seeded_game_replays_identically():
    assert opening(5) == opening(5)
    assert opening(5) != opening(6)


def test_with_a_horizon_every_leaf_is_played_out_to_the_same_turn():
    start = rules.auto_play(rules.new_game(3))
    game = StarRealms(turns=1, epsilon=0.1, until=start.turn + 2)

    def leaf(state: GameState, turns: int) -> float:
        end = playout(state, turns, 0.1, random.Random(1))
        return win_probability(end, 0, SCALE)

    # From the root's turn, the leaf plays this turn and the next...
    assert game.value(start, random.Random(1)) == leaf(start, 2)
    # ...and from a later turn, at least to the end of its own.
    later = replace(start, turn=start.turn + 5)
    assert game.value(later, random.Random(1)) == leaf(later, 1)


def test_with_a_horizon_it_states_it_and_still_takes_a_winning_attack():
    bot = IsmctsBot(0, iterations=50, horizon=2)
    assert bot.settings() == (
        "50 iterations per decision, exploration 0.7,"
        " leaf: epsilon-greedy playout to the end of the 2nd turn from the root,"
        " at least to the end of its own (epsilon 0.1), scale 16"
    )
    state = lethal()
    assert bot.choose(observe(state, 0), rules.legal_actions(state)) == AttackPlayer()
    with pytest.raises(ValueError, match="at least 1 turn"):
        IsmctsBot(0, horizon=0)


def test_a_leaf_can_be_played_out_by_the_heuristic():
    start = rules.auto_play(rules.new_game(3))
    game = StarRealms(turns=1, epsilon=0.1, policy=heuristic)
    end = playout(start, 1, 0.1, random.Random(2), policy=heuristic)
    assert game.value(start, random.Random(2)) == win_probability(end, 0, SCALE)
    bot = IsmctsBot(0, iterations=50, policy="heuristic")
    assert bot.settings() == (
        "50 iterations per decision, exploration 0.7,"
        " leaf: epsilon-heuristic playout to the end of the turn (epsilon 0.1),"
        " scale 16"
    )
    state = lethal()
    assert bot.choose(observe(state, 0), rules.legal_actions(state)) == AttackPlayer()
    with pytest.raises(ValueError, match="greedy or heuristic"):
        IsmctsBot(0, policy="random")
