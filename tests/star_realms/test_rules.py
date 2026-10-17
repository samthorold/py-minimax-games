import dataclasses

import pytest

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.action import AttackPlayer, EndTurn, PlayCard
from minimax_games.games.star_realms.catalogue import SCOUT, VIPER
from minimax_games.games.star_realms.state import GameState


def test_opening_hands_are_three_for_first_player_and_five_for_second():
    state = rules.new_game(seed=1)
    first, second = state.players
    assert len(first.hand) == 3
    assert len(second.hand) == 5
    assert len(first.hand) + len(first.deck) == 10
    assert len(second.hand) + len(second.deck) == 10
    assert first.authority == second.authority == 50


def play_whole_hand(state: GameState) -> GameState:
    while plays := [a for a in rules.legal_actions(state) if isinstance(a, PlayCard)]:
        state = rules.apply(state, plays[0])
    return state


def test_playing_cards_moves_them_into_play_and_adds_their_trade_and_combat():
    state = rules.new_game(seed=1)
    hand = state.players[0].hand
    after = play_whole_hand(state)
    assert after.players[0].hand == ()
    assert sorted(c.name for c in after.players[0].in_play) == sorted(
        c.name for c in hand
    )
    assert after.trade == hand.count(SCOUT)
    assert after.combat == hand.count(VIPER)


def test_attacking_the_opponent_spends_all_combat_on_their_authority():
    state = dataclasses.replace(rules.new_game(seed=1), combat=3)
    after = rules.apply(state, AttackPlayer())
    assert after.players[1].authority == 47
    assert after.combat == 0
    assert AttackPlayer() not in rules.legal_actions(after)
    assert AttackPlayer() in rules.legal_actions(state)


def test_ending_the_turn_loses_unspent_trade_and_combat_and_draws_five():
    state = play_whole_hand(rules.new_game(seed=1))
    state = dataclasses.replace(state, trade=2, combat=4)
    after = rules.apply(state, EndTurn())
    assert after.trade == after.combat == 0
    assert after.players[1].authority == 50
    assert after.current == 1
    assert after.turn == 2
    ended = after.players[0]
    assert ended.in_play == ()
    assert len(ended.discard) == 3
    assert len(ended.hand) == 5
    assert len(ended.deck) == 2


def test_the_discard_pile_is_reshuffled_into_the_deck_when_it_runs_out():
    state = rules.new_game(seed=1)
    for _ in range(3):  # first player, second player, first player again
        state = rules.apply(state, EndTurn())
    first = state.players[0]
    # 2 cards were left in the deck, so 3 more came from the reshuffled 8.
    assert len(first.hand) == 5
    assert len(first.deck) == 5
    assert first.discard == ()


def test_a_player_whose_authority_reaches_zero_loses():
    state = rules.new_game(seed=1)
    assert rules.winner(state) is None
    second = dataclasses.replace(state.players[1], authority=2)
    state = dataclasses.replace(state, players=(state.players[0], second), combat=2)
    after = rules.apply(state, AttackPlayer())
    assert rules.winner(after) == 0
    assert rules.legal_actions(after) == ()


def test_an_illegal_action_is_refused():
    with pytest.raises(ValueError, match="not legal"):
        rules.apply(rules.new_game(seed=1), AttackPlayer())
