import dataclasses

import pytest

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.action import (
    Acquire,
    AttackPlayer,
    EndTurn,
    PlayCard,
    Scrap,
    UseAlly,
)
from minimax_games.games.star_realms.card import Card, CardType
from minimax_games.games.star_realms.catalogue import (
    CATALOGUE,
    EXPLORER,
    SCOUT,
    SLICE_1_TRADE_DECK,
    TRADE_DECK,
    VIPER,
)
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


def names(cards: tuple[Card, ...]) -> list[str]:
    return sorted(card.name for card in cards)


def test_a_new_game_reveals_five_trade_row_cards_from_the_shuffled_trade_deck():
    state = rules.new_game(seed=1)
    assert len(state.trade_row) == 5
    assert names(state.trade_row + state.trade_deck) == names(SLICE_1_TRADE_DECK)
    assert state.explorers == 10
    other = rules.new_game(seed=2)
    assert state.trade_row + state.trade_deck != other.trade_row + other.trade_deck


def test_the_slice_1_trade_deck_is_every_copy_of_the_slice_1_ships():
    assert all(card.in_slice_1 for card in SLICE_1_TRADE_DECK)
    assert all(card.type is CardType.SHIP for card in SLICE_1_TRADE_DECK)
    assert names(SLICE_1_TRADE_DECK) == names(
        tuple(
            card
            for card in TRADE_DECK
            if card.in_slice_1 and card.type is CardType.SHIP
        )
    )
    assert len(SLICE_1_TRADE_DECK) == 21


CUTTER = CATALOGUE["Cutter"]
CORVETTE = CATALOGUE["Corvette"]
BATTLE_BLOB = CATALOGUE["Battle Blob"]


def test_acquiring_spends_the_cost_puts_the_card_in_the_discard_and_refills_the_row():
    state = rules.new_game(seed=1, trade_deck=(CUTTER,) * 6)
    state = dataclasses.replace(state, trade=5)
    after = rules.apply(state, Acquire(CUTTER))
    assert after.trade == 3
    assert after.players[0].discard == (CUTTER,)
    assert after.trade_row == (CUTTER,) * 5
    assert after.trade_deck == ()


def test_the_row_keeps_fewer_cards_once_the_trade_deck_runs_out():
    state = rules.new_game(seed=1, trade_deck=(CUTTER,) * 5)
    state = dataclasses.replace(state, trade=2)
    after = rules.apply(state, Acquire(CUTTER))
    assert after.trade_row == (CUTTER,) * 4


def test_only_cards_the_player_can_afford_can_be_acquired():
    state = rules.new_game(seed=1, trade_deck=(CUTTER,) * 3 + (BATTLE_BLOB,) * 2)
    acquires = [a for a in rules.legal_actions(state) if isinstance(a, Acquire)]
    assert acquires == []
    state = dataclasses.replace(state, trade=5)
    assert Acquire(CUTTER) in rules.legal_actions(state)
    assert Acquire(BATTLE_BLOB) not in rules.legal_actions(state)
    with pytest.raises(ValueError, match="not legal"):
        rules.apply(state, Acquire(BATTLE_BLOB))
    state = dataclasses.replace(state, trade=6)
    assert Acquire(BATTLE_BLOB) in rules.legal_actions(state)


def test_explorers_come_from_their_pile_at_cost_2_until_it_is_empty():
    state = dataclasses.replace(rules.new_game(seed=1), trade=2)
    after = rules.apply(state, Acquire(EXPLORER))
    assert after.explorers == 9
    assert after.trade == 0
    assert after.players[0].discard == (EXPLORER,)
    assert after.trade_row == state.trade_row
    empty = dataclasses.replace(state, explorers=0)
    assert Acquire(EXPLORER) not in rules.legal_actions(empty)


def with_hand(state: GameState, *hand: Card) -> GameState:
    first = dataclasses.replace(state.players[0], hand=hand)
    return dataclasses.replace(state, players=(first, state.players[1]))


def test_playing_a_ship_adds_its_primary_authority_and_trade():
    state = with_hand(rules.new_game(seed=1), CUTTER)
    after = rules.apply(state, PlayCard(CUTTER))
    assert after.players[0].authority == 54
    assert after.trade == 2
    assert after.combat == 0


def test_playing_a_ship_with_a_primary_draw_draws_from_the_players_deck():
    state = with_hand(rules.new_game(seed=1), CORVETTE)
    deck = state.players[0].deck
    after = rules.apply(state, PlayCard(CORVETTE))
    assert after.combat == 1
    assert after.players[0].hand == deck[:1]
    assert after.players[0].deck == deck[1:]


def test_a_primary_draw_reshuffles_the_discard_pile_when_the_deck_is_empty():
    state = with_hand(rules.new_game(seed=1), CORVETTE)
    first = dataclasses.replace(state.players[0], deck=(), discard=(VIPER, VIPER))
    state = dataclasses.replace(state, players=(first, state.players[1]))
    after = rules.apply(state, PlayCard(CORVETTE))
    assert after.players[0].hand == (VIPER,)
    assert after.players[0].deck == (VIPER,)
    assert after.players[0].discard == ()
    assert after.rng_seed != state.rng_seed


RAM = CATALOGUE["Ram"]


def test_scrapping_a_ship_in_play_applies_its_scrap_effect_and_removes_it():
    state = rules.apply(with_hand(rules.new_game(seed=1), RAM, SCOUT), PlayCard(RAM))
    assert state.combat == 5
    after = rules.apply(state, Scrap(RAM))
    assert after.trade == 3
    assert after.combat == 5
    assert after.players[0].in_play == ()
    assert after.players[0].hand == (SCOUT,)
    assert after.scrap_heap == (RAM,)
    ended = rules.apply(after, EndTurn()).players[0]
    assert RAM not in ended.discard + ended.deck + ended.hand


def test_a_scrapped_explorer_goes_back_to_the_explorer_pile():
    state = dataclasses.replace(rules.new_game(seed=1), explorers=9)
    state = rules.apply(with_hand(state, EXPLORER), PlayCard(EXPLORER))
    assert state.trade == 2
    after = rules.apply(state, Scrap(EXPLORER))
    assert after.combat == 2
    assert after.explorers == 10
    assert after.scrap_heap == ()
    assert after.players[0].in_play == ()


def scraps(state: GameState) -> list[Scrap]:
    return [a for a in rules.legal_actions(state) if isinstance(a, Scrap)]


def test_only_cards_in_play_with_a_scrap_ability_can_be_scrapped():
    state = with_hand(rules.new_game(seed=1), RAM, SCOUT, CUTTER, EXPLORER)
    assert scraps(state) == []
    for card in (SCOUT, CUTTER):
        state = rules.apply(state, PlayCard(card))
    assert scraps(state) == []
    with pytest.raises(ValueError, match="not legal"):
        rules.apply(state, Scrap(SCOUT))
    state = rules.apply(state, PlayCard(RAM))
    assert scraps(state) == [Scrap(RAM)]
    state = rules.apply(state, PlayCard(EXPLORER))
    assert scraps(state) == [Scrap(RAM), Scrap(EXPLORER)]


def test_each_copy_in_play_can_be_scrapped_once():
    state = with_hand(rules.new_game(seed=1), RAM, RAM)
    state = rules.apply(rules.apply(state, PlayCard(RAM)), PlayCard(RAM))
    state = rules.apply(state, Scrap(RAM))
    assert state.players[0].in_play == (RAM,)
    state = rules.apply(state, Scrap(RAM))
    assert state.trade == 6
    assert state.scrap_heap == (RAM, RAM)
    assert scraps(state) == []


def test_scrap_abilities_with_effects_not_yet_implemented_are_unavailable():
    frigate = CATALOGUE["Imperial Frigate"]  # scrap: draw a card
    survey = CATALOGUE["Survey Ship"]  # scrap: target opponent discards a card
    state = rules.new_game(seed=1)
    first = dataclasses.replace(state.players[0], hand=(), in_play=(frigate, survey))
    state = dataclasses.replace(state, players=(first, state.players[1]))
    assert scraps(state) == [Scrap(frigate)]


SHUTTLE = CATALOGUE["Federation Shuttle"]


def allies(state: GameState) -> list[UseAlly]:
    return [a for a in rules.legal_actions(state) if isinstance(a, UseAlly)]


def test_an_ally_ability_is_usable_once_another_card_of_its_faction_is_in_play():
    state = with_hand(rules.new_game(seed=1), CUTTER, SCOUT, RAM, SHUTTLE)
    for card in (CUTTER, SCOUT, RAM):
        state = rules.apply(state, PlayCard(card))
    assert allies(state) == []
    with pytest.raises(ValueError, match="not legal"):
        rules.apply(state, UseAlly(CUTTER))
    state = rules.apply(state, PlayCard(SHUTTLE))
    assert allies(state) == [UseAlly(CUTTER), UseAlly(SHUTTLE)]
    combat = state.combat
    after = rules.apply(state, UseAlly(CUTTER))
    assert after.combat == combat + 4
    assert after.players[0].in_play == state.players[0].in_play


def test_an_ally_ability_can_be_used_only_once_per_turn():
    state = with_hand(rules.new_game(seed=1), CUTTER, SHUTTLE)
    state = rules.apply(rules.apply(state, PlayCard(CUTTER)), PlayCard(SHUTTLE))
    state = rules.apply(state, UseAlly(CUTTER))
    assert allies(state) == [UseAlly(SHUTTLE)]
    with pytest.raises(ValueError, match="not legal"):
        rules.apply(state, UseAlly(CUTTER))


def test_two_copies_ally_each_other_and_each_uses_its_ally_ability_once():
    state = with_hand(rules.new_game(seed=1), CUTTER, CUTTER)
    state = rules.apply(state, PlayCard(CUTTER))
    assert allies(state) == []
    state = rules.apply(state, PlayCard(CUTTER))
    state = rules.apply(state, UseAlly(CUTTER))
    assert allies(state) == [UseAlly(CUTTER)]
    state = rules.apply(state, UseAlly(CUTTER))
    assert state.combat == 8
    assert allies(state) == []


def test_ally_abilities_are_available_again_next_turn():
    state = with_hand(rules.new_game(seed=1), CUTTER, SHUTTLE)
    state = rules.apply(rules.apply(state, PlayCard(CUTTER)), PlayCard(SHUTTLE))
    state = rules.apply(rules.apply(state, UseAlly(CUTTER)), UseAlly(SHUTTLE))
    state = rules.apply(rules.apply(state, EndTurn()), EndTurn())
    assert state.current == 0
    state = with_hand(state, CUTTER, SHUTTLE)
    state = rules.apply(rules.apply(state, PlayCard(CUTTER)), PlayCard(SHUTTLE))
    assert allies(state) == [UseAlly(CUTTER), UseAlly(SHUTTLE)]


TRADE_POD = CATALOGUE["Trade Pod"]


def test_a_scrapped_card_no_longer_allies_the_cards_still_in_play():
    state = with_hand(rules.new_game(seed=1), RAM, TRADE_POD)
    state = rules.apply(rules.apply(state, PlayCard(RAM)), PlayCard(TRADE_POD))
    assert allies(state) == [UseAlly(RAM), UseAlly(TRADE_POD)]
    state = rules.apply(state, Scrap(RAM))
    assert allies(state) == []


def test_an_ally_ability_used_before_scrapping_its_card_keeps_its_effect():
    state = with_hand(rules.new_game(seed=1), RAM, TRADE_POD)
    state = rules.apply(rules.apply(state, PlayCard(RAM)), PlayCard(TRADE_POD))
    state = rules.apply(rules.apply(state, UseAlly(RAM)), Scrap(RAM))
    assert state.combat == 5 + 2
    assert state.trade == 3 + 3


def test_scrapping_one_of_two_copies_scraps_the_one_whose_ally_was_used():
    state = with_hand(rules.new_game(seed=1), RAM, RAM, TRADE_POD)
    for card in (RAM, RAM, TRADE_POD):
        state = rules.apply(state, PlayCard(card))
    state = rules.apply(rules.apply(state, UseAlly(RAM)), Scrap(RAM))
    assert allies(state) == [UseAlly(RAM), UseAlly(TRADE_POD)]
    state = rules.apply(state, UseAlly(RAM))
    assert allies(state) == [UseAlly(TRADE_POD)]


def test_ally_abilities_with_effects_not_yet_implemented_are_unavailable():
    destroyer = CATALOGUE["Blob Destroyer"]  # ally: destroy base, scrap Trade Row
    frigate = CATALOGUE["Imperial Frigate"]  # ally: 2 combat
    corvette = CATALOGUE["Corvette"]  # ally: 2 combat
    state = rules.new_game(seed=1)
    in_play = (destroyer, RAM, frigate, corvette, SCOUT, SCOUT)
    first = dataclasses.replace(state.players[0], hand=(), in_play=in_play)
    state = dataclasses.replace(state, players=(first, state.players[1]))
    assert allies(state) == [UseAlly(RAM), UseAlly(frigate), UseAlly(corvette)]
