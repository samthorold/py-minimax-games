import dataclasses
import random
from collections import Counter

import pytest

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.action import (
    Acquire,
    AttackBase,
    AttackPlayer,
    DestroyTarget,
    Discard,
    EndTurn,
    PlayCard,
    Scrap,
    ScrapDiscardCard,
    ScrapHandCard,
    UseAlly,
    UseBase,
)
from minimax_games.games.star_realms.card import Card, CardType
from minimax_games.games.star_realms.catalogue import (
    CATALOGUE,
    EXPLORER,
    PLAYABLE_TRADE_DECK,
    SCOUT,
    TRADE_DECK,
    VIPER,
    playable_deck,
)
from minimax_games.games.star_realms.effect import (
    Authority,
    Combat,
    Draw,
    OpponentDiscards,
    ScrapFromHand,
    ScrapHandOrDiscard,
    ScrapTradeRow,
    Trade,
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
    assert names(state.trade_row + state.trade_deck) == names(PLAYABLE_TRADE_DECK)
    assert state.explorers == 10
    other = rules.new_game(seed=2)
    assert state.trade_row + state.trade_deck != other.trade_row + other.trade_deck


SLICE_1_EFFECTS = frozenset({Trade, Combat, Authority, Draw})


def test_with_only_the_first_slice_effects_the_playable_deck_is_the_first_slice():
    deck = playable_deck(SLICE_1_EFFECTS)
    assert names(deck) == names(
        tuple(
            card
            for card in TRADE_DECK
            if all(
                isinstance(e, Trade | Combat | Authority | Draw)
                for e in card.primary + card.ally + card.scrap
            )
        )
    )
    assert len(deck) == 21 + 9
    bases = {c.name for c in deck if c.type is not CardType.SHIP}
    assert bases == {
        "Blob Wheel",
        "The Hive",
        "Space Station",
        "War World",
        "Battle Station",
    }


SLICE_2_EFFECTS = SLICE_1_EFFECTS | {OpponentDiscards}


def test_the_playable_deck_adds_the_opponent_discard_cards_to_the_first_slice():
    deck = playable_deck(SLICE_2_EFFECTS)
    added = Counter(deck) - Counter(playable_deck(SLICE_1_EFFECTS))
    assert {card.name: n for card, n in added.items()} == {
        "Imperial Fighter": 3,
        "Imperial Frigate": 3,
        "Survey Ship": 3,
        "Royal Redoubt": 1,
    }
    # Battlecruiser's scrap ability also destroys a base, not implemented yet.
    assert CATALOGUE["Battlecruiser"] not in deck
    assert len(deck) == 40


SLICE_3_EFFECTS = SLICE_2_EFFECTS | {ScrapHandOrDiscard, ScrapFromHand}


def test_the_playable_deck_adds_the_scrap_from_hand_or_discard_cards():
    deck = playable_deck(SLICE_3_EFFECTS)
    added = Counter(deck) - Counter(playable_deck(SLICE_2_EFFECTS))
    assert {card.name: n for card, n in added.items()} == {
        "Trade Bot": 3,
        "Missile Bot": 3,
        "Supply Bot": 3,
        "Battle Mech": 1,
        "Junkyard": 1,
        "Machine Base": 1,
    }
    # Patrol Mech's primary is an "or" choice, not implemented yet.
    assert CATALOGUE["Patrol Mech"] not in deck
    assert len(deck) == 52


def test_the_playable_deck_adds_the_destroy_base_cards():
    added = Counter(PLAYABLE_TRADE_DECK) - Counter(playable_deck(SLICE_3_EFFECTS))
    assert {card.name: n for card, n in added.items()} == {
        "Command Ship": 1,
        "Port of Call": 1,
        "Battlecruiser": 1,
        "Missile Mech": 1,
    }
    # Blob Destroyer's ally ability also scraps from the Trade Row.
    assert CATALOGUE["Blob Destroyer"] not in PLAYABLE_TRADE_DECK
    assert len(PLAYABLE_TRADE_DECK) == 56


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
    # Every base-set scrap ability is implemented now, so make one that isn't.
    culler = dataclasses.replace(frigate, name="Culler", scrap=(ScrapTradeRow(1),))
    state = rules.new_game(seed=1)
    in_play = (frigate, culler)
    first = dataclasses.replace(state.players[0], hand=(), in_play=in_play)
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


BLOB_WHEEL = CATALOGUE["Blob Wheel"]
THE_HIVE = CATALOGUE["The Hive"]
SPACE_STATION = CATALOGUE["Space Station"]
WAR_WORLD = CATALOGUE["War World"]
BATTLE_STATION = CATALOGUE["Battle Station"]


def test_a_played_base_applies_its_primary_and_stays_in_play_after_the_turn():
    state = with_hand(rules.new_game(seed=1), BLOB_WHEEL, SCOUT)
    state = rules.apply(rules.apply(state, PlayCard(BLOB_WHEEL)), PlayCard(SCOUT))
    assert state.combat == 1
    assert state.players[0].bases == (BLOB_WHEEL,)
    assert state.players[0].in_play == (SCOUT,)
    ended = rules.apply(state, EndTurn()).players[0]
    assert ended.bases == (BLOB_WHEEL,)
    assert BLOB_WHEEL not in ended.discard + ended.hand + ended.deck


def with_bases(state: GameState, *bases: Card) -> GameState:
    first = dataclasses.replace(state.players[0], hand=(), bases=bases)
    return dataclasses.replace(state, players=(first, state.players[1]))


def base_uses(state: GameState) -> list[UseBase]:
    return [a for a in rules.legal_actions(state) if isinstance(a, UseBase)]


def test_a_base_primary_is_used_once_per_turn_on_later_turns():
    state = with_hand(rules.new_game(seed=1), BLOB_WHEEL)
    state = rules.apply(state, PlayCard(BLOB_WHEEL))
    assert base_uses(state) == []
    state = rules.apply(rules.apply(state, EndTurn()), EndTurn())
    assert base_uses(state) == [UseBase(BLOB_WHEEL)]
    after = rules.apply(state, UseBase(BLOB_WHEEL))
    assert after.combat == state.combat + 1
    assert after.players[0].bases == (BLOB_WHEEL,)
    assert base_uses(after) == []
    with pytest.raises(ValueError, match="not legal"):
        rules.apply(after, UseBase(BLOB_WHEEL))
    state = rules.apply(rules.apply(after, EndTurn()), EndTurn())
    assert base_uses(state) == [UseBase(BLOB_WHEEL)]


def test_each_copy_of_a_base_uses_its_primary_once():
    state = with_bases(rules.new_game(seed=1), WAR_WORLD, WAR_WORLD)
    state = rules.apply(state, UseBase(WAR_WORLD))
    assert base_uses(state) == [UseBase(WAR_WORLD)]
    state = rules.apply(state, UseBase(WAR_WORLD))
    assert state.combat == 6
    assert base_uses(state) == []


def test_a_base_without_a_primary_ability_has_nothing_to_use():
    state = with_bases(rules.new_game(seed=1), BATTLE_STATION)
    assert base_uses(state) == []


def test_bases_ally_ships_and_use_their_own_ally_abilities():
    state = with_bases(rules.new_game(seed=1), THE_HIVE)
    assert allies(state) == []
    state = rules.apply(with_hand(state, BLOB_WHEEL, RAM), PlayCard(RAM))
    assert allies(state) == [UseAlly(RAM), UseAlly(THE_HIVE)]
    hand = state.players[0].hand
    after = rules.apply(state, UseAlly(THE_HIVE))
    assert len(after.players[0].hand) == len(hand) + 1
    assert allies(after) == [UseAlly(RAM)]


def test_a_base_played_on_a_later_turn_allies_a_base_already_in_play():
    state = with_bases(rules.new_game(seed=1), SPACE_STATION)
    state = rules.apply(with_hand(state, WAR_WORLD), PlayCard(WAR_WORLD))
    assert allies(state) == [UseAlly(SPACE_STATION), UseAlly(WAR_WORLD)]


def test_scrapping_a_base_applies_its_scrap_ability_and_removes_it():
    state = with_bases(rules.new_game(seed=1), BATTLE_STATION, SPACE_STATION)
    assert scraps(state) == [Scrap(BATTLE_STATION), Scrap(SPACE_STATION)]
    state = rules.apply(state, Scrap(BATTLE_STATION))
    assert state.combat == 5
    assert state.players[0].bases == (SPACE_STATION,)
    assert state.scrap_heap == (BATTLE_STATION,)


def test_scrapping_one_of_two_base_copies_scraps_the_one_already_used():
    state = with_bases(rules.new_game(seed=1), SPACE_STATION, SPACE_STATION)
    state = rules.apply(
        rules.apply(state, UseBase(SPACE_STATION)), Scrap(SPACE_STATION)
    )
    assert state.players[0].bases == (SPACE_STATION,)
    assert base_uses(state) == [UseBase(SPACE_STATION)]


def with_opponent_bases(state: GameState, *bases: Card, combat: int) -> GameState:
    second = dataclasses.replace(state.players[1], bases=bases)
    return dataclasses.replace(state, players=(state.players[0], second), combat=combat)


def attacks(state: GameState) -> list[AttackBase | AttackPlayer]:
    kinds = AttackBase | AttackPlayer
    return [a for a in rules.legal_actions(state) if isinstance(a, kinds)]


def test_destroying_a_base_spends_its_defense_and_puts_it_in_its_owners_discard():
    state = with_opponent_bases(rules.new_game(seed=1), THE_HIVE, combat=7)
    discard = state.players[1].discard
    after = rules.apply(state, AttackBase(THE_HIVE))
    assert after.combat == 2
    assert after.players[1].bases == ()
    assert after.players[1].discard == (*discard, THE_HIVE)
    assert after.players[1].authority == 50


def test_a_base_needs_combat_at_least_equal_to_its_defense():
    state = with_opponent_bases(rules.new_game(seed=1), THE_HIVE, combat=4)
    assert attacks(state) == [AttackPlayer()]
    with pytest.raises(ValueError, match="not legal"):
        rules.apply(state, AttackBase(THE_HIVE))
    state = dataclasses.replace(state, combat=5)
    assert attacks(state) == [AttackBase(THE_HIVE), AttackPlayer()]


def test_while_an_outpost_is_in_play_only_outposts_can_be_attacked():
    state = rules.new_game(seed=1)
    state = with_opponent_bases(state, THE_HIVE, WAR_WORLD, combat=20)
    assert attacks(state) == [AttackBase(WAR_WORLD)]
    with pytest.raises(ValueError, match="not legal"):
        rules.apply(state, AttackPlayer())
    state = rules.apply(state, AttackBase(WAR_WORLD))
    assert attacks(state) == [AttackBase(THE_HIVE), AttackPlayer()]


def test_two_outposts_must_each_be_destroyed_before_the_player_can_be_attacked():
    state = rules.new_game(seed=1)
    state = with_opponent_bases(state, WAR_WORLD, WAR_WORLD, combat=8)
    state = rules.apply(state, AttackBase(WAR_WORLD))
    assert attacks(state) == [AttackBase(WAR_WORLD)]
    state = rules.apply(state, AttackBase(WAR_WORLD))
    assert state.players[1].discard[-2:] == (WAR_WORLD, WAR_WORLD)
    assert attacks(state) == []


def test_advance_legal_matches_advance_for_every_legal_action():
    rng = random.Random(0)
    for seed in range(10):
        state = rules.auto_play(rules.new_game(seed))
        for _ in range(40):
            actions = rules.legal_actions(state)
            if not actions:
                break
            for action in actions:
                assert rules.advance_legal(state, action) == rules.advance(
                    state, action
                )
            state = rules.advance(state, rng.choice(actions))


def test_the_rules_build_states_without_dataclasses_replace(monkeypatch):
    # It introspects the fields on every call: too slow for the search.
    def refuse(*args, **kwargs):
        raise AssertionError("dataclasses.replace called")

    monkeypatch.setattr(dataclasses, "replace", refuse)
    rng = random.Random(0)
    for seed in range(5):
        state = rules.auto_play(rules.new_game(seed))
        for _ in range(200):
            actions = rules.legal_actions(state)
            if not actions:
                break
            state = rules.advance(state, rng.choice(actions))


IMPERIAL_FIGHTER = CATALOGUE["Imperial Fighter"]  # 2 combat, opponent discards 1
IMPERIAL_FRIGATE = CATALOGUE["Imperial Frigate"]  # 4 combat, opponent discards 1


def discards(state: GameState) -> list[Discard]:
    return [a for a in rules.legal_actions(state) if isinstance(a, Discard)]


def test_the_opponent_discards_a_card_of_their_choice_at_the_start_of_their_turn():
    state = with_hand(rules.new_game(seed=1), IMPERIAL_FIGHTER)
    state = rules.apply(state, PlayCard(IMPERIAL_FIGHTER))
    assert state.combat == 2
    assert state.players[1].to_discard == 1
    state = rules.apply(state, EndTurn())
    hand = state.players[1].hand
    assert len(hand) == 5
    # Discarding comes first: it is the only thing they can do.
    assert rules.legal_actions(state) == tuple(Discard(c) for c in dict.fromkeys(hand))
    card = hand[0]
    after = rules.apply(state, Discard(card))
    second = after.players[1]
    assert second.to_discard == 0
    assert names(second.hand) == names(hand[1:])
    assert second.discard == (*state.players[1].discard, card)
    assert discards(after) == []
    assert EndTurn() in rules.legal_actions(after)


def with_opponent_hand(state: GameState, *hand: Card) -> GameState:
    second = dataclasses.replace(state.players[1], hand=hand)
    return dataclasses.replace(state, players=(state.players[0], second))


def test_discards_stack_one_per_effect():
    state = with_hand(rules.new_game(seed=1), IMPERIAL_FIGHTER, IMPERIAL_FRIGATE)
    for card in (IMPERIAL_FIGHTER, IMPERIAL_FRIGATE):
        state = rules.apply(state, PlayCard(card))
    state = rules.apply(state, EndTurn())
    assert state.players[1].to_discard == 2
    state = rules.apply(state, discards(state)[0])
    assert len(discards(state)) > 0
    state = rules.apply(state, discards(state)[0])
    assert len(state.players[1].hand) == 3
    assert discards(state) == []


def test_a_player_discards_no_more_cards_than_they_hold():
    for hand, made in (((), 0), ((SCOUT,), 1)):
        state = with_hand(rules.new_game(seed=1), IMPERIAL_FIGHTER, IMPERIAL_FRIGATE)
        state = with_opponent_hand(state, *hand)
        for card in (IMPERIAL_FIGHTER, IMPERIAL_FRIGATE):
            state = rules.apply(state, PlayCard(card))
        state = rules.apply(state, EndTurn())
        assert state.players[1].to_discard == made
        for _ in range(made):
            state = rules.apply(state, discards(state)[0])
        assert state.players[1].hand == ()
        assert EndTurn() in rules.legal_actions(state)


TRADE_BOT = CATALOGUE["Trade Bot"]  # Machine Cult: 1 trade, may scrap hand/discard
JUNKYARD = CATALOGUE["Junkyard"]  # outpost: may scrap from hand or discard pile
MACHINE_BASE = CATALOGUE["Machine Base"]  # outpost: draw, then scrap from hand


def hand_or_discard_scraps(
    state: GameState,
) -> list[ScrapHandCard | ScrapDiscardCard]:
    return [
        a
        for a in rules.legal_actions(state)
        if isinstance(a, ScrapHandCard | ScrapDiscardCard)
    ]


def with_discard(state: GameState, *discard: Card) -> GameState:
    first = dataclasses.replace(state.players[0], discard=discard)
    return dataclasses.replace(state, players=(first, state.players[1]))


def test_a_scrap_ability_lets_the_player_scrap_a_card_from_their_discard_pile():
    state = with_discard(with_hand(rules.new_game(seed=1), TRADE_BOT), VIPER)
    state = rules.apply(state, PlayCard(TRADE_BOT))
    assert state.trade == 1
    state = rules.apply(state, ScrapDiscardCard(VIPER))
    assert state.players[0].discard == ()
    assert state.scrap_heap == (VIPER,)


def test_machine_base_draws_then_must_scrap_a_card_from_hand_before_anything_else():
    state = with_hand(rules.new_game(seed=1), MACHINE_BASE, SCOUT)
    deck = state.players[0].deck
    state = rules.apply(state, PlayCard(MACHINE_BASE))
    assert state.players[0].hand == (SCOUT, deck[0])
    assert rules.legal_actions(state) == tuple(
        ScrapHandCard(c) for c in dict.fromkeys((SCOUT, deck[0]))
    )
    state = rules.apply(state, ScrapHandCard(SCOUT))
    assert state.players[0].hand == (deck[0],)
    assert state.scrap_heap == (SCOUT,)
    assert hand_or_discard_scraps(state) == []
    assert EndTurn() in rules.legal_actions(state)


def test_each_scrap_ability_scraps_at_most_one_card_and_skipping_it_is_allowed():
    state = with_hand(rules.new_game(seed=1), TRADE_BOT, VIPER, CORVETTE)
    state = with_discard(state, SCOUT, SCOUT)
    assert hand_or_discard_scraps(state) == []
    state = rules.apply(state, PlayCard(TRADE_BOT))
    assert set(hand_or_discard_scraps(state)) == {
        ScrapHandCard(VIPER),
        ScrapHandCard(CORVETTE),
        ScrapDiscardCard(SCOUT),
    }
    # The ability waits: playing other cards first doesn't use it up.
    state = rules.apply(state, PlayCard(VIPER))
    state = rules.apply(state, ScrapHandCard(CORVETTE))
    assert state.players[0].hand == ()
    assert state.scrap_heap == (CORVETTE,)
    assert hand_or_discard_scraps(state) == []


def test_a_scrap_ability_left_unused_is_lost_at_the_end_of_the_turn():
    state = with_discard(with_hand(rules.new_game(seed=1), TRADE_BOT), SCOUT)
    state = rules.apply(rules.apply(state, PlayCard(TRADE_BOT)), EndTurn())
    assert state.may_scrap == 0
    assert hand_or_discard_scraps(state) == []
    assert state.scrap_heap == ()


def test_an_explorer_scrapped_from_the_discard_pile_goes_back_to_its_pile():
    state = with_discard(with_hand(rules.new_game(seed=1), TRADE_BOT), EXPLORER)
    explorers = state.explorers
    state = rules.apply(
        rules.apply(state, PlayCard(TRADE_BOT)), ScrapDiscardCard(EXPLORER)
    )
    assert state.explorers == explorers + 1
    assert state.players[0].discard == ()
    assert state.scrap_heap == ()


def test_machine_base_owes_no_scrap_when_there_is_nothing_to_draw_or_scrap():
    state = with_hand(rules.new_game(seed=1), MACHINE_BASE)
    first = dataclasses.replace(state.players[0], deck=())
    state = dataclasses.replace(state, players=(first, state.players[1]))
    state = rules.apply(state, PlayCard(MACHINE_BASE))
    assert state.players[0].hand == ()
    assert state.must_scrap == 0
    assert EndTurn() in rules.legal_actions(state)


MISSILE_MECH = CATALOGUE["Missile Mech"]  # 6 combat, may destroy target base
COMMAND_SHIP = CATALOGUE["Command Ship"]  # ally: may destroy target base
PORT_OF_CALL = CATALOGUE["Port of Call"]  # outpost; scrap: draw, destroy base
BATTLECRUISER = CATALOGUE["Battlecruiser"]  # scrap: draw, destroy base


def destroys(state: GameState) -> list[DestroyTarget]:
    return [a for a in rules.legal_actions(state) if isinstance(a, DestroyTarget)]


def test_destroy_target_base_ignores_outposts_and_spends_no_combat():
    state = with_hand(rules.new_game(seed=1), MISSILE_MECH)
    state = with_opponent_bases(state, THE_HIVE, WAR_WORLD, combat=0)
    discard = state.players[1].discard
    assert destroys(state) == []
    state = rules.apply(state, PlayCard(MISSILE_MECH))
    assert destroys(state) == [DestroyTarget(THE_HIVE), DestroyTarget(WAR_WORLD)]
    after = rules.apply(state, DestroyTarget(THE_HIVE))
    assert after.combat == 6
    assert after.players[1].bases == (WAR_WORLD,)
    assert after.players[1].discard == (*discard, THE_HIVE)
    assert destroys(after) == []


def test_destroy_target_base_does_nothing_when_the_opponent_has_no_bases():
    state = with_hand(rules.new_game(seed=1), MISSILE_MECH)
    state = rules.apply(state, PlayCard(MISSILE_MECH))
    assert state.may_destroy == 0
    assert destroys(state) == []


def test_destroying_is_optional_and_lost_at_the_end_of_the_turn():
    state = with_hand(rules.new_game(seed=1), MISSILE_MECH)
    state = with_opponent_bases(state, THE_HIVE, combat=0)
    state = rules.apply(state, PlayCard(MISSILE_MECH))
    assert EndTurn() in rules.legal_actions(state)
    state = rules.apply(state, EndTurn())
    assert state.may_destroy == 0
    assert state.players[1].bases == (THE_HIVE,)


def test_battlecruiser_and_port_of_call_scrap_abilities_draw_and_destroy_a_base():
    state = rules.new_game(seed=1)
    first = dataclasses.replace(
        state.players[0],
        hand=(),
        in_play=(BATTLECRUISER,),
        bases=(PORT_OF_CALL,),
    )
    state = dataclasses.replace(state, players=(first, state.players[1]))
    state = with_opponent_bases(state, WAR_WORLD, THE_HIVE, BLOB_WHEEL, combat=0)
    state = rules.apply(state, Scrap(BATTLECRUISER))
    assert len(state.players[0].hand) == 1
    state = rules.apply(state, Scrap(PORT_OF_CALL))
    assert len(state.players[0].hand) == 2
    assert state.may_destroy == 2
    state = rules.apply(state, DestroyTarget(WAR_WORLD))
    state = rules.apply(state, DestroyTarget(BLOB_WHEEL))
    assert state.players[1].bases == (THE_HIVE,)
    assert destroys(state) == []


def test_command_ships_ally_ability_destroys_a_base():
    state = rules.new_game(seed=1)
    first = dataclasses.replace(
        state.players[0], hand=(), in_play=(COMMAND_SHIP, CUTTER)
    )
    state = dataclasses.replace(state, players=(first, state.players[1]))
    state = with_opponent_bases(state, THE_HIVE, combat=0)
    state = rules.apply(state, UseAlly(COMMAND_SHIP))
    state = rules.apply(state, DestroyTarget(THE_HIVE))
    assert state.players[1].bases == ()


def test_abilities_left_over_once_the_opponent_has_no_bases_are_dropped():
    state = with_hand(rules.new_game(seed=1), MISSILE_MECH)
    state = with_opponent_bases(state, THE_HIVE, combat=0)
    state = dataclasses.replace(state, may_destroy=1)
    state = rules.apply(state, PlayCard(MISSILE_MECH))
    assert state.may_destroy == 2
    state = rules.apply(state, DestroyTarget(THE_HIVE))
    assert state.may_destroy == 0
