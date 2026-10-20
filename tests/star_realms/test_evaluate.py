import dataclasses
import random
import typing

import pytest

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.card import Card, CardType, Faction
from minimax_games.games.star_realms.catalogue import (
    CATALOGUE,
    SCOUT,
    STARTING_DECK,
    VIPER,
)
from minimax_games.games.star_realms.effect import (
    AcquireShipFreeOnTop,
    AllyForAllFactions,
    Authority,
    Choice,
    Combat,
    CopyShip,
    DestroyBase,
    DiscardThenDraw,
    Draw,
    DrawIfBases,
    DrawPerFactionCard,
    Effect,
    NextShipOnTop,
    OpponentDiscards,
    ScrapFromHand,
    ScrapHandOrDiscard,
    ScrapThenDraw,
    ScrapTradeRow,
    ShipsGainCombat,
    Trade,
)
from minimax_games.games.star_realms.evaluate import effect_value, evaluate
from minimax_games.games.star_realms.state import GameState, PlayerState


def seeded_states(seed: int, moves: int = 60) -> list[GameState]:
    """The states of a random game from `seed`, up to `moves` decisions."""
    rng = random.Random(seed)
    state = rules.auto_play(rules.new_game(seed))
    states = [state]
    for _ in range(moves):
        actions = rules.legal_actions(state)
        if not actions:
            break
        state = rules.advance(state, rng.choice(actions))
        states.append(state)
    return states


def test_the_score_is_antisymmetric_on_seeded_states():
    for seed in range(20):
        for state in seeded_states(seed):
            assert evaluate(state, 0) == -evaluate(state, 1)


# Effect-free bases, so only their defense and type differ.
OUTPOST = Card("Wall", type=CardType.OUTPOST, defense=5)
BASE = Card("Keep", type=CardType.BASE, defense=5)
# Ships that differ only in their trade or combat.
FREIGHTER = Card("Freighter", primary=(Trade(3),))
BARGE = Card("Barge", primary=(Trade(1),))
GUNSHIP = Card("Gunship", primary=(Combat(3),))
SKIFF = Card("Skiff", primary=(Combat(1),))


def even_state(**player0: object) -> GameState:
    """Two identical players with starting decks, except `player0`'s changes."""
    player = PlayerState(authority=50, deck=STARTING_DECK)
    return GameState(
        players=(dataclasses.replace(player, **player0), player), rng_seed=0
    )


def assert_better_for_player_0(better: GameState, worse: GameState) -> None:
    assert evaluate(better, 0) > evaluate(worse, 0)
    assert evaluate(better, 1) < evaluate(worse, 1)
    for state in (better, worse):
        assert evaluate(state, 0) == -evaluate(state, 1)


def test_an_even_state_scores_zero():
    assert evaluate(even_state(), 0) == 0


def test_more_authority_scores_higher():
    assert_better_for_player_0(even_state(authority=51), even_state())


def test_a_base_in_play_scores_higher_than_none():
    assert_better_for_player_0(even_state(bases=(BASE,)), even_state())


def test_an_outpost_scores_higher_than_a_base_of_equal_defense():
    assert_better_for_player_0(even_state(bases=(OUTPOST,)), even_state(bases=(BASE,)))


def test_a_deck_with_more_trade_per_card_scores_higher():
    assert_better_for_player_0(
        even_state(deck=(*STARTING_DECK, FREIGHTER)),
        even_state(deck=(*STARTING_DECK, BARGE)),
    )


def test_a_deck_with_more_combat_per_card_scores_higher():
    assert_better_for_player_0(
        even_state(deck=(*STARTING_DECK, GUNSHIP)),
        even_state(deck=(*STARTING_DECK, SKIFF)),
    )


def test_deck_economy_counts_every_pile_the_player_owns():
    in_deck = evaluate(even_state(deck=(*STARTING_DECK, FREIGHTER)), 0)
    elsewhere = [
        even_state(**{pile: (FREIGHTER,)}) for pile in ("hand", "in_play", "discard")
    ]
    assert [evaluate(state, 0) for state in elsewhere] == [in_deck] * 3


def test_owning_fewer_scouts_or_vipers_scores_higher():
    for starter in (SCOUT, VIPER):
        fewer = list(STARTING_DECK)
        fewer.remove(starter)
        assert_better_for_player_0(even_state(deck=tuple(fewer)), even_state())


def test_a_won_state_scores_beyond_any_non_terminal_state_and_a_lost_one_below():
    # Player 0 has won, though on every other term player 1 is far ahead.
    won = GameState(
        players=(
            PlayerState(authority=1, deck=STARTING_DECK),
            PlayerState(authority=0, deck=(GUNSHIP,) * 10, bases=(OUTPOST,) * 20),
        ),
        rng_seed=0,
    )
    # Not over, though player 0 is as far ahead as it can be made.
    lopsided = GameState(
        players=(
            PlayerState(authority=10**12, deck=(GUNSHIP,) * 10, bases=(OUTPOST,) * 20),
            PlayerState(authority=1, deck=STARTING_DECK),
        ),
        rng_seed=0,
    )
    assert evaluate(won, 0) > evaluate(lopsided, 0)
    assert evaluate(won, 1) < evaluate(lopsided, 1)
    assert evaluate(won, 0) == -evaluate(won, 1)


# A ship that does nothing, to compare cards with abilities against.
HULK = Card("Hulk")


def test_a_card_whose_only_ability_is_authority_scores_higher_than_a_blank():
    medic = Card("Medic", primary=(Authority(4),))
    assert_better_for_player_0(
        even_state(deck=(*STARTING_DECK, medic)),
        even_state(deck=(*STARTING_DECK, HULK)),
    )


# Blob ships alike except for an ally ability, and a Blob partner for them.
BLOB_PLAIN = Card("Blob Plain", faction=Faction.BLOB, primary=(Combat(3),))
BLOB_ALLY = dataclasses.replace(BLOB_PLAIN, name="Blob Ally", ally=(Combat(4),))
BLOB_PARTNER = Card("Blob Partner", faction=Faction.BLOB)


def test_an_ally_ability_scores_higher_than_none():
    assert_better_for_player_0(
        even_state(deck=(*STARTING_DECK, BLOB_ALLY, BLOB_PARTNER)),
        even_state(deck=(*STARTING_DECK, BLOB_PLAIN, BLOB_PARTNER)),
    )


def test_an_ally_ability_is_worth_more_with_another_card_of_its_faction():
    federation_partner = Card("Federation Partner", faction=Faction.TRADE_FEDERATION)
    assert_better_for_player_0(
        even_state(deck=(*STARTING_DECK, BLOB_ALLY, BLOB_PARTNER)),
        even_state(deck=(*STARTING_DECK, BLOB_ALLY, federation_partner)),
    )


def test_a_scrap_ability_scores_higher_than_none():
    salvage = Card("Salvage", scrap=(Combat(5),))
    assert_better_for_player_0(
        even_state(deck=(*STARTING_DECK, salvage)),
        even_state(deck=(*STARTING_DECK, HULK)),
    )


def test_a_scrap_ability_is_worth_less_than_the_same_primary_ability():
    salvage = Card("Salvage", scrap=(Combat(5),))
    gun = Card("Gun", primary=(Combat(5),))
    assert_better_for_player_0(
        even_state(deck=(*STARTING_DECK, gun)),
        even_state(deck=(*STARTING_DECK, salvage)),
    )


# One of each kind of effect.
EVERY_EFFECT: tuple[Effect, ...] = (
    Trade(2),
    Combat(2),
    Authority(2),
    Draw(1),
    OpponentDiscards(1),
    DestroyBase(),
    ScrapHandOrDiscard(1),
    ScrapTradeRow(1),
    NextShipOnTop(),
    Choice(((Trade(1),), (Combat(1),))),
    AcquireShipFreeOnTop(),
    DrawIfBases(2, 2),
    DrawPerFactionCard(),
    DiscardThenDraw(2),
    ScrapThenDraw(2),
    ScrapFromHand(),
    CopyShip(),
    ShipsGainCombat(1),
    AllyForAllFactions(),
)


def test_every_effect_kind_is_covered():
    kinds = typing.get_args(Effect.__value__)
    assert {type(e) for e in EVERY_EFFECT} == set(kinds)


@pytest.mark.parametrize("effect", EVERY_EFFECT, ids=lambda e: type(e).__name__)
def test_a_card_with_any_effect_scores_higher_than_a_blank(effect: Effect):
    card = Card("Able", primary=(effect,))
    assert_better_for_player_0(
        even_state(deck=(*STARTING_DECK, card)),
        even_state(deck=(*STARTING_DECK, HULK)),
    )


def test_an_effect_of_an_unknown_kind_has_no_value():
    with pytest.raises(AssertionError):
        effect_value(typing.cast(Effect, object()))


def test_a_choice_is_worth_its_best_option():
    choice = Choice(((Trade(1),), (Combat(3), Authority(1)), (Draw(1),)))
    best = max(sum(effect_value(e) for e in option) for option in choice.options)
    assert effect_value(choice) == best > 0


def test_a_card_is_valued_by_its_abilities_not_its_name():
    real = CATALOGUE["Battle Blob"]
    impostor = Card("Battle Blob")
    with_real = even_state(deck=(*STARTING_DECK, real))
    with_impostor = even_state(deck=(*STARTING_DECK, impostor))
    for _ in range(2):  # Either way round, after either is cached.
        assert_better_for_player_0(with_real, with_impostor)
        assert evaluate(with_impostor, 0) == evaluate(
            even_state(deck=(*STARTING_DECK, HULK)), 0
        )
