import dataclasses
import random

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.card import Card, CardType
from minimax_games.games.star_realms.catalogue import SCOUT, STARTING_DECK, VIPER
from minimax_games.games.star_realms.effect import Combat, Trade
from minimax_games.games.star_realms.evaluate import evaluate
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
