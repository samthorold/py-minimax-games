import random

import pytest

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.action import AttackPlayer
from minimax_games.games.star_realms.bench import POSITIONS, position
from minimax_games.games.star_realms.bots import GreedyBot
from minimax_games.games.star_realms.catalogue import CATALOGUE, SCOUT, STARTING_DECK
from minimax_games.games.star_realms.cheat import CheatBot, decide
from minimax_games.games.star_realms.match import MatchTooLong, play_match
from minimax_games.games.star_realms.sampler import SamplerBot, vote
from minimax_games.games.star_realms.state import GameState, PlayerState
from minimax_games.games.star_realms.view import observe, sample_world


def one_world(seed: int = 0) -> GameState:
    """Player 0's turn with trade and combat to spend, in a game whose hidden
    information can only be arranged one way: every deck is all Scouts, the
    opponent's hand too, and the Trade Deck is empty. Player 0's hand is
    empty (played out), so its order, which sampling doesn't keep, can't
    differ either: only the shuffle seed does."""
    me = PlayerState(authority=30, deck=(SCOUT,) * 12)
    them = PlayerState(authority=12, hand=(SCOUT,) * 5, deck=(SCOUT,) * 12)
    row = tuple(CATALOGUE[name] for name in ("Cutter", "Corvette", "Battle Blob"))
    return GameState(
        players=(me, them), rng_seed=seed, trade_row=row, trade=6, combat=3
    )


def test_with_only_one_possible_world_it_plays_what_the_cheat_plays():
    state = one_world()
    actions = rules.legal_actions(state)
    assert len(actions) > 2
    expected = decide(state, 2, 300).move
    assert expected != actions[0]  # not merely the tie-break
    view = observe(state, 0)
    world = sample_world(view, random.Random(0))
    assert world.players == state.players  # the only world: just the seed differs

    bot = SamplerBot(seed=0, worlds=3, turns=2, budget=900)
    assert decide(world, 2, 300).move == expected
    assert bot.choose(view, actions) == expected


def test_it_takes_an_obvious_winning_attack_whatever_the_hidden_cards():
    deck = STARTING_DECK + tuple(CATALOGUE[n] for n in ("Blob Fighter", "Cutter"))
    me = PlayerState(authority=50, deck=deck)
    them = PlayerState(authority=5, hand=deck[:5], deck=deck[5:])
    row = tuple(CATALOGUE[name] for name in ("Cutter", "Corvette", "Battle Blob"))
    trade_deck = tuple(CATALOGUE[n] for n in ("Blob Wheel", "Federation Shuttle"))
    state = GameState(
        players=(me, them),
        rng_seed=0,
        trade_deck=trade_deck,
        trade_row=row,
        trade=6,
        combat=5,
    )
    actions = rules.legal_actions(state)
    assert AttackPlayer() in actions
    assert len(actions) > 2  # buys too: the attack is not the only option

    for seed in range(3):
        bot = SamplerBot(seed, worlds=3, turns=1, budget=300)
        assert bot.choose(observe(state, 0), actions) == AttackPlayer()


def opening(seed: int, turns: int = 4) -> list[str]:
    """The transcript of the first `turns` turns of a seeded sampler-vs-greedy
    game, at a small budget (whole games are too slow for the suite)."""
    lines: list[str] = []
    bot = SamplerBot(seed, worlds=2, turns=1, budget=100)
    with pytest.raises(MatchTooLong):
        play_match((bot, GreedyBot()), seed, max_turns=turns, log=lines.append)
    return lines


def test_a_seeded_sampler_game_replays_identically():
    assert opening(5) == opening(5)
    assert opening(5) != opening(6)


@pytest.mark.parametrize("budget", [3, 50, 400])
def test_a_decision_never_passes_its_budget_and_is_deterministic(budget):
    for seed, turn in POSITIONS:
        state = position(seed, turn)
        view, actions = observe(state, state.current), rules.legal_actions(state)
        votes = [
            vote(view, actions, random.Random(seed), worlds=3, turns=2, budget=budget)
            for _ in range(2)
        ]
        assert votes[0] == votes[1]
        assert votes[0].nodes <= budget
        assert votes[0].move in actions
        assert sum(votes[0].votes.values()) == 3


def test_it_needs_a_world_a_turn_and_a_node_per_world():
    with pytest.raises(ValueError, match="at least 1 world"):
        SamplerBot(0, worlds=0)
    with pytest.raises(ValueError, match="at least 1 turn"):
        SamplerBot(0, turns=0)
    with pytest.raises(ValueError, match="at least 1 node per world"):
        SamplerBot(0, worlds=4, budget=3)


def test_the_defaults_match_the_cheats_horizon_and_budget():
    bot, cheat_bot = SamplerBot(0), CheatBot()
    assert (bot.turns, bot.budget) == (cheat_bot.turns, cheat_bot.budget)
