import pytest

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.action import AttackPlayer
from minimax_games.games.star_realms.bots import GreedyBot
from minimax_games.games.star_realms.catalogue import CATALOGUE, STARTING_DECK
from minimax_games.games.star_realms.cheat import CheatBot
from minimax_games.games.star_realms.match import MatchTooLong, play_match
from minimax_games.games.star_realms.state import GameState, PlayerState


def mid_turn(combat: int, trade: int, opponent_authority: int) -> GameState:
    """Player 0's turn after automatic play, with `combat` and `trade` to spend
    and a Trade Row to buy from."""
    me = PlayerState(authority=50, deck=STARTING_DECK)
    them = PlayerState(authority=opponent_authority, deck=STARTING_DECK)
    row = tuple(CATALOGUE[name] for name in ("Cutter", "Corvette", "Battle Blob"))
    return GameState(
        players=(me, them),
        rng_seed=0,
        trade_row=row,
        trade=trade,
        combat=combat,
    )


def test_the_cheat_takes_an_obvious_winning_attack():
    state = mid_turn(combat=5, trade=6, opponent_authority=5)
    actions = rules.legal_actions(state)
    assert AttackPlayer() in actions
    assert len(actions) > 2  # buys too: the attack is not the only option

    assert CheatBot(turns=1).choose_from_state(state, actions) == AttackPlayer()


def test_the_horizon_defaults_to_two_turns():
    assert CheatBot().turns == 2
    assert CheatBot(turns=1).turns == 1


def test_searching_leaves_the_state_alone():
    state = rules.auto_play(rules.new_game(seed=3))
    snapshot = (hash(state), state.players, state.rng_seed, state.trade_deck)
    CheatBot(turns=1).choose_from_state(state, rules.legal_actions(state))
    assert (hash(state), state.players, state.rng_seed, state.trade_deck) == snapshot
    assert state == rules.auto_play(rules.new_game(seed=3))


def opening(seed: int, turns: int = 6) -> list[str]:
    """The transcript of the first `turns` turns of a seeded cheat-vs-greedy
    game, the cheat searching one turn deep (whole games are too slow for
    the suite)."""
    lines: list[str] = []
    with pytest.raises(MatchTooLong):
        play_match(
            (CheatBot(turns=1), GreedyBot()), seed, max_turns=turns, log=lines.append
        )
    return lines


def test_a_seeded_cheat_game_replays_identically():
    assert opening(5) == opening(5)
    assert opening(5) != opening(6)


def test_the_horizon_is_at_least_one_turn():
    with pytest.raises(ValueError, match="at least 1"):
        CheatBot(turns=0)
