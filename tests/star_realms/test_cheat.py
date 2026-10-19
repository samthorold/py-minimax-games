from collections import Counter
from collections.abc import Hashable, Iterator
from typing import ClassVar, Self

import pytest

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.action import Action, AttackPlayer
from minimax_games.games.star_realms.bench import position, turn_tree
from minimax_games.games.star_realms.bots import GreedyBot
from minimax_games.games.star_realms.catalogue import CATALOGUE, STARTING_DECK
from minimax_games.games.star_realms.cheat import (
    BUDGET,
    CheatBot,
    Decision,
    decide,
    search,
)
from minimax_games.games.star_realms.match import MatchTooLong, play_match
from minimax_games.games.star_realms.node import StarRealmsNode, transposition_key
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


class ExpansionCounter(StarRealmsNode):
    """A node that records how often each position's children are generated."""

    expanded: ClassVar[Counter[Hashable]] = Counter()

    def children(self) -> Iterator[Self]:
        ExpansionCounter.expanded[transposition_key(self)] += 1
        yield from super().children()


def test_the_cheat_searches_each_position_of_its_turn_once():
    """A 1-turn search is all the mover's choices, so every transposition is
    served from the table: each state is expanded once per number of moves
    that reach it (attacking twice reaches the state attacking once does, in
    one more move)."""
    ExpansionCounter.expanded.clear()
    state = position(seed=0, turn=5)
    root = ExpansionCounter(state, state.current)

    search(root, turns=1)

    assert ExpansionCounter.expanded
    assert max(ExpansionCounter.expanded.values()) == 1
    assert sum(ExpansionCounter.expanded.values()) < turn_tree(state).nodes


def unbudgeted_move(state: GameState, turns: int) -> Action:
    return search(StarRealmsNode(state, state.current), turns).moves[0]


@pytest.mark.parametrize(("seed", "turn"), [(0, 3), (1, 3), (1, 11)])
def test_with_a_budget_big_enough_it_plays_the_unbudgeted_move(seed, turn):
    state = position(seed, turn)
    decision = decide(state, turns=2, budget=1_000_000)
    assert decision.depth == 2
    assert decision.move == unbudgeted_move(state, turns=2)


@pytest.mark.parametrize("budget", [1, 2, 50, 3_000])
def test_a_decision_never_passes_its_budget_and_is_deterministic(budget):
    state = position(seed=1, turn=11)
    decision = decide(state, turns=2, budget=budget)
    assert decision.nodes <= budget
    assert decide(state, turns=2, budget=budget) == decision
    assert decision.move in rules.legal_actions(state)


def test_a_finished_decision_counts_the_nodes_it_searched():
    state = position(seed=1, turn=3)
    decision = decide(state, turns=2, budget=1_000_000)
    assert 1 < decision.nodes < 1_000_000
    assert decide(state, turns=2, budget=decision.nodes) == decision
    assert decide(state, turns=2, budget=decision.nodes - 1).depth == 1


def test_when_the_deeper_search_is_cut_off_it_plays_the_deepest_finished_one():
    state = position(seed=1, turn=9)  # 1 turn and 2 turns disagree here
    one_turn, two_turns = unbudgeted_move(state, 1), unbudgeted_move(state, 2)
    assert one_turn != two_turns
    decision = decide(state, turns=2, budget=1_000)
    assert decision.depth == 1
    assert decision.move == one_turn
    assert decide(state, turns=2, budget=10_000).move == two_turns


def test_a_cut_off_first_search_plays_the_best_root_move_it_finished():
    state = mid_turn(combat=5, trade=6, opponent_authority=5)
    actions = rules.legal_actions(state)
    assert actions.index(AttackPlayer()) == len(actions) - 2  # then EndTurn
    # enough to search every move but ending the turn: the attack wins
    decision = decide(state, turns=1, budget=17)
    assert decision.depth == 0
    assert decision.move == AttackPlayer()


def test_with_no_root_move_searched_it_plays_the_first_legal_action():
    state = position(seed=1, turn=11)
    decision = decide(state, turns=2, budget=1)
    assert decision == Decision(rules.legal_actions(state)[0], nodes=1, depth=0)


def test_the_bot_plays_the_budgeted_decision():
    state = position(seed=2, turn=7)
    actions = rules.legal_actions(state)
    bot = CheatBot(turns=2, budget=1_000)
    assert bot.budget == 1_000
    assert bot.choose_from_state(state, actions) == decide(state, 2, 1_000).move
    assert CheatBot().budget == BUDGET


def test_the_budget_is_at_least_one_node():
    with pytest.raises(ValueError, match="at least 1 node"):
        CheatBot(budget=0)


def test_the_search_skips_the_legality_check_for_moves_from_legal_actions(
    monkeypatch: pytest.MonkeyPatch,
):
    expected = [
        child.state for child in StarRealmsNode(mid_turn(5, 5, 20), 0).children()
    ]

    def checked(state: GameState, action: Action) -> GameState:
        raise AssertionError("the search re-checked a move from legal_actions")

    monkeypatch.setattr(rules, "apply", checked)
    children = [
        child.state for child in StarRealmsNode(mid_turn(5, 5, 20), 0).children()
    ]
    assert children == expected
