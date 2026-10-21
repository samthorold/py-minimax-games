from dataclasses import replace

import pytest

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.action import EndTurn
from minimax_games.games.star_realms.bench import turn_tree
from minimax_games.games.star_realms.bots import HeuristicBot
from minimax_games.games.star_realms.cli import main
from minimax_games.games.star_realms.state import GameState
from minimax_games.games.star_realms.turn_sizes import (
    TurnEnds,
    game_turn_sizes,
    quantile,
    report,
    turn_ends,
    turn_sizes,
    turn_starts,
    unordered,
)
from minimax_games.games.star_realms.view import observe


def brute_force(state: GameState) -> tuple[set[GameState], set[GameState]]:
    """Every end of the turn from `state` and every state within it, by
    walking every order of every play, with nothing merged."""
    ends: set[GameState] = set()
    within: set[GameState] = set()

    def walk(state: GameState) -> None:
        within.add(state)
        for action in rules.legal_actions(state):
            child = rules.advance_legal(state, action)
            if isinstance(action, EndTurn) or rules.winner(child) is not None:
                ends.add(child)
            else:
                walk(child)

    walk(state)
    return ends, within


def small_turns() -> list[GameState]:
    """Turn starts of seeded heuristic games whose whole tree is small enough
    to walk path by path, with some orders that transpose among them."""
    starts = [start for seed in (0, 1, 2) for start in turn_starts(seed)]
    return [
        start
        for start in starts
        if turn_ends(start, 400).complete and turn_tree(start).nodes <= 3_000
    ]


def test_turn_starts_are_each_turns_first_decision_in_a_heuristic_game():
    starts = list(turn_starts(0))
    assert [start.turn for start in starts] == list(range(1, len(starts) + 1))
    assert all(rules.winner(start) is None for start in starts)
    # replaying the heuristic's moves from the first start reaches the others
    bot, state = HeuristicBot(), starts[0]
    for start in starts[1:4]:
        while state.turn < start.turn:
            actions = rules.legal_actions(state)
            state = rules.advance(
                state, bot.choose(observe(state, state.current), actions)
            )
        assert state == start
    assert starts == list(turn_starts(0))


def test_turn_ends_agree_with_brute_force_enumeration():
    turns = small_turns()
    assert len(turns) >= 20
    assert any(turn_tree(start).nodes > turn_tree(start).states for start in turns)
    for start in turns:
        ends, within = brute_force(start)
        counted = turn_ends(start)
        assert counted.complete
        assert counted.ends == len(ends)
        assert counted.states == len(within)
        assert counted.unordered == len({unordered(end) for end in ends})


def test_turn_ends_visit_the_states_turn_tree_counts():
    start = list(turn_starts(0))[8]  # turn 9
    counted = turn_ends(start)
    assert counted.visited == counted.ends + counted.states
    assert counted.visited == turn_tree(start).states
    assert counted.unordered < counted.ends  # some ends differ only in pile order


def test_turn_ends_stop_past_their_state_cap():
    start = list(turn_starts(0))[8]  # turn 9
    whole = turn_ends(start)
    assert turn_ends(start, max_states=whole.visited).complete
    capped = turn_ends(start, max_states=100)
    assert not capped.complete
    assert capped.visited == 100


def test_unordered_ignores_the_order_of_every_pile_but_the_decks():
    state = list(turn_starts(0))[8]
    first, second = state.players
    assert first.deck != first.deck[::-1]
    piles = replace(first, hand=first.hand[::-1], discard=first.discard[::-1])
    reordered = replace(state, players=(piles, second), trade_row=state.trade_row[::-1])
    assert reordered != state
    assert unordered(reordered) == unordered(state)
    deck = replace(state, players=(replace(first, deck=first.deck[::-1]), second))
    assert unordered(deck) != unordered(state)


def test_game_turn_sizes_measure_each_turn_of_the_game():
    sizes = game_turn_sizes(2, max_states=500)
    assert [size.turn for size in sizes] == [s.turn for s in turn_starts(2)]
    assert all(size.seed == 2 for size in sizes)
    assert sizes[0].ends == turn_ends(next(turn_starts(2)), 500)
    assert any(not size.ends.complete for size in sizes)
    assert all(size.seconds > 0 for size in sizes)


def test_turn_sizes_are_the_same_in_parallel():
    def counts(jobs: int) -> list[tuple[int, int, TurnEnds]]:
        found = turn_sizes(range(3, 6), max_states=300, jobs=jobs)
        return [(size.seed, size.turn, size.ends) for size in found]

    assert counts(2) == counts(1)


def test_a_quantile_is_the_nearest_rank_with_capped_values_above_all():
    assert quantile([5, 1, 3, 2, 4], 0.5) == 3
    assert quantile([5, 1, 3, 2, 4], 0.25) == 2
    assert quantile([5, 1, 3, 2, 4], 1.0) == 5
    assert quantile([5, None, 1], 1.0) is None
    assert quantile([5, None, 1], 0.5) == 5
    with pytest.raises(ValueError, match="no values"):
        quantile([], 0.5)


def test_the_report_gives_the_distribution_budgets_and_turn_bands():
    sizes = turn_sizes(range(2), max_states=2_000)
    lines = report(sizes, max_states=2_000, budgets=(100, 1_000, 5_000), band=10)
    capped = sum(not size.ends.complete for size in sizes)
    assert lines[0] == (
        f"{len(sizes)} turns of 2 heuristic-vs-heuristic games (seeds 0-1),"
        f" each enumerated up to 2000 states; {capped} capped"
    )
    header = next(line for line in lines if line.startswith("Count")).split()
    assert header == ["Count", "q1", "median", "q3", "p90", "p99", "max"]
    visited = next(line for line in lines if line.startswith("States visited"))
    assert visited.split()[-1] == (
        "capped" if capped else str(max(size.ends.visited for size in sizes))
    )
    over = [s.ends.visited > 1_000 or not s.ends.complete for s in sizes]
    share = f"{100 * sum(over) / len(sizes):.0f}%"
    over_line = next(line for line in lines if line.startswith("Over budget"))
    assert f"1000 {share}" in over_line
    assert "5000" not in over_line  # past the cap: unknown
    start = lines.index("By turn number (states visited):") + 1
    assert lines[start].split() == ["Turns", "Count", "median", "p90", ">100", ">1000"]
    assert [line.split()[0] for line in lines[start + 1 : start + 3]] == [
        "1-10",
        "11-20",
    ]


def test_the_turn_sizes_command_prints_the_distribution(capsys):
    main(["turn-sizes", "--games", "1", "--base-seed", "2", "--max-states", "500"])
    lines = capsys.readouterr().out.splitlines()
    sizes = game_turn_sizes(2, max_states=500)
    assert lines[0].startswith(f"{len(sizes)} turns of 1 heuristic-vs-heuristic game")
    assert any(line.startswith("End states") for line in lines)
