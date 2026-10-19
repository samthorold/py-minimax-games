import random

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.bench import (
    POSITIONS,
    decision,
    position,
    sampler_decision,
    search,
    turn_tree,
)
from minimax_games.games.star_realms.cheat import CheatBot, decide
from minimax_games.games.star_realms.cli import main
from minimax_games.games.star_realms.sampler import vote
from minimax_games.games.star_realms.view import observe


def test_a_position_is_the_start_of_a_seeded_greedy_game_turn():
    state = position(seed=0, turn=5)
    assert state.turn == 5
    assert rules.winner(state) is None
    assert state == position(seed=0, turn=5)
    assert state != position(seed=1, turn=5)


def test_a_search_counts_its_nodes_and_plays_the_cheats_move():
    state = position(seed=0, turn=1)
    result = search(state, turns=1, max_nodes=100_000)
    assert result.complete
    assert result.nodes > 1
    assert result.seconds > 0
    assert result.move == CheatBot(turns=1).choose_from_state(
        state, rules.legal_actions(state)
    )
    assert search(state, turns=1, max_nodes=100_000).nodes == result.nodes


def test_a_search_past_its_node_budget_stops_there():
    state = position(seed=0, turn=1)
    result = search(state, turns=1, max_nodes=5)
    assert not result.complete
    assert result.nodes == 5
    assert result.move is None
    assert result.nodes_per_second > 0


def test_a_search_can_count_the_distinct_states_it_reaches():
    state = position(seed=0, turn=3)
    plain = search(state, turns=1, max_nodes=100_000)
    assert plain.states is None
    counted = search(state, turns=1, max_nodes=100_000, count_states=True)
    assert counted.nodes == plain.nodes
    assert counted.states is not None
    assert 1 < counted.states < counted.nodes  # some orders transpose


def test_the_bench_command_reports_nodes_rates_and_time_per_decision(capsys):
    argv = ["bench", "--positions", "0:1", "0:3", "--horizons", "1", "2"]
    main([*argv, "--max-nodes", "100", "--states"])
    lines = capsys.readouterr().out.splitlines()
    header = lines[lines.index("") + 1].split()
    assert header[:5] == ["Seed", "Turn", "Horizon", "Nodes", "States"]
    assert "nodes/s" in header
    assert "ms" in header
    rows = [line.split() for line in lines if line[:1].isdigit()]
    assert [row[:3] for row in rows] == [
        ["0", "1", "1"],
        ["0", "1", "2"],
        ["0", "3", "1"],
        ["0", "3", "2"],
    ]
    assert rows[0][3] == "62"
    assert rows[-1][3] == ">100"  # stopped at the budget
    assert any(line.startswith("Horizon 2:") for line in lines)


def test_the_turn_tree_counts_the_one_turn_tree_and_its_distinct_states():
    state = position(seed=0, turn=5)
    tree = turn_tree(state)
    assert tree.states < tree.nodes  # orders of the same plays transpose
    # the cheat's table merges them, so its search visits far fewer nodes
    assert search(state, turns=1, max_nodes=100_000).nodes < tree.nodes


def test_the_bench_command_with_states_sizes_each_whole_first_turn(capsys):
    main(["bench", "--positions", "0:3", "--horizons", "1", "--states"])
    lines = capsys.readouterr().out.splitlines()
    assert (
        "Seed 0 turn 3, one whole turn: 146 nodes, 80 distinct states (1.8x)" in lines
    )


def test_the_default_positions_are_games_still_in_play():
    for seed, turn in POSITIONS:
        state = position(seed, turn)
        assert state.turn == turn
        assert rules.winner(state) is None


def test_a_budgeted_decision_reports_its_depth_nodes_and_time():
    state = position(seed=2, turn=7)
    timed = decision(state, turns=2, budget=1_000)
    assert timed.decision == decide(state, 2, 1_000)
    assert timed.seconds > 0
    assert timed.nodes_per_second > 0


def test_the_bench_command_with_a_budget_times_the_cheats_decisions(capsys):
    argv = ["bench", "--positions", "0:1", "2:7", "--horizons", "2"]
    main([*argv, "--budget", "1000"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == (
        "Cheat decision benchmark: horizon 2 turns, budget 1000 nodes per decision"
    )
    header = lines[lines.index("") + 1].split()
    assert header == ["Seed", "Turn", "Depth", "Nodes", "nodes/s", "ms"]
    start = lines.index("") + 2
    rows = [line.split() for line in lines[start : start + 2]]
    assert [row[:3] for row in rows] == [["0", "1", "2"], ["2", "7", "1"]]
    assert rows[0][3] == "607"
    assert int(rows[1][3]) <= 1000
    assert "Depth finished: 2 turns 1/2, 1 turn 1/2, none 0/2" in lines


def test_a_sampler_decision_votes_from_the_movers_view_seeded_by_the_game():
    state = position(seed=2, turn=7)
    timed = sampler_decision(state, seed=2, worlds=3, turns=2, budget=900)
    view, actions = observe(state, state.current), rules.legal_actions(state)
    assert timed.vote == vote(view, actions, random.Random(2), 3, 2, 900)
    assert timed.seconds > 0


def test_the_bench_command_with_worlds_times_the_samplers_decisions(capsys):
    argv = ["bench", "--positions", "0:1", "2:7", "--horizons", "2"]
    main([*argv, "--budget", "900", "--worlds", "3"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == (
        "Sampler decision benchmark: 3 worlds, horizon 2 turns,"
        " budget 900 nodes per decision (300 per world)"
    )
    header = lines[lines.index("") + 1].split()
    assert header == ["Seed", "Turn", "Depths", "Nodes", "nodes/s", "ms"]
    start = lines.index("") + 2
    rows = [line.split() for line in lines[start : start + 2]]
    expected = [
        sampler_decision(position(seed, turn), seed, 3, 2, 900).vote
        for seed, turn in ((0, 1), (2, 7))
    ]
    assert [row[:2] for row in rows] == [["0", "1"], ["2", "7"]]
    assert [row[2] for row in rows] == [
        ",".join(map(str, found.depths)) for found in expected
    ]
    assert [int(row[3]) for row in rows] == [found.nodes for found in expected]
    depths = [d for found in expected for d in found.depths]
    assert (
        "Worlds' depth finished:"
        f" 2 turns {depths.count(2)}/6, 1 turn {depths.count(1)}/6,"
        f" none {depths.count(0)}/6"
    ) in lines
    assert any(line.endswith("ms per decision") for line in lines)
