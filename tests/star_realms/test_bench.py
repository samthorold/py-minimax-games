from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.bench import (
    POSITIONS,
    position,
    search,
    turn_tree,
)
from minimax_games.games.star_realms.cheat import CheatBot
from minimax_games.games.star_realms.cli import main


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
    main([*argv, "--max-nodes", "60", "--states"])
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
    assert rows[0][3] == "12"
    assert rows[-1][3] == ">60"  # stopped at the budget
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
    assert "Seed 0 turn 3, one whole turn: 88 nodes, 44 distinct states (2.0x)" in lines


def test_the_default_positions_are_games_still_in_play():
    for seed, turn in POSITIONS:
        state = position(seed, turn)
        assert state.turn == turn
        assert rules.winner(state) is None
