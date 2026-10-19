import pstats

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.bench import position
from minimax_games.games.star_realms.cli import main
from minimax_games.games.star_realms.profiling import (
    breakdown,
    game_decisions,
    hashing,
    profile,
)


def test_the_profile_command_writes_a_report_and_the_raw_profile(tmp_path, capsys):
    argv = ["profile", "--positions", "0:3", "--budget", "200", "--game-turns", "0"]
    main([*argv, "--out", str(tmp_path)])
    out = capsys.readouterr().out
    report = (tmp_path / "profile.txt").read_text()
    assert str(tmp_path / "profile.txt") in out
    assert "cheat.py" in report  # the decisions' own functions are in it
    stats = pstats.Stats(str(tmp_path / "profile.pstats"))
    assert stats.total_calls > 0  # ty: ignore[unresolved-attribute]


def test_game_decisions_are_the_cheats_decisions_in_a_seeded_game_with_heuristic():
    states = game_decisions(seed=0, turns=3, budget=200)
    assert states  # the cheat, moving first, decides on turns 1 and 3
    assert {state.current for state in states} == {0}
    assert {state.turn for state in states} == {1, 3}
    assert all(rules.legal_actions(state) for state in states)
    assert states == game_decisions(seed=0, turns=3, budget=200)


def test_the_breakdown_splits_the_search_time_into_nested_areas():
    profiled = profile([position(seed=1, turn=11)], turns=2, budget=500)
    areas = breakdown(profiled.stats)
    names = [area.name for area in areas]
    assert names[0] == "search (cheat.decide)"
    for name in (
        "move generation (legal_actions of each node)",
        "rules.advance_legal",
        "automatic-play scan (rules._next_automatic)",
        "applying actions (rules._apply)",
        "building states (GameState/PlayerState.replace)",
        "evaluate",
        "transposition table",
        "search overhead (the rest)",
    ):
        assert name in names
    search = areas[0]
    assert search.seconds > 0
    # every area is within its parent: the nearest shallower area above it
    for i, area in enumerate(areas[1:], start=1):
        parent = next(a for a in reversed(areas[:i]) if a.level < area.level)
        assert area.seconds <= parent.seconds + 1e-9, (area, parent)
    # the top-level areas add up to the search
    top = [area for area in areas if area.level == 1]
    assert abs(sum(area.seconds for area in top) - search.seconds) < 1e-6


def test_hashing_lists_the_self_time_of_every_hash_and_equality_by_class():
    profiled = profile([position(seed=1, turn=11)], turns=2, budget=500)
    rows = {row.name: row for row in hashing(profiled.stats)}
    assert rows["Card.__hash__"].calls > 0  # move generation dedupes cards
    assert rows["GameState.__hash__"].calls > 0  # the transposition table
    total = rows["all hashing and equality"]
    others = [row for name, row in rows.items() if row is not total]
    assert abs(total.seconds - sum(row.seconds for row in others)) < 1e-6


def test_the_report_gives_the_workload_its_speed_and_the_breakdowns(tmp_path):
    argv = ["profile", "--positions", "1:11", "--budget", "300"]
    main([*argv, "--game-seed", "0", "--game-turns", "3", "--out", str(tmp_path)])
    lines = (tmp_path / "profile.txt").read_text().splitlines()
    decisions = len(game_decisions(seed=0, turns=3, budget=300))
    assert lines[0] == (
        f"Cheat decisions profiled: {1 + decisions} (1 benchmark position,"
        f" {decisions} from seed 0's cheat-vs-heuristic game, turns 1-3),"
        " horizon 2 turns, budget 300 nodes"
    )
    assert any(line.startswith("Unprofiled: ") and "nodes/s" in line for line in lines)
    assert any(line.startswith("Profiled: ") and "overhead" in line for line in lines)
    for name in ("search (cheat.decide)", "evaluate", "all hashing and equality"):
        assert any(name in line for line in lines), name
    assert any("ordered by: internal time" in line.lower() for line in lines)
