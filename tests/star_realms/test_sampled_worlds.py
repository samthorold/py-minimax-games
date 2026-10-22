"""How it chooses, the third test: does the plan change with more sampled
worlds (`sampled_worlds`), on tiny budgets."""

import itertools
import random

import pytest

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.action import Action, EndTurn, PlayCard
from minimax_games.games.star_realms.bots import GreedyBot
from minimax_games.games.star_realms.match import MatchTooLong, play_match
from minimax_games.games.star_realms.planner import reveals
from minimax_games.games.star_realms.sampled_worlds import (
    Probe,
    Record,
    compare,
    line_key,
    measure,
    report,
)
from minimax_games.games.star_realms.state import GameState
from minimax_games.games.star_realms.turn_sizes import turn_starts
from minimax_games.games.star_realms.view import observe, sample_world


def test_when_every_world_is_the_same_more_of_them_never_change_the_line():
    for state in [*turn_starts(0)][2:8]:
        world = sample_world(observe(state, state.current), random.Random(1))
        found = compare([world] * 4, counts=(1, 2, 4), per_world=60)
        assert found.counts == (1, 2, 4)
        assert not any(found.changed(count) for count in (2, 4))
        assert not found.changed_fresh
        assert all(found.seconds[count] > 0 for count in (1, 2, 4))


def two_plays() -> tuple[GameState, Action, Action]:
    """A turn start of a seeded game with two plays from hand that reveal
    nothing and can be made in either order."""
    for state in turn_starts(0):
        plays = [a for a in rules.legal_actions(state) if isinstance(a, PlayCard)]
        for a, b in itertools.permutations(plays, 2):
            after_a = rules.advance_legal(state, a)
            if reveals(state, after_a) or b not in rules.legal_actions(after_a):
                continue
            after_b = rules.advance_legal(state, b)
            if not reveals(state, after_b) and a in rules.legal_actions(after_b):
                return state, a, b
    raise AssertionError("no two plays that commute")


def test_lines_are_the_same_up_to_the_order_of_their_plays():
    state, a, b = two_plays()
    end = EndTurn()
    assert line_key(state, (a, b, end)) == line_key(state, (b, a, end))
    assert line_key(state, (a, end)) != line_key(state, (a, b, end))
    assert line_key(state, (a, b)) != line_key(state, (a, end))


def test_the_probe_compares_counts_at_every_plan_and_plays_the_base_count():
    probe = Probe(seed=4, counts=(1, 2), per_world=150)
    with pytest.raises(MatchTooLong):
        play_match((probe, GreedyBot()), 4, max_turns=10)
    assert probe.worlds == 1
    assert probe.budget == 150
    assert len(probe.compared) == probe.plans > probe.turns >= 4
    turns = [turn for turn, _ in probe.compared]
    assert turns == sorted(turns)
    assert set(turns) == set(range(1, 11, 2))
    assert all(found.counts == (1, 2) for _, found in probe.compared)


def test_measuring_records_every_plan_of_both_players_the_same_on_every_run():
    def run(jobs: int) -> list[Record]:
        return measure(
            2, base_seed=7, counts=(1, 2), per_world=100, max_turns=6, jobs=jobs
        )

    records = run(1)
    assert {r.seed for r in records} == {7, 8}
    assert {r.turn for r in records} == set(range(1, 7))
    assert all(set(r.changed) == {2} and set(r.seconds) == {1, 2} for r in records)

    def outcome(records: list[Record]) -> list[tuple[int, int, dict[int, bool], bool]]:
        return [(r.seed, r.turn, r.changed, r.fresh) for r in records]

    assert outcome(run(2)) == outcome(records)


def test_the_report_gives_the_share_of_changed_lines_by_turn_and_time_per_plan():
    def record(turn: int, at_8: bool, at_16: bool, fresh: bool) -> Record:
        return Record(
            0, turn, {8: at_8, 16: at_16}, fresh, {4: 0.010, 8: 0.020, 16: 0.040}
        )

    records = [
        record(1, False, False, False),
        record(2, True, False, True),
        record(6, True, True, False),
        record(7, False, True, False),
    ]
    lines = report(records, counts=(4, 8, 16), band=5)
    text = "\n".join(lines)
    assert "4 plans in 1 game" in text
    assert table_row(lines, "All") == ["All", "4", "25.0%", "50.0%", "50.0%"]
    assert table_row(lines, "1-5") == ["1-5", "2", "50.0%", "50.0%", "0.0%"]
    assert table_row(lines, "6-10") == ["6-10", "2", "0.0%", "50.0%", "100.0%"]
    assert table_row(lines, "ms per plan") == ["ms per plan", "10.0", "20.0", "40.0"]


def table_row(lines: list[str], label: str) -> list[str]:
    """The cells of the report's row starting with `label`."""
    (line,) = [line for line in lines if line.startswith(label + "  ")]
    return [label, *line.removeprefix(label).split()]


def test_the_champion_is_measured_on_the_rust_enumeration():
    records = measure(
        1, base_seed=3, counts=(2, 4), per_world=50, model="pf01", max_turns=4
    )
    assert {r.turn for r in records} == {1, 2, 3, 4}
    assert all(set(r.changed) == {4} for r in records)
