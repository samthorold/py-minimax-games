"""How it chooses, the first test: does more examination change the plan
where the budget runs out (`examination`), on tiny budgets."""

import pytest

from minimax_games.games.star_realms.bots import GreedyBot
from minimax_games.games.star_realms.evaluate import WIN
from minimax_games.games.star_realms.examination import (
    Probe,
    Record,
    chance,
    measure,
    report,
)
from minimax_games.games.star_realms.match import MatchTooLong, play_match
from minimax_games.games.star_realms.planner import PlannerBot
from minimax_games.games.star_realms.rollout import SPAN


def test_the_probe_replans_every_budget_bound_plan_and_only_those():
    probe = Probe(seed=4, budget=60, factor=4)
    with pytest.raises(MatchTooLong):
        play_match((probe, GreedyBot()), 4, max_turns=12)
    assert len(probe.examined) == probe.plans > probe.turns >= 5
    bound = [found for _, found in probe.examined if not found.base.complete]
    assert bound, "a 60-state budget runs out somewhere"
    assert all(found.more is not None for found in bound)
    assert all(found.more is None for _, found in probe.examined if found.base.complete)
    assert all(f.more and f.more.states > f.base.states for f in bound)


def test_replanning_with_the_same_budget_never_changes_the_line_or_its_value():
    probe = Probe(seed=4, budget=60, factor=1)
    with pytest.raises(MatchTooLong):
        play_match((probe, GreedyBot()), 4, max_turns=12)
    bound = [found for _, found in probe.examined if found.more is not None]
    assert bound
    assert all(found.changed is False and found.rise == 0 for found in bound)
    assert all(
        found.changed is None and found.rise is None
        for _, found in probe.examined
        if found.more is None
    )


def test_more_budget_changes_some_lines_on_tiny_budgets():
    probe = Probe(seed=4, budget=60, factor=8)
    with pytest.raises(MatchTooLong):
        play_match((probe, GreedyBot()), 4, max_turns=16)
    bound = [found for _, found in probe.examined if found.more is not None]
    assert any(found.changed for found in bound)
    assert all(found.rise is not None for found in bound)


def test_measuring_records_every_plan_of_both_players_the_same_on_every_run():
    def run(jobs: int) -> list[Record]:
        return measure(
            2, base_seed=7, budget=60, factor=4, worlds=2, max_turns=10, jobs=jobs
        )

    records = run(1)
    assert {r.seed for r in records} == {7, 8}
    assert {r.turn for r in records} == set(range(1, 11))
    bound = [r for r in records if r.bound]
    assert bound and len(bound) < len(records)
    assert all(r.changed is not None and r.rise is not None for r in bound)
    assert all(r.more_seconds is not None and r.more_seconds > 0 for r in bound)
    assert all(
        r.changed is None and r.more_seconds is None for r in records if not r.bound
    )

    def outcome(records: list[Record]) -> list[tuple]:
        return [
            (r.seed, r.turn, r.bound, r.changed, r.rise, r.more_complete)
            for r in records
        ]

    assert outcome(run(2)) == outcome(records)


def test_the_champions_rises_are_in_judged_chance_on_the_rust_enumeration():
    records = measure(1, base_seed=3, budget=100, worlds=2, model="pf01", max_turns=8)
    assert {r.turn for r in records} == set(range(1, 9))
    rises = [r.rise for r in records if r.bound]
    assert rises
    assert all(rise is not None and -1 <= rise <= 1 for rise in rises)


def test_a_score_reads_as_a_judged_chance_and_a_win_as_certain():
    assert chance(0) == 0.5
    assert chance(SPAN / 2) == 0.75
    assert chance(-SPAN) == 0
    assert chance((WIN + 3 * -SPAN) / 4) == 1


def test_the_report_gives_changed_lines_and_rises_of_budget_bound_plans_by_turn():
    def record(turn: int, changed: bool | None, rise: float | None = None) -> Record:
        bound = changed is not None
        return Record(
            0,
            turn,
            bound,
            changed,
            rise,
            True if bound else None,
            0.010,
            0.040 if bound else None,
        )

    records = [
        record(1, None),
        record(2, None),
        record(3, False, 0.0),
        record(6, True, 0.04),
        record(7, False, 0.01),
        record(8, True, 0.02),
    ]
    lines = report(records, factor=4, band=5)
    text = "\n".join(lines)
    assert "6 plans in 1 game" in text
    assert table_row(lines, "All") == [
        "All", "6", "4", "66.7%", "50.0%", "+1.75", "+3.00", "100.0%"
    ]  # fmt: skip
    assert table_row(lines, "1-5") == [
        "1-5", "3", "1", "33.3%", "0.0%", "+0.00", "-", "100.0%"
    ]  # fmt: skip
    assert table_row(lines, "6-10")[:3] == ["6-10", "3", "3"]
    assert table_row(lines, "ms per plan") == ["ms per plan", "10.0", "10.0", "40.0"]


def table_row(lines: list[str], label: str) -> list[str]:
    """The cells of the report's row starting with `label`."""
    (line,) = [line for line in lines if line.startswith(label + "  ")]
    return [label, *line.removeprefix(label).split()]


def test_the_probe_plays_the_plain_planners_game_from_the_same_seed():
    def transcript(bot: PlannerBot) -> list[str]:
        lines: list[str] = []
        with pytest.raises(MatchTooLong):
            play_match((GreedyBot(), bot), 6, max_turns=12, log=lines.append)
        return lines

    assert transcript(Probe(seed=6, budget=60)) == transcript(PlannerBot(6, budget=60))
