"""The diagnostics (`diagnostics`; docs/star-realms.md, How progress is
measured): the judgement's error by how near each player is to a reshuffle
and on positions holding a rare card, and the share of the champion's turns
not examined in full, on constructed cases with known answers."""

import math

import numpy as np
import pytest

from minimax_games.games.star_realms import diagnostics, features, gap
from minimax_games.games.star_realms.selfplay import Examined, Pairing, Planner
from minimax_games.games.star_realms.value import load

_COLUMN = {name: i for i, name in enumerate(features.NAMES)}


def _rows(*places: str) -> np.ndarray:
    """One row per place, each holding one copy of a card there."""
    rows = np.zeros((len(places), len(features.NAMES)))
    for row, place in zip(rows, places, strict=True):
        row[_COLUMN[place]] = 1
    return rows


def test_a_position_holding_a_card_the_trade_deck_has_one_copy_of_is_rare():
    # Battle Blob and Blob World have one copy each, Blob Fighter three; a
    # card in the Trade Row isn't held by either player
    found = gap.rare(
        _rows(
            "ended: pool Battle Blob",
            "next: pool Battle Blob",
            "next: base Blob World",
            "ended: pool Blob Fighter",
            "row Battle Blob",
        )
    )
    assert found.tolist() == [True, True, True, False, False]


def _champion() -> Planner:
    return Planner(load("pf01"), worlds=1, budget=60)


def test_a_champion_s_turn_is_examined_in_full_when_every_plan_in_it_was():
    champion = _champion()
    games: list[Pairing] = [(0, champion, "heuristic"), (1, "heuristic", champion)]
    plans = Examined(
        # game 0: the champion's turn 1 in two plans, one incomplete; its
        # turn 3 in one complete plan. game 1: the champion's turn 2,
        # complete, and turn 4 incomplete; seat 0's plans aren't its
        game=np.array([0, 0, 0, 1, 1, 1]),
        turn=np.array([1, 1, 3, 1, 2, 4]),
        seat=np.array([0, 0, 0, 0, 1, 1]),
        complete=np.array([True, False, True, False, True, False]),
    )
    found = gap.turns(plans, games, champion)
    assert found.game.tolist() == [0, 0, 1, 1]
    assert found.turn.tolist() == [1, 3, 2, 4]
    assert found.full.tolist() == [False, True, True, False]


def test_the_share_not_examined_in_full_is_given_by_turn_with_its_counts():
    found = gap.Turns(
        game=np.zeros(6, dtype=int),
        turn=np.array([1, 3, 5, 7, 9, 30]),
        full=np.array([True, True, False, False, True, False]),
    )
    shares = diagnostics.not_full(found, edges=(1, 5, 10, math.inf))
    assert [(s.low, s.high) for s in shares] == [(1, 5), (5, 10), (10, math.inf)]
    assert [(s.turns, s.not_full) for s in shares] == [(2, 0), (3, 2), (1, 1)]
    assert [s.share for s in shares] == pytest.approx([0.0, 2 / 3, 1.0])
    # an empty bin has no share
    empty = diagnostics.not_full(found, edges=(40, 50))
    assert empty[0].turns == 0
    assert math.isnan(empty[0].share)


def _measured() -> gap.Measurement:
    chosen = gap.Positions(
        judged=np.full(4, 0.7),
        won=np.array([1.0, 1.0, 0.0, 0.0]),
        game=np.array([0, 0, 1, 1]),
        turn=np.array([1, 2, 1, 2]),
        # cards left: the player who just ended, then the next
        left=np.array([[2, 8], [7, 20], [3, 3], [16, 1]]),
        rare=np.array([True, False, False, False]),
    )
    held_out = gap.Positions(
        judged=np.array([0.4, 0.4]),
        won=np.array([0.0, 1.0]),
        game=np.array([2, 2]),
        turn=np.array([3, 4]),
        left=np.array([[0, 0], [9, 9]]),
        rare=np.array([True, True]),
    )
    turns = gap.Turns(
        game=np.array([0, 0, 1]),
        turn=np.array([1, 3, 30]),
        full=np.array([True, False, False]),
    )
    return gap.Measurement(
        "pf01",
        ("mlp",),
        range(100, 106),
        2,
        200,
        gap.Split(chosen, held_out),
        every=2,
        turns=turns,
    )


def _row(text: str, label: str) -> list[str]:
    """The cells of the first line of `text` after `label`."""
    line = next(line for line in text.splitlines() if line.startswith(label))
    return line.removeprefix(label).split()


def test_the_report_gives_each_breakdown_with_its_counts():
    text = diagnostics.report(_measured())
    assert "3 of 6 games" in text
    reshuffle, rare, _, examined = text.split("\n\n(")[1:]
    ended, following = reshuffle.split("By the next player")
    # chosen: near a reshuffle (0-4 left) for the player who just ended,
    # rows 1 and 3: won 1 of 2 judged 0.7, a gap of +0.2 and squared misses
    # of 0.09 and 0.49; held out: the row with none left, judged 0.4, lost
    near = _row(ended, "0-4")
    assert near[:3] == ["2", "2", "+0.200"]
    assert near[5] == "0.290"
    assert near[8:11] == ["1", "1", "+0.400"]
    assert _row(ended, "far 5+")[:2] == ["2", "2"]
    assert _row(following, "0-4")[:2] == ["2", "1"]
    assert _row(following, "15+")[:2] == ["1", "1"]
    # a rare card: one chosen row, won, judged 0.7; both held-out rows
    assert _row(rare, "with")[:3] == ["1", "1", "-0.300"]
    assert _row(rare, "with")[8:10] == ["2", "1"]
    assert _row(rare, "without")[:2] == ["3", "2"]
    assert _row(rare, "without")[8] == "0"
    # turns: 1 of 2 in turns 1-6 not examined in full, 1 of 1 past 24
    assert _row(examined, "turns 1-6") == ["2", "1", "0.500"]
    assert _row(examined, "turns 25+") == ["1", "1", "1.000"]
    assert _row(examined, "turns 7-12") == ["0", "0", "-"]
    assert _row(examined, "all") == ["3", "2", "0.667"]


def test_the_report_compares_near_with_far_and_rare_with_not_within_each_phase():
    # the game's phase confounds both comparisons (decks grow and rare
    # cards are bought as a game goes on, and late positions are easier to
    # call), so each is also given phase by phase, on the chosen positions
    text = diagnostics.report(_measured())
    within = text.split("\n\n(")[3]
    # turns 1-6 hold all four chosen rows: near (rows 1 and 3) and far (2
    # and 4) both miss by 0.09 and 0.49 squared; with a rare card row 1
    # only, without it the other three
    assert _row(within, "turns 1-6") == [
        *("2", "0.290", "2", "0.290"),
        *("1", "0.090", "3", "0.357"),
    ]
    assert _row(within, "turns 7-12") == ["0", "-"] * 4
