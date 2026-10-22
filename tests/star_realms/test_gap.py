"""The exploitation gap (`gap`; docs/star-realms.md, What it learns from):
how much a player's judgement of positions exceeds how often the player who
had just ended went on to win from them, on constructed cases with known
answers, and measured on the champion's own games."""

import numpy as np
import pytest

from minimax_games.games.star_realms import gap
from minimax_games.games.star_realms.selfplay import Pairing, Planner, play
from minimax_games.games.star_realms.value import load


def test_the_gap_is_the_mean_judgement_less_the_share_won():
    # judged 0.7 everywhere, won half: a gap of 0.2
    judged = np.full(8, 0.7)
    won = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    game = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    found = gap.gap(judged, won, game)
    assert found.rows == 8
    assert found.games == 4
    assert found.judged == pytest.approx(0.7)
    assert found.won == pytest.approx(0.5)
    assert found.gap == pytest.approx(0.2)


def test_the_error_counts_a_game_s_positions_together():
    # A game's positions share its result, so they aren't independent: the
    # error is the spread of each game's summed misses (cluster-robust).
    judged = np.full(8, 0.7)
    game = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    split = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    # each game's misses cancel out: no spread between games
    assert gap.gap(judged, split, game).error == pytest.approx(0.0)
    # each game's two misses are both -0.5 or both +0.5: sqrt(4 * 1) / 8
    shared = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=float)
    assert gap.gap(judged, shared, game).gap == pytest.approx(0.2)
    assert gap.gap(judged, shared, game).error == pytest.approx(0.25)


def test_a_breakdown_gives_each_bin_s_gap_with_its_counts():
    # judged 0.15 and won 1 of 4; judged 0.85 and won 3 of 4; nothing in
    # the middle bin
    judged = np.array([0.1, 0.2, 0.1, 0.2, 0.8, 0.9, 0.8, 0.9])
    won = np.array([1, 0, 0, 0, 1, 1, 0, 1], dtype=float)
    game = np.arange(8)
    bins = gap.breakdown(judged, won, game, by=judged, edges=(0.0, 0.3, 0.7, 1.0))
    assert [(b.low, b.high) for b in bins] == [(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)]
    low, middle, high = (b.gap for b in bins)
    assert (low.rows, middle.rows, high.rows) == (4, 0, 4)
    assert low.judged == pytest.approx(0.15)
    assert low.won == pytest.approx(0.25)
    assert low.gap == pytest.approx(-0.1)
    assert high.gap == pytest.approx(0.1)


def test_a_breakdown_puts_the_top_edge_in_the_last_bin_and_the_rest_left_closed():
    by = np.array([0.0, 0.5, 1.0, 7.0])
    bins = gap.breakdown(by, np.zeros(4), np.arange(4), by=by, edges=(0.0, 0.5, 1.0))
    assert [b.gap.rows for b in bins] == [1, 2]


def _games() -> tuple[Planner, list[Pairing]]:
    champion = Planner(load("pf01"), worlds=1, budget=60)
    other = Planner(load("mlp"), worlds=1, budget=60)
    games: list[Pairing] = [
        (10, champion, champion),
        (11, champion, "heuristic"),
        (12, "heuristic", champion),
        (13, other, champion),
    ]
    return champion, games


def test_the_champion_s_ends_are_chosen_and_its_opponents_ends_held_out():
    champion, games = _games()
    played = play(games, judge=True)
    assert played.judgement is not None
    found = gap.positions(played, games, champion)
    player = [games[g][1 + s] for g, s in zip(played.game, played.seat, strict=True)]
    mine = np.array([p is champion for p in player])
    assert mine.any()
    assert (~mine).any()
    # chosen: the champion's ends, with the judgement it recorded
    assert len(found.chosen.judged) == mine.sum()
    assert np.array_equal(found.chosen.judged, played.judgement[mine])
    assert np.array_equal(found.chosen.won, played.y[mine])
    assert np.array_equal(found.chosen.game, played.game[mine])
    assert np.array_equal(found.chosen.turn, played.position[mine] + 1)
    # held out: the other players' ends, judged by the champion
    assert np.array_equal(found.held_out.won, played.y[~mine])
    np.testing.assert_allclose(
        found.held_out.judged,
        champion.model.predict(played.x[~mine].astype(np.float64)),
        rtol=0,
        atol=1e-12,
    )
