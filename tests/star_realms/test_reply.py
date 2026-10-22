"""The paired match of the planner looking through the opponent's reply
against the planner with more worlds (`reply`)."""

import functools

import numpy as np

from minimax_games.games.star_realms import reply
from minimax_games.games.star_realms.selfplay import Reply


@functools.cache
def _small() -> reply.Measurement:
    # tiny budgets, so the match takes about a second
    return reply.measure(
        "mlp",
        range(50, 56),
        worlds=2,
        budget=300,
        reply=Reply(3, 2, 60),
        rival_worlds=4,
        rival_budget=600,
        threads=2,
    )


def test_every_seed_is_played_in_both_seats():
    found = _small()
    assert found.games == 12
    assert found.replier.tolist() == [0, 1] * 6
    both, split, neither = found.pairs
    assert both + split + neither == found.seeds_finished == 6
    assert 2 * both + split == found.wins
    assert found.finished == 12


def test_the_reply_player_s_plans_record_their_replies_and_the_rival_s_none():
    found = _small()
    notes = found.played.replies
    assert notes is not None
    mine = found.mine
    assert mine.sum() > 0
    assert not notes.looked[~mine].any()
    assert notes.looked[mine].any()
    # the shares are of the reply player's plans
    assert found.changed == np.mean(notes.rank[mine] > 1)
    assert 0 < found.states_per_plan(rival=False) <= 300 + 3 * 2 * 60
    assert 0 < found.states_per_plan(rival=True) <= 600
    assert found.ms_per_plan(rival=False) > 0


def test_the_sign_z_counts_seeds_won_both_against_lost_both():
    assert reply.sign_z(30, 17) == (30 - 17) / 47**0.5
    assert reply.sign_z(0, 0) == 0.0


def test_the_reading_follows_the_design_s_rule():
    assert reply.reading(1.9, 0.10).startswith("holds")
    assert reply.reading(2.0, 0.05).startswith("fails")
    assert reply.reading(1.0, 0.20).startswith("neither")


def test_the_report_gives_strength_and_the_changed_choices_by_phase():
    lines = "\n".join(reply.report(_small()))
    assert "sign z" in lines
    assert "ms per plan" in lines
    assert "states per plan" in lines
    assert "turns 1-6" in lines
    assert "Changed" in lines
    assert "rank" in lines
    assert "examined in full" in lines
