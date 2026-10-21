"""Policy iteration (`iterate`, docs/star-realms.md, Learned values: policy
iteration): the opponent mix and seats, the seed ranges, the window, the
gates, the stopping rules, and whole tiny runs that resume where they
stopped."""

import csv
import itertools
import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from minimax_games.games.star_realms import features, iterate, value
from minimax_games.games.star_realms.iterate import Settings
from minimax_games.games.star_realms.sprt import Hypothesis, Sprt

TINY = Settings(
    train_games=12,
    held_out_games=8,
    gate_games=10,
    gate_batch=4,
    ladder_games=4,
    worlds=1,
    budget=60,
    epochs=1,
    hidden=4,
    threads=2,
)
"""Rounds that take a second or two: a handful of games, tiny plans."""


# --- the opponent mix and seats ---------------------------------------------


def test_half_the_games_are_the_best_against_itself_a_quarter_each_the_others():
    seeds = range(100, 200)
    games = iterate.mix("best", ["old1", "old2"], seeds, random.Random(0))
    assert [seed for seed, _, _ in games] == list(seeds)
    kinds = Counter(
        "self"
        if (a, b) == ("best", "best")
        else "heuristic"
        if "heuristic" in (a, b)
        else "pool"
        for _, a, b in games
    )
    assert kinds == {"self": 50, "heuristic": 25, "pool": 25}
    for _, a, b in games:
        assert "best" in (a, b)


def test_the_best_moves_first_in_alternate_games_against_each_opponent():
    games = iterate.mix("best", ["old"], range(40), random.Random(1))
    heuristic = [(a, b) for _, a, b in games if "heuristic" in (a, b)]
    assert heuristic == [("best", "heuristic"), ("heuristic", "best")] * 5
    pool = [(a, b) for _, a, b in games[30:]]
    assert pool == [("best", "old"), ("old", "best")] * 5


def test_earlier_champions_are_chosen_uniformly_from_the_pool():
    games = iterate.mix("c", ["a", "b", "c"], range(12_000), random.Random(2))
    pool = Counter(b if a == "c" else a for _, a, b in games[9000:])
    assert set(pool) == {"a", "b", "c"}
    for count in pool.values():
        assert 850 < count < 1150


def test_the_mix_is_the_same_for_the_same_rng_seed():
    first = iterate.mix("b", ["a", "b"], range(50), random.Random(5))
    assert first == iterate.mix("b", ["a", "b"], range(50), random.Random(5))


# --- seeds -----------------------------------------------------------------


def test_training_held_out_gating_and_ladder_seeds_never_overlap():
    settings = Settings()

    ranges = [("ladder", settings.ladder_seeds)]
    for r in range(1, iterate.MAX_ROUNDS + 1):
        ranges += [
            (f"train {r}", settings.train_seeds(r)),
            (f"held-out {r}", settings.held_out_seeds(r)),
            (f"gate {r}", settings.gate_seeds(r)),
        ]
    assert all(len(seeds) > 0 for _, seeds in ranges)
    # contiguous ranges are disjoint if, in order, each ends before the next
    spans = sorted((s.start, s.stop, name) for name, s in ranges)
    for (_, stop, name), (start, _, other) in itertools.pairwise(spans):
        assert stop <= start, (name, other)


def test_the_ladder_is_seeds_0_to_499_and_rounds_are_5000_games():
    settings = Settings()
    assert settings.ladder_seeds == range(500)
    assert len(settings.train_seeds(1)) + len(settings.held_out_seeds(1)) == 5000


def test_settings_too_big_for_the_seed_ranges_are_refused():
    with pytest.raises(ValueError, match="seeds"):
        Settings(train_games=iterate.STRIDE + 1)
    with pytest.raises(ValueError, match="rounds"):
        Settings(max_rounds=iterate.MAX_ROUNDS + 1)


# --- the window -------------------------------------------------------------


def test_the_window_is_the_last_rounds_up_to_its_size():
    assert iterate.window(1, 4) == [1]
    assert iterate.window(3, 4) == [1, 2, 3]
    assert iterate.window(4, 4) == [1, 2, 3, 4]
    assert iterate.window(7, 4) == [4, 5, 6, 7]


# --- the candidates ---------------------------------------------------------


def _rows(rows: int = 60):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(rows, len(features.NAMES)))
    y = (rng.random(rows) < 0.5).astype(np.float64)
    return x, y


def test_one_member_is_a_single_fit_from_the_fit_seed():
    x, y = _rows()
    settings = replace(TINY, fit_seed=5)
    assert settings.members == 1
    for form, model in iterate.fit_candidates(x, y, settings).items():
        assert len(model.members) == 1
        alone = iterate.fit_candidates(x, y, settings)[form]
        assert np.array_equal(model.predict(x), alone.predict(x))
    mlp = iterate.fit_candidates(x, y, settings)["mlp"]
    single = value.fit(
        value.Mlp(hidden=TINY.hidden), x, y, seed=5, epochs=1, names=features.NAMES
    )
    assert all(np.array_equal(mlp.params[n], single.params[n]) for n in mlp.params)


def test_k_members_are_fits_from_k_consecutive_seeds():
    x, y = _rows()
    settings = replace(TINY, fit_seed=5, members=3)
    candidates = iterate.fit_candidates(x, y, settings)
    for form in ("logistic", "mlp"):
        assert len(candidates[form].members) == 3
    expected = value.fit_ensemble(
        value.LOGISTIC, x, y, seeds=(5, 6, 7), epochs=1, names=features.NAMES
    )
    got = candidates["logistic"]
    assert np.array_equal(got.predict(x), expected.predict(x))


def test_members_must_be_at_least_one():
    with pytest.raises(ValueError, match="members"):
        Settings(members=0)


# --- the gates --------------------------------------------------------------


def test_the_offline_gate_needs_a_strictly_lower_log_loss():
    assert iterate.pregate(0.540, 0.550)
    assert not iterate.pregate(0.550, 0.550)
    assert not iterate.pregate(0.560, 0.550)


def test_the_match_stops_at_the_game_that_decides_it():
    test = Sprt(0.5, 0.55, 0.05, 0.05)
    wins = [1] * 40
    verdict, played = iterate.decide(wins, test, 1000)
    assert verdict is Hypothesis.H1
    assert played == 31  # 31 wins: LLR 2.95 > 2.94
    verdict, played = iterate.decide([0] * 40, test, 1000)
    assert verdict is Hypothesis.H0
    assert played == 28


def test_unfinished_games_count_as_neither_but_toward_the_limit():
    test = Sprt(0.5, 0.55, 0.05, 0.05)
    verdict, played = iterate.decide([None] * 5 + [1] * 40, test, 1000)
    assert (verdict, played) == (Hypothesis.H1, 36)
    verdict, played = iterate.decide([None, 1] * 100, test, 20)
    assert (verdict, played) == (None, 20)


def test_the_match_is_undecided_at_its_limit():
    test = Sprt(0.5, 0.55, 0.05, 0.05)
    level = ([1, 0] * 19 + [1, 1]) * 30  # 52.5%: the ratio barely moves
    verdict, played = iterate.decide(level, test, 1000)
    assert (verdict, played) == (None, 1000)


def test_only_an_accepted_h1_promotes():
    assert iterate.promotes(Hypothesis.H1)
    assert not iterate.promotes(Hypothesis.H0)
    assert not iterate.promotes(None)


# --- stopping ---------------------------------------------------------------


def test_stops_after_patience_rounds_in_a_row_without_a_promotion():
    settings = Settings(max_rounds=10)
    assert iterate.stop_reason([], 0, settings) is None
    assert iterate.stop_reason([False, False], 0, settings) is None
    assert iterate.stop_reason([True, False, False, True, False], 0, settings) is None
    assert "without a promotion" in (
        iterate.stop_reason([True, False, False, False], 0, settings) or ""
    )


def test_stops_after_the_last_round():
    settings = Settings(max_rounds=10)
    rounds = [True, False] * 5
    assert "10 rounds" in (iterate.stop_reason(rounds, 0, settings) or "")
    assert iterate.stop_reason(rounds[:9], 0, settings) is None


def test_stops_once_the_time_cap_has_passed():
    settings = Settings(hours=2.0)
    assert iterate.stop_reason([True], 7199, settings) is None
    assert "2 hours" in (iterate.stop_reason([True], 7200, settings) or "")


def test_a_default_run_is_short():
    settings = Settings()
    assert "3 rounds played" in (
        iterate.stop_reason([True, True, True], 0, settings) or ""
    )
    assert iterate.stop_reason([True], 1799, settings) is None
    assert "0.5 hours" in (iterate.stop_reason([True], 1800, settings) or "")


# --- whole runs --------------------------------------------------------------


def _state(out: Path) -> dict:
    return json.loads((out / "state.json").read_text())


def test_a_tiny_run_records_every_round_and_writes_its_games(tmp_path: Path):
    lines: list[str] = []
    rounds = iterate.run(replace(TINY, max_rounds=2), tmp_path, log=lines.append)
    assert [r.round for r in rounds] == [1, 2]
    state = _state(tmp_path)
    assert [r["round"] for r in state["rounds"]] == [1, 2]
    first = state["rounds"][0]
    assert first["champion"] == "mlp"
    assert first["pool"] == ["mlp"]
    assert first["train"]["games"] + first["train"]["unfinished"] == 12
    assert first["held_out"]["games"] + first["held_out"]["unfinished"] == 8
    assert set(first["losses"]) == {"logistic", "mlp", "champion"}
    assert first["candidate"] in {"logistic", "mlp"}
    assert set(first["seconds"]) >= {"play", "fit", "gate", "total"}
    if first["pregate"]:
        with (tmp_path / "gate-r01.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == first["sprt"]["games"]
        assert [int(row["seed"]) for row in rows] == list(
            TINY.gate_seeds(1)[: len(rows)]
        )
        assert {row["first"] for row in rows} == {"pi01", "mlp"}
    else:
        assert first["sprt"] is None
    assert any("Round 1" in line for line in lines)
    assert "stopped" in state


def test_a_promoted_candidate_becomes_champion_and_joins_the_pool(
    tmp_path: Path, monkeypatch
):
    # the match is rigged to accept H1, so the wiring of a promotion is seen
    monkeypatch.setattr(iterate, "pregate", lambda candidate, best: True)
    monkeypatch.setattr(
        iterate, "decide", lambda outcomes, test, limit: (Hypothesis.H1, 3)
    )
    iterate.run(replace(TINY, max_rounds=2, prefix="t"), tmp_path, log=lambda _: None)
    first, second = _state(tmp_path)["rounds"]
    assert first["promoted"] == "t01"
    assert second["champion"] == "t01"
    assert second["pool"] == ["mlp", "t01"]
    assert (tmp_path / "models" / "t01.npz").is_file()
    with (tmp_path / "ladder-r01.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [int(row["seed"]) for row in rows] == [0, 1, 2, 3]
    assert [row["first"] for row in rows] == ["t01", "heuristic"] * 2
    assert first["ladder"]["games"] == 4
    assert len(list(csv.DictReader((tmp_path / "gate-r01.csv").open()))) == 3


def test_a_failed_candidate_is_dropped(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(iterate, "pregate", lambda candidate, best: True)
    monkeypatch.setattr(
        iterate, "decide", lambda outcomes, test, limit: (Hypothesis.H0, 2)
    )
    iterate.run(replace(TINY, max_rounds=2), tmp_path, log=lambda _: None)
    first, second = _state(tmp_path)["rounds"]
    assert first["promoted"] is None
    assert first["ladder"] is None
    assert second["champion"] == "mlp"
    assert second["pool"] == ["mlp"]
    assert second["window"] == [1, 2]


def test_the_run_stops_after_patience_rounds_without_promotion(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(iterate, "pregate", lambda candidate, best: False)
    iterate.run(replace(TINY, patience=2, max_rounds=5), tmp_path, log=lambda _: None)
    state = _state(tmp_path)
    assert len(state["rounds"]) == 2
    assert "without a promotion" in state["stopped"]


def test_a_run_resumes_after_its_finished_rounds(tmp_path: Path, monkeypatch):
    settings = replace(TINY, max_rounds=1)
    iterate.run(settings, tmp_path, log=lambda _: None)
    before = _state(tmp_path)["rounds"]
    played: list[int] = []
    original = iterate.selfplay.play

    def counting(games, *args, **kwargs):
        played.append(len(games))
        return original(games, *args, **kwargs)

    monkeypatch.setattr(iterate.selfplay, "play", counting)
    iterate.run(replace(settings, max_rounds=2), tmp_path, log=lambda _: None)
    after = _state(tmp_path)["rounds"]
    assert after[0] == before[0]
    assert [r["round"] for r in after] == [1, 2]
    # round 2's games only: the first's were not played again
    assert played[0] == TINY.train_games + TINY.held_out_games


def test_a_round_interrupted_after_its_games_reuses_them(tmp_path: Path, monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(iterate, "fit_candidates", interrupted)
    with pytest.raises(KeyboardInterrupt):
        iterate.run(TINY, tmp_path, log=lambda _: None)
    assert (tmp_path / "data" / "round-01.npz").is_file()
    assert "rounds" not in _state(tmp_path) or not _state(tmp_path)["rounds"]
    monkeypatch.undo()
    played: list[int] = []
    original = iterate.selfplay.play

    def counting(games, *args, **kwargs):
        played.append(len(games))
        return original(games, *args, **kwargs)

    monkeypatch.setattr(iterate.selfplay, "play", counting)
    iterate.run(replace(TINY, max_rounds=1), tmp_path, log=lambda _: None)
    assert TINY.train_games + TINY.held_out_games not in played
    assert len(_state(tmp_path)["rounds"]) == 1


def test_a_run_refuses_to_resume_with_other_settings(tmp_path: Path):
    iterate.run(replace(TINY, max_rounds=1), tmp_path, log=lambda _: None)
    with pytest.raises(ValueError, match="settings"):
        iterate.run(replace(TINY, budget=80, max_rounds=2), tmp_path)
    # the stopping rules and threads may change between invocations
    iterate.run(replace(TINY, max_rounds=1, threads=1, hours=1.0), tmp_path)


def test_a_run_from_before_ensembles_resumes_as_one_member(tmp_path: Path):
    iterate.run(replace(TINY, max_rounds=1), tmp_path, log=lambda _: None)
    path = tmp_path / "state.json"
    state = json.loads(path.read_text())
    del state["settings"]["members"]
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="members"):
        iterate.run(replace(TINY, max_rounds=1, members=2), tmp_path)
    iterate.run(replace(TINY, max_rounds=1), tmp_path, log=lambda _: None)


def test_a_promoted_ensemble_is_saved_whole(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(iterate, "pregate", lambda candidate, best: True)
    monkeypatch.setattr(
        iterate, "decide", lambda outcomes, test, limit: (Hypothesis.H1, 3)
    )
    settings = replace(TINY, max_rounds=2, members=2, prefix="e")
    iterate.run(settings, tmp_path, log=lambda _: None)
    _, second = _state(tmp_path)["rounds"]
    assert second["champion"] == "e01"
    saved = value.Model.load(tmp_path / "models" / "e01.npz")
    assert len(saved.members) == 2
    candidate = tmp_path / "candidates" / "round-01-mlp.npz"
    assert len(value.Model.load(candidate).members) == 2


def test_the_time_cap_counts_the_recorded_rounds(tmp_path: Path):
    ticks = iter(range(0, 10**6, 1000))
    iterate.run(
        replace(TINY, max_rounds=5, hours=1.0),
        tmp_path,
        log=lambda _: None,
        clock=lambda: float(next(ticks)),
    )
    state = _state(tmp_path)
    total = sum(r["seconds"]["total"] for r in state["rounds"])
    assert total >= 3600
    assert total - state["rounds"][-1]["seconds"]["total"] < 3600
    assert "hours" in state["stopped"]
