import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from minimax_games._native import star_realms as native
from minimax_games.games.star_realms import regret, selfplay
from minimax_games.games.star_realms.cli import main
from minimax_games.games.star_realms.value import load

# tiny budgets, so a rollout takes milliseconds
SMALL = regret.Settings(worlds=1, budget=200, threads=1)

WIN = (("AttackPlayer",),)
"""Seed 1's first player attacks with its 1 combat: lethal on 1 Authority."""
PASS = (("EndTurn",),)
"""Ending the turn without attacking."""


@pytest.fixture(scope="module")
def pf01() -> Any:
    return load("pf01")


def lethal() -> dict[str, Any]:
    """Seed 1's first decision with both players on 1 Authority: the first
    player can win at once by attacking with its 1 combat, and if it passes,
    the second wins with any Viper in its hand."""
    state = native.Game(1).state()
    assert state["combat"] == 1
    for player in state["players"]:
        player["authority"] = 1
    return state


def test_a_rollout_of_a_winning_line_wins_and_one_that_passes_up_the_win_mostly_loses(
    pf01: Any,
) -> None:
    found = regret.roll(pf01, lethal(), [WIN, PASS], range(8), SMALL)
    assert found.won.shape == (2, 8)
    assert found.won[0].tolist() == [1.0] * 8
    assert found.won[1].mean() < 0.5


def test_the_shortlist_is_the_planner_s_candidates_best_first(pf01: Any) -> None:
    seed = 3
    played = selfplay.play(
        [(seed, selfplay.Planner(pf01, 1, 200), selfplay.Planner(pf01, 1, 200))],
        record=True,
    )
    assert played.actions is not None
    options = regret.shortlist(pf01, native.Game(seed).state(), SMALL, seed)
    # the first is the line the planner plays, with the same worlds
    assert tuple(options[0].line[0]) == tuple(played.actions[0][0])
    judged = [o.judged for o in options]
    assert judged == sorted(judged, reverse=True)
    assert all(0 <= j <= 1 for j in judged)
    assert len({tuple(o.line) for o in options}) == len(options) > 1


def test_a_winning_line_is_judged_certain_and_first(pf01: Any) -> None:
    options = regret.shortlist(pf01, lethal(), SMALL, 0)
    assert options[0].line[-1] == ("AttackPlayer",)
    assert options[0].judged == 1.0


def test_regret_reports_the_loss_of_passing_up_a_win_on_fresh_rollouts(
    pf01: Any,
) -> None:
    calls: list[tuple[list[Any], list[int]]] = []

    def recording(
        model: Any,
        state: dict[str, Any],
        lines: Any,
        seeds: Any,
        settings: regret.Settings,
    ) -> regret.Rollouts:
        calls.append((list(lines), list(seeds)))
        return regret.roll(model, state, lines, seeds, settings)

    # the champion "chose" to pass; the shortlist's other option wins at once
    options = [regret.Option(PASS, 0.6), regret.Option(WIN, 1.0)]
    budget = regret.Budget(per_round=6, fresh=10)
    found = regret.measure(
        pf01, lethal(), options, SMALL, budget, seed=11, roll=recording
    )
    assert found.best == 1
    assert found.regret > 0.5
    assert found.regret == pytest.approx(
        float(np.mean(found.fresh[0] - found.fresh[1]))
    )
    # every call rolls its options out on the same seeds; the screen's seeds
    # and the re-estimate's are disjoint
    *screen, fresh = calls
    assert [len(seeds) for _, seeds in screen] == [6]
    assert fresh[0] == [WIN, PASS]
    assert len(fresh[1]) == 10
    assert not set(fresh[1]) & {s for _, seeds in screen for s in seeds}


def test_no_regret_when_the_chosen_option_comes_out_best(pf01: Any) -> None:
    options = [regret.Option(WIN, 1.0), regret.Option(PASS, 0.6)]
    found = regret.measure(
        pf01, lethal(), options, SMALL, regret.Budget(per_round=4, fresh=4), seed=2
    )
    assert found.best == 0
    assert found.regret == 0.0


def test_sequential_halving_keeps_half_the_options_each_round(pf01: Any) -> None:
    rounds: list[int] = []

    def counting(
        model: Any, state: dict[str, Any], lines: Any, seeds: Any, settings: Any
    ) -> regret.Rollouts:
        rounds.append(len(lines))
        return regret.roll(model, state, lines, seeds, settings)

    state = native.Game(5).state()
    options = regret.shortlist(pf01, state, SMALL, 0)[:4]
    assert len(options) == 4
    budget = regret.Budget(per_round=2, fresh=2)
    regret.measure(pf01, state, options, SMALL, budget, seed=3, roll=counting)
    # 4 options, then 2, then the best and the chosen re-estimated (unless
    # the chosen came out best)
    assert rounds[:2] == [4, 2]
    assert rounds[2:] in ([], [2])


def test_the_kinds_of_sub_choice_two_turns_differ_in() -> None:
    play = ("PlayCard", "Scout")
    buy = ("Acquire", "Cutter")
    end = ("EndTurn",)
    assert regret.differ((play, buy, end), (buy, play, end)) == frozenset()
    assert regret.differ((play, buy, end), (play, end)) == {"buy"}
    assert regret.differ((buy, end), (("AcquireOnTop", "Cutter"), end)) == {"buy"}
    assert regret.differ(
        (("ScrapTradeRowCard", "Ram"), end), (("ScrapTradeRowCard", "Cutter"), end)
    ) == {"Trade Row scrap"}
    assert regret.differ((("AttackPlayer",), end), (("AttackBase", "Ram"), end)) == {
        "attack"
    }
    assert regret.differ(
        (("Choose", "Patrol Mech", 0), buy, end), (("Choose", "Patrol Mech", 1), end)
    ) == {"or choice", "buy"}
    assert regret.differ((("ScrapHandCard", "Scout"), end), (end,)) == {"other"}
    assert regret.group(frozenset({"buy"})) == "buy"
    assert regret.group(frozenset({"or choice", "buy"})) == "mixed"
    assert regret.group(frozenset()) == "order only"


def test_sampled_decisions_are_starts_of_turns_in_the_champion_s_games(
    pf01: Any,
) -> None:
    found, game_turns = regret.decisions(pf01, range(40, 43), SMALL, per_game=3, seed=0)
    assert sorted(game_turns) == [0, 1, 2]
    assert len(found) == 9
    for d in found:
        assert d.seed == 40 + d.game
        assert d.state["turn"] == d.turn
        assert d.state["current"] == d.seat == (d.turn - 1) % 2
        assert 1 <= d.turn <= game_turns[d.game]
    turns = [(d.game, d.turn) for d in found]
    assert len(set(turns)) == len(turns)
    again, _ = regret.decisions(pf01, range(40, 43), SMALL, per_game=3, seed=0)
    assert [(d.game, d.turn) for d in again] == turns


def test_the_shortlist_keeps_the_options_judged_near_the_chosen() -> None:
    options = [
        regret.Option(PASS, 0.60),
        regret.Option(WIN, 0.55),
        regret.Option(PASS, 0.52),
        regret.Option(WIN, 0.40),
        regret.Option(PASS, 0.59),
    ]
    near = regret.sensible(options, candidates=3, margin=0.1)
    assert [o.judged for o in near] == [0.60, 0.55, 0.52]
    assert regret.sensible(options, candidates=4, margin=0.06) == options[:2]
    # a win, or nothing near it: one sensible end
    assert regret.sensible([regret.Option(WIN, 1.0), *options], 4, 0.5) == [
        regret.Option(WIN, 1.0)
    ]


def test_reading_a_run_measures_every_sampled_turn(pf01: Any) -> None:
    found, game_turns = regret.decisions(pf01, range(40, 42), SMALL, per_game=3, seed=0)
    budget = regret.Budget(per_round=2, fresh=2)
    turns = regret.read(pf01, found, SMALL, budget, candidates=3, margin=0.2)
    assert [(t.game, t.turn) for t in turns] == [(d.game, d.turn) for d in found]
    for t in turns:
        assert 1 <= t.options <= 3
        if t.options == 1:
            assert (t.regret, t.group) == (0.0, None)
        assert -1 <= t.regret <= 1
        assert 0 <= t.judged <= 1
    assert any(t.options > 1 for t in turns)
    assert all(np.isfinite(regret.Run(turns, game_turns).loss()))


def test_pairing_compares_shared_and_independent_randomness(pf01: Any) -> None:
    state = native.Game(5).state()
    same = (regret.Option(PASS, 0.5), regret.Option(PASS, 0.5))
    found = regret.pairing(pf01, state, same, SMALL, range(12))
    assert set(found.won) == {"independent", "engine shuffles", "keyed shuffles"}
    # the same option twice: shared randomness makes the two the same game
    assert found.discordant("engine shuffles") == 0.0
    assert found.discordant("keyed shuffles") == 0.0
    assert found.discordant("independent") > 0.0
    assert found.seconds > 0


def _turn(game: int, turn: int, loss: float, group: str | None = None) -> Any:
    return regret.Turn(
        game=game,
        turn=turn,
        seat=turn % 2,
        judged=0.5,
        options=2,
        regret=loss,
        group=group,
    )


def test_loss_per_game_weights_each_sampled_turn_by_the_turns_it_stands_for() -> None:
    run = regret.Run(
        turns=[
            _turn(0, 3, 0.1, "buy"),
            _turn(0, 8, 0.0),
            _turn(1, 5, 0.2, "attack"),
        ],
        game_turns={0: 20, 1: 10},
    )
    # game 0: 20 turns, 2 sampled: 10 x 0.1 lost in all, 0.5 per player;
    # game 1: 10 turns, 1 sampled: 10 x 0.2, 1.0 per player
    mean, se = run.loss()
    assert mean == pytest.approx(0.75)
    assert se == pytest.approx(0.25)
    # a part's loss is its turns' share; the parts add up to the whole
    buys, _ = run.loss(lambda t: t.group == "buy")
    attacks, _ = run.loss(lambda t: t.group == "attack")
    assert buys == pytest.approx(0.25)
    assert buys + attacks == pytest.approx(mean)


def test_the_report_gives_the_loss_by_phase_decidedness_and_kind() -> None:
    run = regret.Run(
        turns=[
            _turn(0, 3, 0.1, "buy"),
            _turn(0, 8, 0.0),
            regret.Turn(0, 9, 0, 1.0, 1, 0.0, None),
            _turn(1, 20, 0.2, "mixed"),
        ],
        game_turns={0: 20, 1: 30},
    )
    text = "\n".join(regret.report(run))
    for heading in ("By game phase", "By judged chance", "By kind of sub-choice"):
        assert heading in text
    assert "turns 1-6" in text
    assert "mixed" in text
    assert "Left out (one sensible end): 1" in text


def test_the_pairing_report_splits_decisions_by_whether_their_options_buy_alike() -> (
    None
):
    won = {
        "independent": np.array([[1, 0, 1, 0], [0, 1, 1, 0]], dtype=float),
        "engine shuffles": np.array([[1, 0, 1, 0], [1, 1, 1, 0]], dtype=float),
        "keyed shuffles": np.array([[1, 0, 1, 0], [1, 0, 1, 0]], dtype=float),
    }
    pairs = [
        (True, regret.Pairing(won, 1.2, 12)),
        (False, regret.Pairing(won, 0.8, 12)),
    ]
    text = "\n".join(regret.pairing_report(pairs))
    assert "Options buy differently: 1" in text
    assert "keyed shuffles" in text
    # discordance 0.5 independent, 0.25 engine, 0 keyed; ms per rollout 83
    assert "0.500" in text
    assert "0.250" in text
    assert "83 ms" in text


TINY = [
    "--games", "2", "--per-game", "2", "--worlds", "1", "--budget", "200",
    "--rollout-worlds", "1", "--rollout-budget", "200", "--base-seed", "40",
]  # fmt: skip


def test_the_command_reads_a_run_and_writes_each_turn(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "turns.csv"
    main(["regret", *TINY, "--per-round", "2", "--fresh", "2", "--out", str(out)])
    printed = capsys.readouterr().out
    assert "Win chance lost per game by one player" in printed
    assert "By kind of sub-choice" in printed
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 4
    measured = [r for r in rows if r["options"] != "1"]
    assert all(int(r["rollouts"]) > 0 for r in measured)
    assert all(r["detail"].startswith("chosen: ") for r in rows if r["group"])
    assert "rollouts" in printed
    assert set(rows[0]) >= {
        "game",
        "turn",
        "seat",
        "judged",
        "options",
        "regret",
        "group",
    }


def test_the_command_measures_pairing_before_a_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["regret", *TINY, "--pair", "3"])
    printed = capsys.readouterr().out
    assert "ms per rollout" in printed
    assert "keyed shuffles" in printed


@pytest.mark.parametrize("keyed", [True, False])
def test_options_rolled_out_on_the_same_seeds_share_their_worlds_and_luck(
    pf01: Any, keyed: bool
) -> None:
    state = native.Game(5).state()
    settings = regret.Settings(worlds=1, budget=200, keyed=keyed, threads=1)
    found = regret.roll(pf01, state, [PASS, PASS], range(10), settings)
    # the same option twice: the same world and randomness, the same games
    assert found.won[0].tolist() == found.won[1].tolist()
    assert found.turns[0].tolist() == found.turns[1].tolist()
    # different seeds are different games
    assert len(set(found.turns[0].tolist())) > 1
    on_two = regret.roll(
        pf01,
        state,
        [PASS, PASS],
        range(10),
        regret.Settings(worlds=1, budget=200, keyed=keyed, threads=2),
    )
    assert on_two.won.tolist() == found.won.tolist()
