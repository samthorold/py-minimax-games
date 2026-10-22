import json

import pytest

from minimax_games.games.star_realms import features, planner
from minimax_games.games.star_realms.action import Action
from minimax_games.games.star_realms.bots import GreedyBot, HeuristicBot
from minimax_games.games.star_realms.cheat import BUDGET, CheatBot
from minimax_games.games.star_realms.cli import BOTS, configured, main
from minimax_games.games.star_realms.evaluate import LINEAR
from minimax_games.games.star_realms.ismcts import ITERATIONS, IsmctsBot
from minimax_games.games.star_realms.planner import PlannerBot
from minimax_games.games.star_realms.rollout import ROLLOUT_BUDGET, Rollout
from minimax_games.games.star_realms.sampler import WORLDS, SamplerBot
from minimax_games.games.star_realms.state import GameState
from minimax_games.games.star_realms.value import Model
from minimax_games.games.star_realms.view import observe


def test_prints_the_result_of_a_seeded_game(capsys):
    main(["--seed", "7"])
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1
    assert "wins on turn" in out[0]


def test_with_log_prints_the_transcript(capsys):
    main(["--seed", "7", "--log"])
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("Turn 1: Player 1 to play.")
    assert "wins on turn" in out[-1]


def test_bots_can_be_picked_by_name(capsys):
    main(["--seed", "7", "--bots", "greedy", "random", "--log"])
    out = capsys.readouterr().out.splitlines()
    assert out[-1].startswith("Player 1 wins on turn")
    assert any(line.startswith("Player 1 acquires ") for line in out)


def test_the_heuristic_bot_is_registered_as_heuristic(capsys):
    assert isinstance(BOTS["heuristic"](0), HeuristicBot)
    main(["--seed", "7", "--bots", "heuristic", "greedy"])
    assert "wins on turn" in capsys.readouterr().out


def test_tournament_reports_win_rates_by_seat_and_time_per_move(capsys):
    main(["tournament", "--bots", "greedy", "random", "--games", "4"])
    out = capsys.readouterr().out
    assert "greedy vs random: 4 games, seeds 0-3, max 300 turns" in out
    assert "Unfinished: 0" in out
    lines = out.splitlines()
    table = lines[lines.index("") + 1 :][:3]
    header, greedy, random_ = (line.split() for line in table)
    assert header == ["Bot", "Wins", "Win%", "As", "1st", "As", "2nd", "ms/move"]
    assert greedy[1] == "4/4"
    assert greedy[2] == "100.0%"
    assert random_[1] == "0/4"


def test_tournament_jobs_split_the_games_without_changing_the_results(capsys):
    def results(*jobs: str) -> list[list[str]]:
        main(["tournament", "--bots", "greedy", "heuristic", "--games", "6", *jobs])
        out = capsys.readouterr().out.splitlines()
        assert "Unfinished: 0" in out
        bots = [line.split() for line in out if line.startswith(("greedy", "heur"))]
        return [row[:-1] for row in bots]  # all but ms/move, which varies

    assert results("--jobs", "3") == results()


def test_tournament_with_sprt_stops_once_decided_and_says_why(capsys):
    argv = ["tournament", "--bots", "heuristic", "random", "--games", "100"]
    main([*argv, "--sprt", "--h0", "0.5", "--h1", "0.7"])
    out = capsys.readouterr().out
    assert "heuristic vs random: SPRT, at most 100 games, seeds 0-99" in out
    assert "SPRT: H0 heuristic wins 50.0%, H1 70.0%; alpha 0.05, beta 0.05" in out
    decision = next(line for line in out.splitlines() if line.startswith("Decision"))
    assert decision.startswith("Decision: H1 accepted after ")
    assert "LLR " in out
    assert "bounds [-2.94, 2.94]" in out
    assert "95% interval" in out


def test_tournament_with_sprt_reports_an_undecided_limit(capsys):
    argv = ["tournament", "--bots", "random", "random", "--games", "2", "--sprt"]
    main(argv)
    out = capsys.readouterr().out
    assert "Decision: undecided after 2 games (the limit)" in out


def test_tournament_lists_unfinished_games_and_tells_same_bots_apart(capsys):
    argv = ["tournament", "--bots", "random", "random", "--games", "2"]
    main([*argv, "--max-turns", "5"])
    out = capsys.readouterr().out.splitlines()
    assert not any("budget" in line for line in out)  # no search, no budget
    assert "Unfinished: 2 (seeds 0, 1)" in out
    assert "Win rates are of the 0 finished games." in out
    assert any(line.startswith("random-A  0/0") for line in out)
    assert any(line.startswith("random-B  0/0") for line in out)


def test_tournament_plays_the_cheat_and_states_its_budget(capsys):
    # Whole games at the default budget take seconds each: stop early.
    argv = ["tournament", "--bots", "cheat", "greedy", "--games", "2"]
    main([*argv, "--max-turns", "4"])
    out = capsys.readouterr().out.splitlines()
    assert (
        f"cheat: 2-turn horizon, budget {BUDGET} nodes per decision, linear evaluation"
    ) in out
    assert "Unfinished: 2 (seeds 0, 1)" in out
    cheat = next(line for line in out if line.split()[:2] == ["cheat", "0/0"])
    assert float(cheat.split()[-1]) > 0  # it made timed decisions


def test_the_cheat_is_marked_benchmark_only_in_the_help(capsys):
    for argv in (["--help"], ["tournament", "--help"]):
        with pytest.raises(SystemExit):
            main(argv)
        help_text = " ".join(capsys.readouterr().out.split())
        assert "cheat (benchmark only: sees the whole state)" in help_text


class Peeker:
    """A fast stand-in for the cheat: benchmark-only, plays greedily."""

    def __init__(self) -> None:
        self.calls = 0

    def choose_from_state(
        self, state: GameState, actions: tuple[Action, ...]
    ) -> Action:
        self.calls += 1
        return GreedyBot().choose(observe(state, state.current), actions)


def test_a_single_game_plays_the_cheat(capsys, monkeypatch):
    # The real cheat is too slow for a whole game: check it is the one
    # registered, then play a fast benchmark-only stand-in under its name.
    assert isinstance(BOTS["cheat"](0), CheatBot)
    peeker = Peeker()
    monkeypatch.setitem(BOTS, "cheat", lambda seed: peeker)
    main(["--seed", "7", "--bots", "cheat", "greedy"])
    assert "wins on turn" in capsys.readouterr().out
    assert peeker.calls > 0


def test_a_single_game_plays_the_sampler(capsys, monkeypatch):
    # Whole games at the default budget are too slow: check the registered
    # sampler is seeded from the game, then play a small-budget one.
    sampler = BOTS["sampler"](7)
    assert isinstance(sampler, SamplerBot)
    assert (sampler.worlds, sampler.turns, sampler.budget) == (WORLDS, 2, BUDGET)
    monkeypatch.setitem(
        BOTS, "sampler", lambda seed: SamplerBot(seed, worlds=2, turns=1, budget=40)
    )
    main(["--seed", "7", "--bots", "sampler", "greedy"])
    assert "wins on turn" in capsys.readouterr().out


def test_tournament_plays_the_sampler_and_states_its_worlds_and_budget(capsys):
    argv = ["tournament", "--bots", "sampler", "greedy", "--games", "2"]
    main([*argv, "--max-turns", "2"])
    out = capsys.readouterr().out.splitlines()
    assert (
        f"sampler: {WORLDS} worlds, 2-turn horizon,"
        f" budget {BUDGET} nodes per decision ({BUDGET // WORLDS} per world),"
        " linear evaluation"
    ) in out
    sampler = next(line for line in out if line.split()[:2] == ["sampler", "0/0"])
    assert float(sampler.split()[-1]) > 0  # it made timed decisions


def test_tournament_sets_the_search_bots_budget_and_the_samplers_worlds(capsys):
    argv = ["tournament", "--bots", "cheat", "sampler", "--games", "2"]
    tail = ["--max-turns", "2", "--budget", "60", "--worlds", "3"]
    main([*argv, *tail])
    out = capsys.readouterr().out.splitlines()
    assert "cheat: 2-turn horizon, budget 60 nodes per decision, linear evaluation" in (
        out
    )
    assert (
        "sampler: 3 worlds, 2-turn horizon, budget 60 nodes per decision"
        " (20 per world), linear evaluation"
    ) in out
    main([*argv, *tail, "--jobs", "2"])  # the configured bots reach the workers
    parallel = capsys.readouterr().out.splitlines()
    assert parallel[:4] == out[:4]  # the same settings and unfinished games


def test_tournament_can_give_the_search_bots_the_rollout_evaluation(capsys):
    argv = ["tournament", "--bots", "cheat", "sampler", "--games", "2"]
    tail = ["--max-turns", "2", "--budget", "60", "--eval", "rollout"]
    rollout = ["--rollouts", "3", "--rollout-turns", "1", "--epsilon", "0.2"]
    main([*argv, *tail, *rollout])
    out = capsys.readouterr().out.splitlines()
    evaluator = "rollout evaluation (3 x 1 turns, epsilon 0.2, scale 16)"
    assert f"cheat: 2-turn horizon, budget 60 nodes per decision, {evaluator}" in out
    assert (
        "sampler: 4 worlds, 2-turn horizon, budget 60 nodes per decision"
        f" (15 per world), {evaluator}"
    ) in out
    main([*argv, *tail, *rollout, "--jobs", "2"])  # it reaches the workers
    assert capsys.readouterr().out.splitlines()[:4] == out[:4]


def test_with_rollouts_the_search_bots_budget_defaults_to_equal_compute(capsys):
    argv = ["tournament", "--bots", "cheat", "greedy", "--games", "1"]
    main([*argv, "--max-turns", "1", "--eval", "rollout"])
    out = capsys.readouterr().out.splitlines()
    assert out[1].startswith(f"cheat: 2-turn horizon, budget {ROLLOUT_BUDGET} nodes")


def test_the_rollout_evaluation_is_seeded_per_game():
    factory = configured("cheat", evaluator=Rollout(rollouts=2))
    first, second = factory(7), factory(8)
    assert isinstance(first, CheatBot) and isinstance(second, CheatBot)
    assert first.evaluator == Rollout(rollouts=2, seed=7)
    assert second.evaluator == Rollout(rollouts=2, seed=8)
    default = BOTS["cheat"](7)
    assert isinstance(default, CheatBot)
    assert default.evaluator == LINEAR


def test_tune_prints_each_step_and_resumes_from_its_cache(capsys, tmp_path):
    cache = tmp_path / "trials.jsonl"
    argv = ["tune", "--fields", "trade", "--games", "2", "--budget", "20"]
    main([*argv, "--cache", str(cache)])
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("Tuning trade: cheat vs heuristic, 2 games per trial,")
    assert out[1].startswith("start: ")
    assert out[3].startswith("  trade 20 -> 40: ")
    assert out[-1].startswith("Weights: Weights(")
    trials = cache.read_text()
    assert len(trials.splitlines()) == 3

    main([*argv, "--cache", str(cache)])
    assert capsys.readouterr().out.splitlines() == out
    assert cache.read_text() == trials


def test_tune_screen_ranks_the_weights_by_decisions_changed(capsys):
    fields = ["--fields", "trade", "draw"]
    main(
        ["tune", "--screen", *fields, "--games", "1", "--budget", "20", "--every", "20"]
    )
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("Screen: ")
    assert len(out) == 3
    assert all("% of decisions changed" in line for line in out[1:])


def test_a_single_game_plays_the_ismcts_bot(capsys, monkeypatch):
    # Whole games at the default budget are too slow: check the registered
    # bot is the default one, then play a small-budget one.
    bot = BOTS["ismcts"](7)
    assert isinstance(bot, IsmctsBot)
    assert bot.iterations == ITERATIONS
    monkeypatch.setitem(BOTS, "ismcts", lambda seed: IsmctsBot(seed, iterations=5))
    main(["--seed", "7", "--bots", "ismcts", "greedy"])
    assert "wins on turn" in capsys.readouterr().out


def test_tournament_plays_the_ismcts_bot_and_states_its_iterations(capsys):
    argv = ["tournament", "--bots", "ismcts", "greedy", "--games", "2"]
    main([*argv, "--max-turns", "2", "--iterations", "7"])
    out = capsys.readouterr().out.splitlines()
    assert f"ismcts: {IsmctsBot(0, iterations=7).settings()}" in out
    assert out[1].startswith("ismcts: 7 iterations per decision")
    ismcts = next(line for line in out if line.split()[:2] == ["ismcts", "0/0"])
    assert float(ismcts.split()[-1]) > 0  # it made timed decisions
    main([*argv, "--max-turns", "2", "--iterations", "7", "--jobs", "2"])
    assert capsys.readouterr().out.splitlines()[:3] == out[:3]


def test_fit_prints_each_held_out_log_loss_and_saves_the_models(capsys, tmp_path):
    main(
        [
            *("fit", "--games", "6", "--test-games", "3", "--epochs", "2"),
            *("--hidden", "3", "--save", str(tmp_path)),
        ]
    )
    out = capsys.readouterr().out
    assert "Train: 6 games (seeds 20000-20005)" in out
    assert "sigmoid(evaluate / 16):" in out
    assert "logistic:" in out
    assert "mlp:" in out
    for name in ("logistic", "mlp"):
        assert Model.load(tmp_path / f"{name}.npz").form == name


def test_native_bench_times_the_rust_engine(capsys):
    main(["native-bench", "--games", "3"])
    out = capsys.readouterr().out
    assert "greedy games: 3 in" in out
    assert "games/s" in out
    assert "calls/s" in out
    assert "heuristic games: 3 in" in out


def test_a_single_game_plays_the_planner(capsys, monkeypatch):
    bot = BOTS["planner"](7)
    assert isinstance(bot, PlannerBot)
    assert (bot.budget, bot.worlds) == (planner.BUDGET, planner.WORLDS)
    monkeypatch.setitem(BOTS, "planner", lambda seed: PlannerBot(seed, budget=40))
    main(["--seed", "7", "--bots", "planner", "greedy"])
    assert "wins on turn" in capsys.readouterr().out


def test_tournament_plays_the_planner_and_reports_its_plans(capsys):
    argv = ["tournament", "--bots", "planner", "greedy", "--games", "2"]
    main([*argv, "--max-turns", "3", "--budget", "60", "--worlds", "2"])
    out = capsys.readouterr().out.splitlines()
    assert f"planner: {PlannerBot(0, worlds=2, budget=60).settings()}" in out
    plans = next(line for line in out if "plans per turn" in line)
    assert plans.startswith("planner: ")
    assert plans.endswith(" ms per plan")
    main([*argv, "--max-turns", "3", "--budget", "60", "--worlds", "2", "--jobs", "2"])
    assert capsys.readouterr().out.splitlines()[:3] == out[:3]


def test_tournament_with_sprt_plays_the_planner_and_reports_its_plans(capsys):
    argv = ["tournament", "--bots", "planner", "greedy", "--games", "2", "--sprt"]
    main([*argv, "--max-turns", "3", "--budget", "60", "--worlds", "2"])
    out = capsys.readouterr().out.splitlines()
    assert "Decision: undecided after 2 games (the limit)" in out
    plans = next(line for line in out if "plans per turn" in line)
    assert plans.startswith("planner: ")
    assert plans.endswith(" ms per plan")


def test_the_planner_keeps_its_own_budget_with_rollouts():
    bot = configured("planner", None, None, Rollout())(3)
    assert isinstance(bot, PlannerBot)
    assert bot.budget == planner.BUDGET
    assert bot.evaluator == Rollout(seed=3)


def test_planner_with_a_learned_value_uses_the_model_and_the_rust_enumeration():
    bot = configured("planner:mlp", 60, 2)(3)
    assert isinstance(bot, PlannerBot)
    assert (bot.budget, bot.worlds, bot.backend) == (60, 2, "rust")
    assert bot.evaluator.describe() == "learned value (mlp)"
    other = configured("planner:logistic")(3)
    assert isinstance(other, PlannerBot)
    assert other.budget == planner.BUDGET


def test_tournament_pairs_the_planner_with_a_learned_value_against_the_planner(
    capsys,
):
    argv = ["tournament", "--bots", "planner:mlp", "planner", "--games", "2"]
    main([*argv, "--max-turns", "3", "--budget", "60", "--worlds", "2", "--sprt"])
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("planner:mlp vs planner: ")
    learned = PlannerBot(0, worlds=2, budget=60, evaluator=LINEAR).settings()
    assert f"planner: {learned}" in out
    assert any(
        line.startswith("planner:mlp: ") and "learned value (mlp)" in line
        for line in out
    )


def test_tournament_refuses_an_unknown_model_or_bot(capsys):
    for bot in ("planner:nonesuch", "planner:", "greedy:mlp"):
        with pytest.raises(SystemExit):
            main(["tournament", "--bots", bot, "greedy", "--games", "1"])
        assert bot in capsys.readouterr().err


def test_a_single_game_plays_the_planner_with_a_learned_value(capsys):
    main(["--seed", "7", "--bots", "planner:logistic", "greedy"])
    assert "wins on turn" in capsys.readouterr().out


def test_iterate_runs_rounds_into_its_directory_and_resumes(capsys, tmp_path):
    argv = [
        *("iterate", "--out", str(tmp_path), "--rounds", "1", "--threads", "2"),
        *("--train-games", "6", "--held-out-games", "4", "--gate-games", "4"),
        *("--ladder-games", "2", "--worlds", "1", "--budget", "40"),
        *("--epochs", "1", "--hidden", "3"),
    ]
    main(argv)
    out = capsys.readouterr().out
    assert "Round 1: champion mlp, pool mlp" in out
    assert "Stopped: 1 rounds played" in out
    main(argv)
    assert "Round" not in capsys.readouterr().out


def test_iterate_fits_candidates_of_several_members(capsys, tmp_path):
    argv = [
        *("iterate", "--out", str(tmp_path), "--rounds", "1", "--threads", "2"),
        *("--train-games", "6", "--held-out-games", "4", "--gate-games", "4"),
        *("--ladder-games", "2", "--worlds", "1", "--budget", "40"),
        *("--epochs", "1", "--hidden", "3", "--members", "2"),
    ]
    main(argv)
    assert "2 members" in capsys.readouterr().out
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["settings"]["members"] == 2


def test_iterate_takes_td_targets(capsys, tmp_path):
    argv = [
        *("iterate", "--out", str(tmp_path), "--rounds", "1", "--threads", "2"),
        *("--train-games", "6", "--held-out-games", "4", "--gate-games", "4"),
        *("--ladder-games", "2", "--worlds", "1", "--budget", "40"),
        *("--epochs", "1", "--hidden", "3", "--targets", "td", "--lam", "0.8"),
    ]
    main(argv)
    assert "TD(0.8) targets" in capsys.readouterr().out
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["settings"]["targets"] == "td"
    assert state["settings"]["lam"] == 0.8


def test_iterate_fits_with_the_hand_made_features(capsys, tmp_path):
    argv = [
        *("iterate", "--out", str(tmp_path), "--rounds", "1", "--threads", "2"),
        *("--train-games", "6", "--held-out-games", "4", "--gate-games", "4"),
        *("--ladder-games", "2", "--worlds", "1", "--budget", "40"),
        *("--epochs", "1", "--hidden", "3", "--extra"),
    ]
    main(argv)
    assert "hand-made features" in capsys.readouterr().out
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["settings"]["extra"] is True
    model = Model.load(tmp_path / "candidates" / "round-01-mlp.npz")
    assert model.extra == features.EXTRA


def test_iterate_fits_without_the_hand_made_features_by_default(capsys, tmp_path):
    argv = [
        *("iterate", "--out", str(tmp_path), "--rounds", "1", "--threads", "2"),
        *("--train-games", "6", "--held-out-games", "4", "--gate-games", "4"),
        *("--ladder-games", "2", "--worlds", "1", "--budget", "40"),
        *("--epochs", "1", "--hidden", "3"),
    ]
    main(argv)
    assert "hand-made features" not in capsys.readouterr().out
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["settings"]["extra"] is False


def test_iterate_refuses_unknown_targets(capsys, tmp_path):
    with pytest.raises(SystemExit):
        main(["iterate", "--targets", "sarsa"])
    assert "sarsa" in capsys.readouterr().err


def test_gap_reports_the_champion_s_chosen_and_held_out_positions(capsys):
    main(
        [
            *("gap", "--champion", "pf01", "--pool", "mlp", "pf01"),
            *("--games", "8", "--base-seed", "20", "--threads", "2"),
            *("--worlds", "1", "--budget", "60"),
        ]
    )
    out = capsys.readouterr().out
    assert "8 games (seeds 20-27): pf01 against itself, the heuristic" in out
    for heading in ("Chosen", "Held out"):
        assert f"{heading}: " in out
    assert "judged" in out
    assert "Calibration" in out
    assert "By game phase" in out
    assert "0.9-1.0" in out
    assert "turns 25+" in out


def test_probe_reports_the_champion_s_synergy_by_function_and_phase(capsys):
    argv = [
        *("probe", "synergy", "--model", "pf01", "--games", "2"),
        *("--worlds", "1", "--budget", "40", "--samples", "1"),
    ]
    main(argv)
    out = capsys.readouterr().out
    assert "pf01" in out
    assert "positions" in out
    for word in ("draw", "scrap", "discard", "none", "early", "middle", "late"):
        assert word in out
    assert "logit" in out


def test_probe_reports_the_champion_s_faction_pair_table(capsys):
    argv = [
        *("probe", "faction-pair", "--model", "pf01", "--games", "2"),
        *("--worlds", "1", "--budget", "40", "--samples", "1"),
    ]
    main(argv)
    out = capsys.readouterr().out
    assert "Faction-pair probe of pf01" in out
    for faction in ("Blob", "Trade Federation", "Star Empire", "Machine Cult"):
        assert faction in out
    assert "mostly of no one faction" in out
    assert "interaction" in out
    assert "logit" in out


def test_probe_reports_the_champion_s_denial_by_phase_beside_the_heuristic_s(
    capsys,
):
    argv = [
        *("probe", "denial", "--model", "pf01", "--games", "2"),
        *("--worlds", "1", "--budget", "40"),
    ]
    main(argv)
    out = capsys.readouterr().out
    assert "Denial probe of pf01" in out
    for word in ("early", "middle", "late", "logit", "seen"):
        assert word in out
    assert "buy" in out
    assert "heuristic" in out


def test_probe_investment_reports_trade_against_combat_and_the_buys(capsys):
    argv = [
        *("probe", "investment", "--model", "pf01", "--games", "2"),
        *("--worlds", "1", "--budget", "40"),
    ]
    main(argv)
    out = capsys.readouterr().out
    assert "Investment probe of pf01" in out
    for word in ("Federation Shuttle", "Blob Destroyer", "early", "late", "logit"):
        assert word in out
    assert "Buys by turn" in out
    for word in ("pf01", "heuristic", "economy", "combat", "mixed", "neither"):
        assert word in out
    assert "25+" in out


def test_worlds_compares_the_plans_by_count_then_plays_the_paired_match(
    capsys: pytest.CaptureFixture[str],
):
    main(
        [
            "worlds",
            "--model",
            "linear",
            "--counts",
            "1",
            "2",
            "--per-world",
            "60",
            "--games",
            "1",
            "--max-turns",
            "4",
            "--match-seeds",
            "1",
            "--match-base-seed",
            "5",
        ]
    )
    out = capsys.readouterr().out
    assert "the planner with the linear evaluation" in out
    assert "plans in 1 game" in out
    assert "2 worlds vs 1 worlds: 1 seed in both seats (seed 5)" in out
    assert "Seeds won both / split / lost both by 2 worlds:" in out
    assert "2 worlds:" in out and "ms per plan" in out
