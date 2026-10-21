"""Watch one seeded game between two bots, or run a tournament.

`python -m minimax_games.games.star_realms --seed 7 --log` prints the game's
transcript; without `--log` it prints only who won. `--bots greedy random`
picks the bots, first player first (both random by default): `random`,
`greedy`, `heuristic`, `sampler` (search over worlds sampled from its
view) or `ismcts` (information-set Monte Carlo tree search from its
view). `cheat` is a benchmark, not a legitimate player: it searches the
true state.

`python -m minimax_games.games.star_realms tournament --bots greedy random
--games 500` plays seeded games with alternating seats (`--base-seed`,
`--max-turns`) and prints each search bot's settings (the cheat's horizon
and node budget, the sampler's worlds too, the ISMCTS bot's iterations
and leaf settings) if it plays,
win rates overall and by seat, unfinished games and each bot's mean time per
move. `--jobs N` spreads the games over N processes, with the same results.
`--sprt` stops as soon as a sequential probability ratio test (`sprt.Sprt`)
decides between bot A winning `--h0` and `--h1` of its finished games, with
error rates `--alpha` and `--beta`, `--games` then being the most it plays;
it prints the decision, the games played, A's win rate and its 95%
interval, and the log-likelihood ratio and its bounds.
`--budget N` sets the search bots' nodes per decision and `--worlds K` the
sampler's worlds (both default to the bots' own defaults). `--eval rollout`
has them score positions by rollouts (`rollout.Rollout`: `--rollouts`,
`--rollout-turns`, `--epsilon`, seeded per game) instead of the linear
evaluation, with `rollout.ROLLOUT_BUDGET` nodes unless `--budget` says.
`--iterations N` sets the ISMCTS bot's iterations per decision.
Game `seed` replays on its own with `--seed` and its bots in seat order,
first player first: bot A moves first in games `base_seed`,
`base_seed + 2`, ...

`python -m minimax_games.games.star_realms bench` times the cheat's search on
fixed seeded positions (see `bench`): nodes, nodes per second and time per
decision at each horizon, each search stopped past `--max-nodes`. `--states`
also counts distinct states, to measure transpositions. `--budget N` times
the cheat's budgeted decisions instead (`cheat.decide`, deepening up to the
largest horizon within N nodes): how deep each got, its nodes and time.
With `--worlds K` too, it times the sampler's decisions (`sampler.vote`, K
worlds from the mover's view sharing the N nodes): how deep each world's
search got, the nodes and time.

`python -m minimax_games.games.star_realms profile` profiles the cheat's
budgeted decisions (see `profiling`) at the same positions and in the first
`--game-turns` turns of a seeded cheat-vs-heuristic game, and writes a report
(`profile.txt`) and the raw profile (`profile.pstats`) to `--out`
(`profiles/` by default, which git ignores).

`python -m minimax_games.games.star_realms tune` tunes `evaluate`'s weights
by tournament (see `tune`): a coordinate search over `--fields` (all by
default, in order), each weight multiplied and divided by `--factor`, the
cheat with each candidate playing the heuristic on `--games` games from
`--base-seed` at `--budget` nodes, a change kept only if its extra wins on
the same seeds pass `--z` standard deviations of noise. It prints each step.
`--cache FILE` keeps every trial, so a stopped run resumes. `--screen` runs
the cheap screen instead: per weight, the share of the cheat's decisions
(every `--every`-th, in the games from `--base-seed`) its step changes.
"""

import argparse
import sys
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import assert_never

from minimax_games.games.star_realms import bench, cheat, profiling, tune
from minimax_games.games.star_realms.bots import (
    AnyBot,
    Budgeted,
    GreedyBot,
    HeuristicBot,
    RandomBot,
)
from minimax_games.games.star_realms.cheat import CheatBot
from minimax_games.games.star_realms.evaluate import LINEAR, Evaluator
from minimax_games.games.star_realms.ismcts import ITERATIONS, IsmctsBot
from minimax_games.games.star_realms.match import play_match
from minimax_games.games.star_realms.rollout import ROLLOUT_BUDGET, Rollout
from minimax_games.games.star_realms.sampler import WORLDS, SamplerBot
from minimax_games.games.star_realms.sprt import Hypothesis, Sprt, wilson
from minimax_games.games.star_realms.tournament import (
    MAX_TURNS,
    SprtTournament,
    Tournament,
    play_sprt,
    play_tournament,
)


def _greedy(seed: int) -> GreedyBot:
    return GreedyBot()


def _heuristic(seed: int) -> HeuristicBot:
    return HeuristicBot()


def _seeded(evaluator: Evaluator, seed: int) -> Evaluator:
    """`evaluator`, its rollouts (if any) seeded from the game's `seed`."""
    return (
        replace(evaluator, seed=seed) if isinstance(evaluator, Rollout) else evaluator
    )


def _cheat(
    seed: int, budget: int = cheat.BUDGET, evaluator: Evaluator = LINEAR
) -> CheatBot:
    return CheatBot(budget=budget, evaluator=_seeded(evaluator, seed))


def _sampler(
    seed: int,
    budget: int = cheat.BUDGET,
    worlds: int = WORLDS,
    evaluator: Evaluator = LINEAR,
) -> SamplerBot:
    return SamplerBot(
        seed, worlds=worlds, budget=budget, evaluator=_seeded(evaluator, seed)
    )


def _ismcts(seed: int, iterations: int = ITERATIONS) -> IsmctsBot:
    return IsmctsBot(seed, iterations=iterations)


BOTS: dict[str, Callable[[int], AnyBot]] = {
    "random": RandomBot,
    "greedy": _greedy,
    "heuristic": _heuristic,
    "cheat": _cheat,
    "sampler": _sampler,
    "ismcts": _ismcts,
}
"""Each bot by name, built from a seed (which only `random`, `sampler` and
`ismcts` use). Module
functions, not lambdas, so `tournament --jobs` can send them to workers."""


def configured(
    name: str,
    budget: int | None = None,
    worlds: int | None = None,
    evaluator: Evaluator = LINEAR,
    iterations: int | None = None,
) -> Callable[[int], AnyBot]:
    """`BOTS[name]`, with `budget` nodes per decision and `evaluator` if it is
    an alpha-beta search bot, `worlds` worlds if it is the sampler, and
    `iterations` per decision if it is the ISMCTS bot (each left at its
    default if None). A rollout evaluator's seed is set to each game's. A
    `partial` of a module function, so `--jobs` can still send it to
    workers."""
    if name == "cheat":
        return partial(
            _cheat,
            budget=cheat.BUDGET if budget is None else budget,
            evaluator=evaluator,
        )
    if name == "sampler":
        return partial(
            _sampler,
            budget=cheat.BUDGET if budget is None else budget,
            worlds=WORLDS if worlds is None else worlds,
            evaluator=evaluator,
        )
    if name == "ismcts":
        return partial(
            _ismcts, iterations=ITERATIONS if iterations is None else iterations
        )
    return BOTS[name]


BENCHMARK_ONLY = {"cheat"}
"""Bots that see the true state: benchmarks, not legitimate players."""


def _names() -> str:
    """The bots' names for help texts, benchmark-only ones marked."""
    return ", ".join(
        f"{name} (benchmark only: sees the whole state)"
        if name in BENCHMARK_ONLY
        else name
        for name in BOTS
    )


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["tournament"]:
        tournament(argv[1:])
        return
    if argv[:1] == ["bench"]:
        benchmark(argv[1:])
        return
    if argv[:1] == ["profile"]:
        profile(argv[1:])
        return
    if argv[:1] == ["tune"]:
        tuning(argv[1:])
        return
    parser = argparse.ArgumentParser(prog="python -m minimax_games.games.star_realms")
    parser.add_argument("--seed", type=int, default=0, help="the game's seed")
    parser.add_argument("--log", action="store_true", help="print the transcript")
    parser.add_argument(
        "--bots",
        nargs=2,
        choices=BOTS,
        default=["random", "random"],
        metavar="BOT",
        help=f"the two bots, first player first: {_names()}",
    )
    args = parser.parse_args(argv)

    first, second = (BOTS[name](args.seed + i) for i, name in enumerate(args.bots))
    result = play_match((first, second), args.seed, log=print if args.log else None)
    if not args.log:
        print(f"Player {result.winner + 1} wins on turn {result.turns}")


def tournament(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m minimax_games.games.star_realms tournament"
    )
    parser.add_argument(
        "--bots",
        nargs=2,
        choices=BOTS,
        required=True,
        metavar="BOT",
        help=f"bots A and B (A moves first in even games): {_names()}",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=500,
        help="how many games (with --sprt, the most it plays)",
    )
    parser.add_argument("--base-seed", type=int, default=0, help="the first seed")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=MAX_TURNS,
        help="games past this many turns are unfinished",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="play the games in this many processes (same results)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        help="the search bots' nodes per decision, shared by the sampler's"
        f" worlds (default: {cheat.BUDGET}, or {ROLLOUT_BUDGET} with --eval"
        " rollout, about the same time per decision)",
    )
    parser.add_argument(
        "--worlds",
        type=int,
        help=f"the sampler's worlds per decision (default: {WORLDS})",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help=f"the ISMCTS bot's iterations per decision (default: {ITERATIONS})",
    )
    parser.add_argument(
        "--eval",
        choices=("linear", "rollout"),
        default="linear",
        help="how the search bots score positions: the linear evaluation, or"
        " rollouts (playouts with the greedy policy; default: %(default)s)",
    )
    parser.add_argument(
        "--sprt",
        action="store_true",
        help="stop as soon as a sequential probability ratio test decides"
        " between --h0 and --h1",
    )
    test = Sprt()
    parser.add_argument(
        "--h0",
        type=float,
        default=test.p0,
        help="with --sprt, bot A's win rate under H0 (default: %(default)s)",
    )
    parser.add_argument(
        "--h1",
        type=float,
        default=test.p1,
        help="with --sprt, bot A's win rate under H1 (default: %(default)s)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=test.alpha,
        help="with --sprt, the chance of accepting H1 if H0 is true"
        " (default: %(default)s)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=test.beta,
        help="with --sprt, the chance of accepting H0 if H1 is true"
        " (default: %(default)s)",
    )
    defaults = Rollout()
    parser.add_argument(
        "--rollouts",
        type=int,
        default=defaults.rollouts,
        help="with --eval rollout, playouts per position (default: %(default)s)",
    )
    parser.add_argument(
        "--rollout-turns",
        type=int,
        default=defaults.turns,
        help="with --eval rollout, turn ends each playout plays (default: %(default)s)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=defaults.epsilon,
        help="with --eval rollout, the chance a playout move is random"
        " (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    evaluator: Evaluator = LINEAR
    budget = args.budget
    if args.eval == "rollout":
        evaluator = Rollout(
            rollouts=args.rollouts, turns=args.rollout_turns, epsilon=args.epsilon
        )
        if budget is None:
            budget = ROLLOUT_BUDGET
    a, b = args.bots
    factories = {
        name: configured(name, budget, args.worlds, evaluator, args.iterations)
        for name in dict.fromkeys(args.bots)
    }
    names = (a, b) if a != b else (f"{a}-A", f"{b}-B")
    bots = (factories[a], factories[b])
    sprt = None
    if args.sprt:
        try:
            test = Sprt(p0=args.h0, p1=args.h1, alpha=args.alpha, beta=args.beta)
        except ValueError as error:
            parser.error(str(error))
        sprt = play_sprt(
            bots, test, args.games, args.base_seed, args.max_turns, args.jobs
        )
        played = sprt.tournament
    else:
        played = play_tournament(
            bots, args.games, args.base_seed, args.max_turns, args.jobs
        )
    last = args.base_seed + args.games - 1
    print(
        f"{a} vs {b}: {'SPRT, at most ' if args.sprt else ''}{args.games} games,"
        f" seeds {args.base_seed}-{last}, max {args.max_turns} turns"
    )
    for name, factory in factories.items():
        bot = factory(args.base_seed)
        if isinstance(bot, Budgeted):
            print(f"{name}: {bot.settings()}")
    if sprt is not None:
        for line in sprt_report(sprt, names[0], args.games):
            print(line)
    for line in report(played, names):
        print(line)


def _seed_turn(text: str) -> tuple[int, int]:
    seed, _, turn = text.partition(":")
    return int(seed), int(turn)


def benchmark(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m minimax_games.games.star_realms bench"
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        type=_seed_turn,
        default=bench.POSITIONS,
        metavar="SEED:TURN",
        help="start of TURN in seed SEED's greedy-vs-greedy game"
        " (default: %(default)s)",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=bench.HORIZONS,
        metavar="TURNS",
        help="search depths in turns (default: %(default)s)",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=bench.MAX_NODES,
        help="stop a search past this many nodes (default: %(default)s)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        help="time the cheat's budgeted decisions instead, deepening up to the"
        " largest horizon within BUDGET nodes per decision",
    )
    parser.add_argument(
        "--states",
        action="store_true",
        help="also count distinct states, in each search and in each"
        " position's whole first turn (slows the search: time without)",
    )
    parser.add_argument(
        "--worlds",
        type=int,
        help="with --budget, time the sampler's decisions instead, sampling"
        " WORLDS worlds from the mover's view and sharing the budget",
    )
    args = parser.parse_args(argv)

    if args.worlds is not None and args.budget is None:
        parser.error("--worlds needs --budget")
    if args.worlds is not None:
        turns = max(args.horizons)
        print(
            f"Sampler decision benchmark: {args.worlds} worlds,"
            f" horizon {turns} turns, budget {args.budget} nodes per decision"
            f" ({args.budget // args.worlds} per world)"
        )
        voted = bench.sampler_decisions(
            tuple(args.positions), args.worlds, turns, args.budget
        )
        for line in bench.vote_report(voted, turns):
            print(line)
        return
    if args.budget is not None:
        turns = max(args.horizons)
        print(
            f"Cheat decision benchmark: horizon {turns} turns,"
            f" budget {args.budget} nodes per decision"
        )
        timed = bench.decisions(tuple(args.positions), turns, args.budget)
        for line in bench.decision_report(timed, turns):
            print(line)
        return
    print(
        "Cheat search benchmark: horizons"
        f" {', '.join(map(str, args.horizons))} turns,"
        f" max {args.max_nodes} nodes per search"
    )
    entries = bench.run(
        tuple(args.positions),
        tuple(args.horizons),
        args.max_nodes,
        count_states=args.states,
    )
    for line in bench.report(entries):
        print(line)
    if args.states:
        print()
        for seed, turn in args.positions:
            tree = bench.turn_tree(bench.position(seed, turn))
            print(
                f"Seed {seed} turn {turn}, one whole turn: {tree.nodes} nodes,"
                f" {tree.states} distinct states ({tree.nodes / tree.states:.1f}x)"
            )


def profile(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m minimax_games.games.star_realms profile"
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        type=_seed_turn,
        default=bench.POSITIONS,
        metavar="SEED:TURN",
        help="start of TURN in seed SEED's greedy-vs-greedy game"
        " (default: %(default)s)",
    )
    parser.add_argument(
        "--turns", type=int, default=2, help="the cheat's horizon (default: 2)"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=cheat.BUDGET,
        help="nodes per decision (default: %(default)s)",
    )
    parser.add_argument(
        "--game-seed",
        type=int,
        default=0,
        help="the seed of the cheat-vs-heuristic game (default: %(default)s)",
    )
    parser.add_argument(
        "--game-turns",
        type=int,
        default=6,
        help="also profile the cheat's decisions in the first GAME_TURNS turns"
        " of that game, the cheat moving first; 0 for none (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("profiles"),
        help="where to write the report (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    states = [bench.position(seed, turn) for seed, turn in args.positions]
    game = profiling.game_decisions(
        args.game_seed, args.game_turns, args.budget, args.turns
    )
    positions = len(states)
    workload = (
        f"Cheat decisions profiled: {positions + len(game)}"
        f" ({positions} benchmark position{'s' if positions != 1 else ''},"
        f" {len(game)} from seed {args.game_seed}'s cheat-vs-heuristic game,"
        f" turns 1-{args.game_turns}), horizon {args.turns} turns,"
        f" budget {args.budget} nodes"
    )
    print(workload)
    profiled = profiling.profile([*states, *game], args.turns, args.budget)
    args.out.mkdir(parents=True, exist_ok=True)
    text = args.out / "profile.txt"
    text.write_text("\n".join(profiling.report(profiled, workload)) + "\n")
    profiled.stats.dump_stats(args.out / "profile.pstats")
    print(f"Wrote {text}")


def tuning(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m minimax_games.games.star_realms tune"
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        choices=tune.FIELDS,
        default=list(tune.FIELDS),
        metavar="WEIGHT",
        help="the weights to tune, in order (default: all, in Weights' order)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=3000,
        help="the first seed (default: %(default)s)",
    )
    parser.add_argument(
        "--games", type=int, default=200, help="games per trial (default: %(default)s)"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=cheat.BUDGET,
        help="the cheat's nodes per decision (default: %(default)s)",
    )
    parser.add_argument("--jobs", type=int, default=1, help="processes (same results)")
    parser.add_argument(
        "--factor",
        type=float,
        default=tune.FACTOR,
        help="each step multiplies or divides a weight by this (default: %(default)s)",
    )
    parser.add_argument(
        "--z",
        type=float,
        default=tune.Z,
        help="standard deviations of noise a gain must pass (default: %(default)s)",
    )
    parser.add_argument(
        "--passes", type=int, default=1, help="passes over the weights (default: 1)"
    )
    parser.add_argument(
        "--cache", type=Path, help="keep each trial in this file, to resume a run"
    )
    parser.add_argument(
        "--screen",
        action="store_true",
        help="screen the weights instead: how often each one's step changes a move",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=10,
        help="with --screen, every EVERY-th decision of the --games games"
        " (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    last = args.base_seed + args.games - 1
    if args.screen:
        states = tune.decision_states(
            range(args.base_seed, last + 1), args.budget, args.every
        )
        print(
            f"Screen: {len(states)} cheat decisions (every {args.every}th, seeds"
            f" {args.base_seed}-{last}), budget {args.budget}, factor {args.factor:g}"
        )
        shares = tune.screen(
            states, args.budget, args.fields, factor=args.factor, jobs=args.jobs
        )
        for field, share in shares.items():
            print(f"{field}: {100 * share:.1f}% of decisions changed")
        return
    print(
        f"Tuning {', '.join(args.fields)}: cheat vs heuristic, {args.games} games"
        f" per trial, seeds {args.base_seed}-{last}, budget {args.budget},"
        f" factor {args.factor:g}, z {args.z:g}, up to {args.passes} passes"
    )
    games = partial(
        tune.play,
        base_seed=args.base_seed,
        games=args.games,
        budget=args.budget,
        jobs=args.jobs,
    )
    played = games
    if args.cache is not None:
        cache = tune.TrialCache(
            args.cache,
            {"base_seed": args.base_seed, "games": args.games, "budget": args.budget},
        )
        played = partial(cache.play, play=games)
    tuned = tune.search(
        played,
        args.fields,
        factor=args.factor,
        z=args.z,
        passes=args.passes,
        log=lambda line: print(line, flush=True),
    )
    print(f"Weights: {tuned}")


def sprt_report(played: SprtTournament, name: str, max_games: int) -> list[str]:
    """The test, its decision and the evidence for it, as plain-text lines:
    `name` is bot A's."""
    test, result = played.test, played.tournament.result
    games = len(result.games)
    match played.verdict:
        case Hypothesis.H0 | Hypothesis.H1 as verdict:
            decision = f"{verdict.value} accepted after {games} games"
        case None:
            limit = " (the limit)" if games == max_games else ""
            decision = f"undecided after {games} games{limit}"
        case _:
            assert_never(played.verdict)
    wins, losses = result.wins(0), result.wins(1)
    low, high = wilson(wins, result.finished)
    rate = wins / result.finished if result.finished else 0.0
    return [
        f"SPRT: H0 {name} wins {test.p0:.1%}, H1 {test.p1:.1%};"
        f" alpha {test.alpha}, beta {test.beta}",
        f"Decision: {decision}",
        f"LLR {test.llr(wins, losses):.2f}, bounds [{test.lower:.2f},"
        f" {test.upper:.2f}]",
        f"{name} win rate {rate:.1%} (95% interval {low:.1%}-{high:.1%})"
        f" of {result.finished} finished games",
    ]


def report(played: Tournament, names: tuple[str, str]) -> list[str]:
    """The tournament's results as plain-text lines: a summary and a table."""
    result, timing = played.result, played.timing
    unfinished = result.unfinished
    seeds = f" (seeds {', '.join(map(str, unfinished))})" if unfinished else ""

    def rate(wins: int, games: int) -> str:
        return f"{100 * wins / games:.1f}%" if games else "-"

    rows = [("Bot", "Wins", "Win%", "As 1st", "As 2nd", "ms/move")]
    for bot, name in enumerate(names):
        wins = result.wins(bot)
        seats = [result.record(bot, seat) for seat in (0, 1)]
        rows.append(
            (
                name,
                f"{wins}/{result.finished}",
                rate(wins, result.finished),
                *(f"{w}/{n} {rate(w, n)}" for w, n in seats),
                f"{timing.ms_per_move[bot]:.3g}",
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    table = [
        "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip()
        for row in rows
    ]
    games_per_second = (
        len(result.games) / timing.wall_seconds if timing.wall_seconds else 0.0
    )
    return [
        f"Unfinished: {len(unfinished)}{seeds}",
        f"Mean turns: {result.mean_turns:.1f} (finished games)",
        f"Win rates are of the {result.finished} finished games.",
        "",
        *table,
        "",
        f"Wall time: {timing.wall_seconds:.1f}s ({games_per_second:.1f} games/s)",
    ]
