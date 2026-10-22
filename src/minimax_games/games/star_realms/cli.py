"""Watch one seeded game between two bots, or run a tournament.

`python -m minimax_games.games.star_realms --seed 7 --log` prints the game's
transcript; without `--log` it prints only who won. `--bots greedy random`
picks the bots, first player first (both random by default): `random`,
`greedy`, `heuristic`, `sampler` (search over worlds sampled from its
view), `ismcts` (information-set Monte Carlo tree search from its
view) or `planner` (the whole-turn planner, over worlds sampled from its
view, scoring with `evaluate`). `planner:<model>` is the planner scoring with
a committed learned value instead (`value.load(<model>)`, e.g.
`planner:mlp`), on the Rust enumeration. `cheat` is a benchmark, not a
legitimate player: it searches the true state.

`python -m minimax_games.games.star_realms tournament --bots greedy random
--games 500` plays seeded games with alternating seats (`--base-seed`,
`--max-turns`) and prints each search bot's settings (the cheat's horizon
and node budget, the sampler's worlds too, the ISMCTS bot's iterations
and leaf settings, the planner's worlds and states per plan) if it plays,
win rates overall and by seat, unfinished games and each bot's mean time per
move, and the planner's plans per turn and time per plan. `--jobs N`
spreads the games over N processes, with the same results.
`--sprt` stops as soon as a sequential probability ratio test (`sprt.Sprt`)
decides between bot A winning `--h0` and `--h1` of its finished games, with
error rates `--alpha` and `--beta`, `--games` then being the most it plays;
it prints the decision, the games played, A's win rate and its 95%
interval, and the log-likelihood ratio and its bounds. `--budget N`
sets the search bots' nodes per decision (the planner's states per plan)
and `--worlds K` the sampler's and the planner's worlds (both default to
the bots' own defaults). `--eval rollout` has them score positions by
rollouts (`rollout.Rollout`: `--rollouts`,
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

`python -m minimax_games.games.star_realms turn-sizes` counts the distinct
ends of every turn of `--games` seeded heuristic-vs-heuristic games from
`--base-seed` (see `turn_sizes`), each turn stopped past `--max-states`
states, over `--jobs` processes: their distribution, the share of turns
over candidate state budgets, and both by turn number.

`python -m minimax_games.games.star_realms native-bench` times the Rust
engine (`minimax_games._native.star_realms`) on `--games` full-deck
greedy-vs-greedy games from `--base-seed`: games per second, `advance` calls
per second over their recorded decisions, and heuristic-vs-heuristic games
per second.

`python -m minimax_games.games.star_realms worlds` tests whether the
champion's plan changes with more sampled worlds (see `sampled_worlds`): in
`--games` self-play games of `planner:<--model>` (pf01 by default; `linear`
for the planner with the linear evaluation) from `--base-seed`, every plan is
made from each of `--counts` worlds (4, 8 and 16) at `--per-world` states per
world, and it prints the share of plans whose line changes from the first
count's, by turn number, and the time per plan at each count. With
`--match-seeds N`, it then plays a paired match (`tournament.play_paired`) of
the largest count against the first on N seeds from `--match-base-seed`,
each in both seats. `--jobs` spreads both over processes.

`python -m minimax_games.games.star_realms reply` tests whether looking
through the opponent's reply changes the champion's choice (see `reply`):
`planner:<--model>` (pf01) looking through the replies of its best
`--candidates` against itself at `--rival-worlds` (16) worlds and the same
80,000 states, `--seeds` seeds from `--base-seed`, each in both seats, in
one Rust batch on `--threads`. It prints the strength (win rates, the
seeds won both, split and lost both, the sign z, ms and states per plan)
and the changed choices by game phase, with the design's reading.

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

`python -m minimax_games.games.star_realms fit` fits learned values offline
(see `fit`): it plays `--games` heuristic-vs-heuristic games from
`--base-seed` and `--test-games` more after them, fits logistic regression
and a one-hidden-layer network (`--hidden`) to the turn-boundary states of
the first, and prints each one's held-out log-loss beside
`sigmoid(evaluate / 16)`'s on the same states. Deterministic given the
seeds; `--save DIR` writes the models there as `logistic.npz` and
`mlp.npz`.

`python -m minimax_games.games.star_realms iterate` runs policy iteration
(see `iterate`): rounds of Rust self-play with the current best value, a
refit of both forms on a window of recent rounds, an offline gate and an
SPRT match against the current best, into `--out` (`runs/iterate/` by
default, which git ignores), each round's records written as it finishes.
Run it again with the same `--out` to resume after the last finished round.
`--threads` plays the games on that many threads; `--rounds`, `--patience`
and `--hours` are the stopping rules; the round's sizes and the planner's
settings default to the agreed design's.

`python -m minimax_games.games.star_realms gap` measures the exploitation
gap (see `gap`): `--champion`'s games on `--games` seeds from `--base-seed`,
mixed as a policy-iteration round mixes them with `--pool`'s champions,
every end judged; it prints the gap, a calibration table and the gap by game
phase, for the ends the champion chose and for those its opponents chose.
`--every K` plays only every Kth game of that mix.

`python -m minimax_games.games.star_realms diagnose` plays the same games
(the same options; by default every 3rd of the 1,000, a third of the time)
and prints the other diagnostics (see `diagnostics`): the judgement's error
by each player's cards left before a reshuffle and with a rare card held or
not, and the share of the champion's turns not examined in full, by turn.

`python -m minimax_games.games.star_realms population PLAYER` plays a player
against the reference population (see `population`): `--games` games
against each member but itself on seeds 0 up, on `--threads` threads, with
the same results whatever the threads. It prints each opponent's win rate
with its interval and their mean, and writes the per-seed CSVs to `--out`
(`runs/population/<player>/` by default); `--paired DIR` pairs them seed by
seed with a recorded run's CSVs in DIR.

`python -m minimax_games.games.star_realms probe synergy` probes a
judgement (see `probe`): it plays `--games` of the champion (`--model`)
against itself in Rust on `--threads`, and reports what a first, second and
third card of each function adds to the judged chance of winning at every
turn's end of them, by game phase. `probe faction-pair` plays the same
games and reports, for every ordered pair of factions, what one card of the
second adds to decks mostly of the first: a 4-by-4 table with spreads, and
each pair's interaction. `probe denial` reports, from the same games, the
judged effect of a Trade Row card the next player would value against one
they wouldn't, by phase, and the share of the champion's buys and Trade Row
scraps that are denial, beside the heuristic's in its own games on the same
seeds. `probe investment` reports what a ship giving only trade and one of
the same cost giving only combat add per unit, by phase, and the share of
economy and combat buys by turn in those games and in the heuristic's
against itself on the same seeds.
"""

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from functools import cache, partial
from pathlib import Path
from typing import assert_never

from minimax_games._native import star_realms as native
from minimax_games.games.star_realms import (
    bench,
    cheat,
    diagnostics,
    fit,
    gap,
    iterate,
    planner,
    population,
    probe,
    profiling,
    reply,
    sampled_worlds,
    selfplay,
    tune,
    turn_sizes,
    value,
)
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
from minimax_games.games.star_realms.planner import PlannerBot
from minimax_games.games.star_realms.rollout import ROLLOUT_BUDGET, Rollout
from minimax_games.games.star_realms.sampler import WORLDS, SamplerBot
from minimax_games.games.star_realms.sprt import Hypothesis, Sprt, wilson
from minimax_games.games.star_realms.tournament import (
    MAX_TURNS,
    PairedTournament,
    SprtTournament,
    Tournament,
    play_paired,
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


def _planner(
    seed: int,
    budget: int = planner.BUDGET,
    worlds: int = planner.WORLDS,
    evaluator: Evaluator = LINEAR,
) -> PlannerBot:
    return PlannerBot(
        seed, worlds=worlds, budget=budget, evaluator=_seeded(evaluator, seed)
    )


@cache
def _value(model: str) -> value.ValueEvaluator:
    """The committed model `model` as an evaluator, loaded once a process."""
    return value.ValueEvaluator(value.load(model))


def _learned_planner(
    seed: int,
    model: str,
    budget: int = planner.BUDGET,
    worlds: int = planner.WORLDS,
) -> PlannerBot:
    return PlannerBot(
        seed, worlds=worlds, budget=budget, evaluator=_value(model), backend="rust"
    )


LEARNED = "planner:"
"""The prefix of a planner scoring with a committed learned value, by name:
`planner:mlp`."""


BOTS: dict[str, Callable[[int], AnyBot]] = {
    "random": RandomBot,
    "greedy": _greedy,
    "heuristic": _heuristic,
    "cheat": _cheat,
    "sampler": _sampler,
    "ismcts": _ismcts,
    "planner": _planner,
}
"""Each bot by name, built from a seed (which only `random`, `sampler`,
`ismcts` and `planner` use). Module
functions, not lambdas, so `tournament --jobs` can send them to workers."""


def bot_name(text: str) -> str:
    """`text` if it names a bot (`BOTS`, or `planner:<model>` for a committed
    model), for argparse: an unknown name or model is an argument error."""
    if text in BOTS:
        return text
    model = text.removeprefix(LEARNED)
    if model != text and model:
        try:
            _value(model)
        except FileNotFoundError as error:
            raise argparse.ArgumentTypeError(f"{text}: {error}") from None
        return text
    msg = f"no bot {text!r}: {_names()}, or {LEARNED}<model>"
    raise argparse.ArgumentTypeError(msg)


def configured(
    name: str,
    budget: int | None = None,
    worlds: int | None = None,
    evaluator: Evaluator = LINEAR,
    iterations: int | None = None,
) -> Callable[[int], AnyBot]:
    """`BOTS[name]`, with `budget` nodes per decision and `evaluator` if it is
    an alpha-beta search bot, `worlds` worlds if it is the sampler, `budget`
    states per plan, `worlds` worlds and `evaluator` if it is the planner, and
    `iterations` per decision if it is the ISMCTS bot (each left at its
    default if None). `planner:<model>` is the planner with the committed
    learned value `<model>` and the Rust enumeration, whatever `evaluator`.
    A rollout evaluator's seed is set to each game's. A `partial` of a
    module function, so `--jobs` can still send it to workers."""
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
    if name == "planner":
        return partial(
            _planner,
            budget=planner.BUDGET if budget is None else budget,
            worlds=planner.WORLDS if worlds is None else worlds,
            evaluator=evaluator,
        )
    if name.startswith(LEARNED):
        return partial(
            _learned_planner,
            model=name.removeprefix(LEARNED),
            budget=planner.BUDGET if budget is None else budget,
            worlds=planner.WORLDS if worlds is None else worlds,
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
    if argv[:1] == ["turn-sizes"]:
        sizing(argv[1:])
        return
    if argv[:1] == ["profile"]:
        profile(argv[1:])
        return
    if argv[:1] == ["tune"]:
        tuning(argv[1:])
        return
    if argv[:1] == ["fit"]:
        fitting(argv[1:])
        return
    if argv[:1] == ["iterate"]:
        iterating(argv[1:])
        return
    if argv[:1] == ["probe"]:
        probing(argv[1:])
        return
    if argv[:1] == ["native-bench"]:
        native_bench(argv[1:])
        return
    if argv[:1] == ["gap"]:
        print(gap.report(_measure(argv[1:], "gap", every=1)))
        return
    if argv[:1] == ["diagnose"]:
        print(diagnostics.report(_measure(argv[1:], "diagnose", every=3)))
        return
    if argv[:1] == ["population"]:
        populating(argv[1:])
        return
    if argv[:1] == ["worlds"]:
        worlds(argv[1:])
        return
    if argv[:1] == ["reply"]:
        replying(argv[1:])
        return
    parser = argparse.ArgumentParser(prog="python -m minimax_games.games.star_realms")
    parser.add_argument("--seed", type=int, default=0, help="the game's seed")
    parser.add_argument("--log", action="store_true", help="print the transcript")
    parser.add_argument(
        "--bots",
        nargs=2,
        type=bot_name,
        default=["random", "random"],
        metavar="BOT",
        help=f"the two bots, first player first: {_names()},"
        f" or {LEARNED}<model> (a committed learned value)",
    )
    args = parser.parse_args(argv)

    first, second = (
        (BOTS[name] if name in BOTS else configured(name))(args.seed + i)
        for i, name in enumerate(args.bots)
    )
    result = play_match((first, second), args.seed, log=print if args.log else None)
    if not args.log:
        print(f"Player {result.winner + 1} wins on turn {result.turns}")


def _measure(argv: list[str], command: str, every: int) -> gap.Measurement:
    """A champion's games, as the `gap` and `diagnose` commands play them
    (`gap.measure`)."""
    parser = argparse.ArgumentParser(
        prog=f"python -m minimax_games.games.star_realms {command}"
    )
    parser.add_argument(
        "--champion", default="pf01", help="the committed model (default: %(default)s)"
    )
    parser.add_argument(
        "--pool",
        nargs="+",
        default=["mlp", "pi03", "pf01"],
        help="the champions a quarter of the games are against, one drawn per"
        " game (default: %(default)s)",
    )
    parser.add_argument(
        "--games", type=int, default=1000, help="how many games (default: %(default)s)"
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=gap.SEEDS,
        help="the first game's seed (default: %(default)s)",
    )
    parser.add_argument(
        "--threads", type=int, default=1, help="threads to play on (default: 1)"
    )
    parser.add_argument(
        "--worlds",
        type=int,
        default=planner.WORLDS,
        help="the planners' worlds (default: %(default)s)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=planner.BUDGET,
        help="the planners' states per plan (default: %(default)s)",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=every,
        help="play only every Kth game of the mix (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    return gap.measure(
        args.champion,
        args.pool,
        range(args.base_seed, args.base_seed + args.games),
        threads=args.threads,
        worlds=args.worlds,
        budget=args.budget,
        every=args.every,
    )


def native_bench(argv: list[str]) -> None:
    """Time the Rust engine on full-deck greedy-vs-greedy games: games per
    second, and `advance` calls per second over their recorded decisions;
    then heuristic-vs-heuristic games per second."""
    parser = argparse.ArgumentParser(
        prog="python -m minimax_games.games.star_realms native-bench"
    )
    parser.add_argument("--games", type=int, default=2000, help="how many games")
    parser.add_argument("--base-seed", type=int, default=0, help="the first seed")
    args = parser.parse_args(argv)

    games, decisions, seconds = native.bench_games(args.games, args.base_seed)
    print(
        f"greedy games: {games} in {seconds:.3f} s, {games / seconds:,.0f} games/s,"
        f" {decisions / seconds:,.0f} decisions/s"
    )
    calls, seconds = native.bench_advance(args.games, args.base_seed)
    print(f"advance: {calls} calls in {seconds:.3f} s, {calls / seconds:,.0f} calls/s")
    games, decisions, seconds = native.bench_heuristic_games(args.games, args.base_seed)
    print(
        f"heuristic games: {games} in {seconds:.3f} s,"
        f" {games / seconds:,.0f} games/s, {decisions / seconds:,.0f} decisions/s"
    )


def tournament(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m minimax_games.games.star_realms tournament"
    )
    parser.add_argument(
        "--bots",
        nargs=2,
        type=bot_name,
        required=True,
        metavar="BOT",
        help=f"bots A and B (A moves first in even games): {_names()},"
        f" or {LEARNED}<model> (the planner with a committed learned value)",
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
        " rollout, about the same time per decision), and the planner's"
        f" states per plan, shared by its worlds (default: {planner.BUDGET})",
    )
    parser.add_argument(
        "--worlds",
        type=int,
        help="the sampler's worlds per decision and the planner's per plan"
        f" (default: {WORLDS} and {planner.WORLDS})",
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
    # The rollout evaluation's default budget is in alpha-beta nodes, not
    # in the planner's states.
    factories = {
        name: configured(
            name,
            args.budget if name == "planner" else budget,
            args.worlds,
            evaluator,
            args.iterations,
        )
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


def worlds(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m minimax_games.games.star_realms worlds"
    )
    parser.add_argument(
        "--model",
        default="pf01",
        help="the champion: a committed learned value, on the Rust enumeration,"
        " or 'linear' for the planner with the linear evaluation in Python"
        " (default: %(default)s)",
    )
    parser.add_argument(
        "--counts",
        nargs="+",
        type=int,
        default=list(sampled_worlds.COUNTS),
        metavar="WORLDS",
        help="the counts of worlds compared, today's first (default: %(default)s)",
    )
    parser.add_argument(
        "--per-world",
        type=int,
        default=sampled_worlds.PER_WORLD,
        help="states per world at every count (default: %(default)s)",
    )
    parser.add_argument("--games", type=int, default=10, help="self-play games probed")
    parser.add_argument(
        "--base-seed", type=int, default=0, help="the first probed game's seed"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=sampled_worlds.MAX_TURNS,
        help="stop a probed game after this many turns (default: %(default)s)",
    )
    parser.add_argument(
        "--match-seeds",
        type=int,
        default=0,
        help="seeds of the paired match of the largest count against the"
        " first, each played in both seats (default: none)",
    )
    parser.add_argument(
        "--match-base-seed", type=int, default=0, help="the match's first seed"
    )
    parser.add_argument("--jobs", type=int, default=1, help="processes (same results)")
    args = parser.parse_args(argv)
    if len(args.counts) < 2 or sorted(set(args.counts)) != args.counts:
        parser.error("--counts needs at least 2 counts, in increasing order")
    model = None if args.model == "linear" else args.model
    name = "planner" if model is None else bot_name(LEARNED + model)
    who = "the planner with the linear evaluation" if model is None else name
    print(
        f"Plans by count of worlds: {who} against itself, {args.games} games"
        f" from seed {args.base_seed}, {args.per_world} states per world,"
        f" at most {args.max_turns} turns"
    )
    records = sampled_worlds.measure(
        args.games,
        args.base_seed,
        args.counts,
        args.per_world,
        model,
        args.max_turns,
        args.jobs,
    )
    for line in sampled_worlds.report(records, args.counts):
        print(line)
    if not args.match_seeds:
        return
    counts = (args.counts[-1], args.counts[0])
    names = (f"{counts[0]} worlds", f"{counts[1]} worlds")
    bots = (
        configured(name, args.per_world * counts[0], counts[0]),
        configured(name, args.per_world * counts[1], counts[1]),
    )
    paired = play_paired(bots, args.match_seeds, args.match_base_seed, jobs=args.jobs)
    last = args.match_base_seed + args.match_seeds - 1
    one = args.match_seeds == 1
    seeds = f"seed {last}" if one else f"seeds {args.match_base_seed}-{last}"
    print()
    print(
        f"{names[0]} vs {names[1]}: {args.match_seeds} seed{'' if one else 's'}"
        f" in both seats ({seeds}), max {MAX_TURNS} turns"
    )
    for line in paired_report(paired, names):
        print(line)


def replying(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m minimax_games.games.star_realms reply"
    )
    default = selfplay.Reply()

    def option(flag: str, value: int, text: str) -> None:
        parser.add_argument(
            flag, type=int, default=value, help=f"{text} (default: %(default)s)"
        )

    parser.add_argument(
        "--model", default="pf01", help="the committed model (default: %(default)s)"
    )
    option("--seeds", 150, "seeds, each played in both seats")
    option("--base-seed", reply.SEEDS, "the first seed")
    option("--worlds", planner.WORLDS, "the reply player's worlds")
    option("--budget", planner.BUDGET, "the reply player's own states per plan")
    option("--candidates", default.candidates, "candidates looked through")
    option("--reply-worlds", default.worlds, "sub-worlds per reply")
    option("--reply-budget", default.budget, "states per reply, over its sub-worlds")
    option("--rival-worlds", 16, "the rival's worlds")
    option("--rival-budget", 80_000, "the rival's states per plan")
    option("--threads", 1, "threads to play on (same results)")
    option("--max-turns", reply.MAX_TURNS, "stop a game unfinished past this turn")
    args = parser.parse_args(argv)
    start = time.perf_counter()
    found = reply.measure(
        args.model,
        range(args.base_seed, args.base_seed + args.seeds),
        worlds=args.worlds,
        budget=args.budget,
        reply=selfplay.Reply(args.candidates, args.reply_worlds, args.reply_budget),
        rival_worlds=args.rival_worlds,
        rival_budget=args.rival_budget,
        threads=args.threads,
        max_turns=args.max_turns,
    )
    for line in reply.report(found):
        print(line)
    print(f"{time.perf_counter() - start:.0f} s wall on {args.threads} threads")


def paired_report(paired: PairedTournament, names: tuple[str, str]) -> list[str]:
    """A paired match's results as plain-text lines: bot A's win rate and its
    interval, how the seeds' pairs of games went (the sign z is (won both -
    lost both) / sqrt(their sum)), and the tournament's table."""
    result = paired.tournament.result
    wins = result.wins(0)
    low, high = wilson(wins, result.finished)
    both, split, neither = (paired.pairs[n] for n in (2, 1, 0))
    decided = both + neither
    z = (both - neither) / decided**0.5 if decided else 0.0
    rate = 100 * wins / result.finished if result.finished else 0.0
    return [
        f"{names[0]}: {wins}/{result.finished} ({rate:.1f}%),"
        f" 95% interval {100 * low:.1f}-{100 * high:.1f}%",
        f"Seeds won both / split / lost both by {names[0]}:"
        f" {both} / {split} / {neither} (sign z {z:.2f})",
        *report(paired.tournament, names),
    ]


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


def sizing(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m minimax_games.games.star_realms turn-sizes"
    )
    parser.add_argument(
        "--games", type=int, default=20, help="games (default: %(default)s)"
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="the first game's seed (default: %(default)s)",
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=turn_sizes.MAX_STATES,
        help="stop a turn past this many states (default: %(default)s)",
    )
    parser.add_argument(
        "--jobs", type=int, default=1, help="processes (default: %(default)s)"
    )
    args = parser.parse_args(argv)
    seeds = range(args.base_seed, args.base_seed + args.games)
    sizes = turn_sizes.turn_sizes(seeds, args.max_states, args.jobs)
    for line in turn_sizes.report(sizes, args.max_states):
        print(line)


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


def fitting(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m minimax_games.games.star_realms fit"
    )
    defaults = fit.Settings()
    parser.add_argument(
        "--games",
        type=int,
        default=defaults.train_games,
        help="training games (default: %(default)s)",
    )
    parser.add_argument(
        "--test-games",
        type=int,
        default=defaults.test_games,
        help="held-out games, the seeds after the training ones (default: %(default)s)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=defaults.base_seed,
        help="the first training game's seed (default: %(default)s)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=defaults.epsilon,
        help="the chance each move is random (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help="seeds the fits' starting weights and shuffles (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=defaults.epochs,
        help="passes over the training data (default: %(default)s)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=defaults.batch,
        help="minibatch size (default: %(default)s)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=defaults.learning_rate,
        help="Adam's step size (default: %(default)s)",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=defaults.l2,
        help="the weight penalty (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=defaults.hidden,
        help="the network's hidden units (default: %(default)s)",
    )
    parser.add_argument(
        "--jobs", type=int, default=1, help="play the games in this many processes"
    )
    parser.add_argument("--save", type=Path, help="write the models to this directory")
    args = parser.parse_args(argv)

    settings = fit.Settings(
        train_games=args.games,
        test_games=args.test_games,
        base_seed=args.base_seed,
        epsilon=args.epsilon,
        seed=args.seed,
        epochs=args.epochs,
        batch=args.batch,
        learning_rate=args.learning_rate,
        l2=args.l2,
        hidden=args.hidden,
        jobs=args.jobs,
    )
    result = fit.run(settings)
    for line in fit.report(result, settings):
        print(line)
    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        for name, model in result.models.items():
            path = args.save / f"{name}.npz"
            model.save(path)
            print(f"Wrote {path}")


def probing(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m minimax_games.games.star_realms probe"
    )
    parser.add_argument(
        "probe",
        choices=["synergy", "faction-pair", "denial", "investment"],
        help="the probe to run",
    )
    parser.add_argument(
        "--model", default="pf01", help="the champion's value (default: %(default)s)"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=200,
        help="the champion's games the positions come from (default: %(default)s)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=probe.BASE_SEED,
        help="the first game's seed (default: %(default)s)",
    )
    parser.add_argument(
        "--worlds",
        type=int,
        default=planner.WORLDS,
        help="the planner's worlds (default: %(default)s)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=planner.BUDGET,
        help="the planner's states per plan (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4,
        help="random draws of cards per position (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="seeds the draws (default: %(default)s)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="threads to play the games on (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    model = value.load(args.model)
    champion = selfplay.Planner(model, worlds=args.worlds, budget=args.budget)
    seeds = range(args.base_seed, args.base_seed + args.games)
    if args.probe == "denial":
        _denial(args, model, champion, seeds)
        return
    if args.probe == "investment":
        _investment(args, model, champion, seeds)
        return
    start = time.perf_counter()
    rows = probe.positions(champion, seeds, threads=args.threads)
    played = time.perf_counter() - start
    name = "Synergy" if args.probe == "synergy" else "Faction-pair"
    print(
        f"{name} probe of {args.model}: {len(rows)} positions from its games"
        f" against itself on seeds {seeds.start}-{seeds.stop - 1}"
        f" ({args.worlds} worlds, {args.budget} states per plan),"
        f" {args.samples} draws per position (seed {args.seed})"
    )
    if args.probe == "synergy":
        found = probe.synergy(model.predict, rows, seed=args.seed, samples=args.samples)
        lines = [*probe.report(found), "", *probe.report(found, logit=True)]
    else:
        pairs = probe.faction_pairs(
            model.predict, rows, seed=args.seed, samples=args.samples
        )
        lines = [
            probe.deck_counts(pairs),
            "",
            *probe.pairs_report(pairs),
            "",
            *probe.pairs_report(pairs, logit=True),
        ]
    probed = time.perf_counter() - start - played
    for line in lines:
        print(line)
    print(f"Games {played:.1f} s, probe {probed:.1f} s")


def _denial(
    args: argparse.Namespace,
    model: value.Model,
    champion: selfplay.Planner,
    seeds: range,
) -> None:
    start = time.perf_counter()
    played = probe.games(champion, seeds, threads=args.threads, record=True)
    heuristic = probe.games("heuristic", seeds, threads=args.threads, record=True)
    playing = time.perf_counter() - start
    found = probe.denial(model.predict, played.x)
    shares = {}
    for name, games in ((args.model, played), ("heuristic", heuristic)):
        chosen = probe.choices(games)
        shares[name] = probe.denial_shares(
            probe.worth(model.predict, games.x, chosen), chosen
        )
    probed = time.perf_counter() - start - playing
    print(
        f"Denial probe of {args.model}: {len(played.x)} positions from its games"
        f" against itself on seeds {seeds.start}-{seeds.stop - 1}"
        f" ({args.worlds} worlds, {args.budget} states per plan), and the"
        " heuristic's against itself on the same seeds"
    )
    for line in probe.denial_report(found):
        print(line)
    print()
    for line in probe.denial_report(found, logit=True):
        print(line)
    print()
    for line in probe.share_report(shares):
        print(line)
    print(f"Games {playing:.1f} s, probe {probed:.1f} s")


def _investment(
    args: argparse.Namespace,
    model: value.Model,
    champion: selfplay.Planner,
    seeds: range,
) -> None:
    start = time.perf_counter()
    played = probe.games(champion, seeds, threads=args.threads, record=True)
    heuristic = probe.games("heuristic", seeds, threads=args.threads, record=True)
    playing = time.perf_counter() - start
    assert played.actions is not None
    assert heuristic.actions is not None
    found = probe.investment(model.predict, played.x)
    probed = time.perf_counter() - start - playing
    print(
        f"Investment probe of {args.model}: {len(played.x)} positions from its"
        f" games against itself on seeds {seeds.start}-{seeds.stop - 1}"
        f" ({args.worlds} worlds, {args.budget} states per plan)"
    )
    for line in probe.investment_report(found):
        print(line)
    print()
    for line in probe.investment_report(found, logit=True):
        print(line)
    print()
    print(
        f"{args.model} against itself, and the heuristic against itself,"
        f" on seeds {seeds.start}-{seeds.stop - 1}"
    )
    players = {
        args.model: probe.buys(played.actions),
        "heuristic": probe.buys(heuristic.actions),
    }
    for line in probe.buys_report(players):
        print(line)
    print(f"Games {playing:.1f} s, probe {probed:.1f} s")


def iterating(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m minimax_games.games.star_realms iterate"
    )
    defaults = iterate.Settings()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/iterate"),
        help="the run's directory, resumed if it has rounds (default: %(default)s)",
    )
    for flag, field, text in (
        ("--threads", "threads", "threads playing the games"),
        ("--rounds", "max_rounds", "stop after this many rounds"),
        ("--patience", "patience", "stop after this many rounds without promotion"),
        ("--train-games", "train_games", "training games per round"),
        ("--held-out-games", "held_out_games", "held-out games per round"),
        ("--window", "window", "rounds of training games each fit sees"),
        ("--gate-games", "gate_games", "the match's most games"),
        ("--ladder-games", "ladder_games", "a promoted value's games vs heuristic"),
        ("--worlds", "worlds", "the planner's worlds"),
        ("--budget", "budget", "the planner's states per plan"),
        ("--epochs", "epochs", "the fits' passes over the data"),
        ("--hidden", "hidden", "the network's hidden units"),
        ("--members", "members", "fits per candidate, from consecutive seeds"),
    ):
        parser.add_argument(
            flag,
            type=int,
            dest=field,
            default=getattr(defaults, field),
            help=f"{text} (default: %(default)s)",
        )
    parser.add_argument(
        "--hours",
        type=float,
        default=defaults.hours,
        help="start no round once the rounds have taken this long"
        " (default: %(default)s)",
    )
    parser.add_argument(
        "--targets",
        choices=iterate.TARGETS,
        default=defaults.targets,
        help="fit to game results (mc) or TD(lam) returns bootstrapping from"
        " the current best (td) (default: %(default)s)",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=defaults.lam,
        help="the lambda of TD targets (default: %(default)s)",
    )
    parser.add_argument(
        "--extra",
        action="store_true",
        default=defaults.extra,
        help="fit values that read the hand-made features too (features.EXTRA)",
    )
    parser.add_argument(
        "--prefix",
        default=defaults.prefix,
        help="round r's candidate is named <prefix><r> (default: %(default)s)",
    )
    parser.add_argument(
        "--start",
        type=lambda name: bot_name(LEARNED + name).removeprefix(LEARNED),
        default=defaults.start,
        help="the starting champion, a committed model (default: %(default)s)",
    )
    args = vars(parser.parse_args(argv))
    out = args.pop("out")
    settings = iterate.Settings(**args)
    iterate.run(settings, out)


def populating(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m minimax_games.games.star_realms population"
    )
    parser.add_argument(
        "player",
        help="heuristic, greedy, a committed model (played by the planner with"
        " it) or the path of a saved model (.npz)",
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=list(population.POPULATION),
        help="the opponents, the player itself left out"
        " (default: the reference population, %(default)s)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=population.GAMES,
        help="games against each opponent, on seeds 0 up (default: %(default)s)",
    )
    parser.add_argument(
        "--threads", type=int, default=8, help="threads (default: %(default)s)"
    )
    parser.add_argument(
        "--worlds",
        type=int,
        default=planner.WORLDS,
        help="the planners' worlds (default: %(default)s)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=planner.BUDGET,
        help="the planners' states per plan (default: %(default)s)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=fit.MAX_TURNS,
        help="games past this many turns are unfinished (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="where the per-seed CSVs go, one per opponent"
        " (default: runs/population/<player>)",
    )
    parser.add_argument(
        "--paired",
        type=Path,
        help="a recorded run's directory (e.g. docs/population/pf01): pair"
        " with its games against each opponent, seed by seed",
    )
    args = parser.parse_args(argv)
    name = population.name(args.player)
    out = args.out or Path("runs/population") / name
    start = time.perf_counter()
    matches = population.play(
        args.player,
        opponents=args.opponents,
        games=args.games,
        threads=args.threads,
        worlds=args.worlds,
        budget=args.budget,
        max_turns=args.max_turns,
    )
    seconds = time.perf_counter() - start
    population.write(matches, out)
    paired = []
    missing = []
    for match in matches:
        recorded = args.paired / f"{match.opponent}.csv" if args.paired else None
        if recorded is None:
            continue
        if recorded.is_file():
            paired.append(population.pair(match, population.read(recorded)))
        else:
            missing.append(f"{name} against {match.opponent}: no recorded results")
    print(
        f"{name} against the reference population: {args.games} games each,"
        f" seeds 0-{args.games - 1}, seats alternating; planners {args.worlds}"
        f" worlds, {args.budget} states per plan; max {args.max_turns} turns"
    )
    for line in population.report(matches, paired):
        print(line)
    for line in missing:
        print(line)
    print(f"Per-seed results: {out}/")
    print(f"Wall time: {seconds:.1f}s on {args.threads} threads")


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
    plans = [
        f"{name}: {timing.plans_per_turn(bot):.2f} plans per turn,"
        f" {timing.ms_per_plan(bot):.3g} ms per plan"
        for bot, name in enumerate(names)
        if timing.plans[bot]
    ]
    return [
        f"Unfinished: {len(unfinished)}{seeds}",
        f"Mean turns: {result.mean_turns:.1f} (finished games)",
        f"Win rates are of the {result.finished} finished games.",
        "",
        *table,
        "",
        *plans,
        f"Wall time: {timing.wall_seconds:.1f}s ({games_per_second:.1f} games/s)",
    ]
