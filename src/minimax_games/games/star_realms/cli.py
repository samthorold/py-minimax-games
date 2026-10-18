"""Watch one seeded game between two bots, or run a tournament.

`python -m minimax_games.games.star_realms --seed 7 --log` prints the game's
transcript; without `--log` it prints only who won. `--bots greedy random`
picks the bots, first player first (both random by default): `random`,
`greedy` or `heuristic`. `cheat` is a benchmark, not a legitimate player: it
searches the true state.

`python -m minimax_games.games.star_realms tournament --bots greedy random
--games 500` plays seeded games with alternating seats (`--base-seed`,
`--max-turns`) and prints the cheat's horizon and node budget if it plays,
win rates overall and by seat, unfinished games and each bot's mean time per
move. `--jobs N` spreads the games over N processes, with the same results.
Game `seed` replays on its own with `--seed` and its bots in seat order,
first player first: bot A moves first in games `base_seed`,
`base_seed + 2`, ...

`python -m minimax_games.games.star_realms bench` times the cheat's search on
fixed seeded positions (see `bench`): nodes, nodes per second and time per
decision at each horizon, each search stopped past `--max-nodes`. `--states`
also counts distinct states, to measure transpositions. `--budget N` times
the cheat's budgeted decisions instead (`cheat.decide`, deepening up to the
largest horizon within N nodes): how deep each got, its nodes and time.

`python -m minimax_games.games.star_realms profile` profiles the cheat's
budgeted decisions (see `profiling`) at the same positions and in the first
`--game-turns` turns of a seeded cheat-vs-heuristic game, and writes a report
(`profile.txt`) and the raw profile (`profile.pstats`) to `--out`
(`profiles/` by default, which git ignores).
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from minimax_games.games.star_realms import bench, cheat, profiling
from minimax_games.games.star_realms.bots import (
    AnyBot,
    GreedyBot,
    HeuristicBot,
    RandomBot,
)
from minimax_games.games.star_realms.cheat import CheatBot
from minimax_games.games.star_realms.match import play_match
from minimax_games.games.star_realms.tournament import (
    MAX_TURNS,
    Tournament,
    play_tournament,
)


def _greedy(seed: int) -> GreedyBot:
    return GreedyBot()


def _heuristic(seed: int) -> HeuristicBot:
    return HeuristicBot()


def _cheat(seed: int) -> CheatBot:
    return CheatBot()


BOTS: dict[str, Callable[[int], AnyBot]] = {
    "random": RandomBot,
    "greedy": _greedy,
    "heuristic": _heuristic,
    "cheat": _cheat,
}
"""Each bot by name, built from a seed (which only `random` uses). Module
functions, not lambdas, so `tournament --jobs` can send them to workers."""

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
    parser.add_argument("--games", type=int, default=500, help="how many games")
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
    args = parser.parse_args(argv)

    a, b = args.bots
    played = play_tournament(
        (BOTS[a], BOTS[b]), args.games, args.base_seed, args.max_turns, args.jobs
    )
    last = args.base_seed + args.games - 1
    print(
        f"{a} vs {b}: {args.games} games, seeds {args.base_seed}-{last},"
        f" max {args.max_turns} turns"
    )
    for name in dict.fromkeys(args.bots):
        bot = BOTS[name](args.base_seed)
        if isinstance(bot, CheatBot):
            print(
                f"{name}: {bot.turns}-turn horizon,"
                f" budget {bot.budget} nodes per decision"
            )
    for line in report(played, (a, b) if a != b else (f"{a}-A", f"{b}-B")):
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
    args = parser.parse_args(argv)

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
