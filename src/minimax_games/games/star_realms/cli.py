"""Watch one seeded game between two bots, or run a tournament.

`python -m minimax_games.games.star_realms --seed 7 --log` prints the game's
transcript; without `--log` it prints only who won. `--bots greedy random`
picks the bots, first player first (both random by default).

`python -m minimax_games.games.star_realms tournament --bots greedy random
--games 500` plays seeded games with alternating seats (`--base-seed`,
`--max-turns`) and prints win rates overall and by seat, unfinished games and
each bot's mean time per move. Game `seed` replays on its own with `--seed`
and its bots in seat order, first player first: bot A moves first in games
`base_seed`, `base_seed + 2`, ...
"""

import argparse
import sys
from collections.abc import Callable

from minimax_games.games.star_realms.bots import Bot, GreedyBot, RandomBot
from minimax_games.games.star_realms.match import play_match
from minimax_games.games.star_realms.tournament import (
    MAX_TURNS,
    Tournament,
    play_tournament,
)

BOTS: dict[str, Callable[[int], Bot]] = {
    "random": RandomBot,
    "greedy": lambda seed: GreedyBot(),
}
"""Each bot by name, built from a seed (which only `random` uses)."""


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["tournament"]:
        tournament(argv[1:])
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
        help=f"the two bots, first player first: {', '.join(BOTS)}",
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
        help=f"bots A and B (A moves first in even games): {', '.join(BOTS)}",
    )
    parser.add_argument("--games", type=int, default=500, help="how many games")
    parser.add_argument("--base-seed", type=int, default=0, help="the first seed")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=MAX_TURNS,
        help="games past this many turns are unfinished",
    )
    args = parser.parse_args(argv)

    a, b = args.bots
    played = play_tournament(
        (BOTS[a], BOTS[b]), args.games, args.base_seed, args.max_turns
    )
    last = args.base_seed + args.games - 1
    print(
        f"{a} vs {b}: {args.games} games, seeds {args.base_seed}-{last},"
        f" max {args.max_turns} turns"
    )
    for line in report(played, (a, b) if a != b else (f"{a}-A", f"{b}-B")):
        print(line)


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
