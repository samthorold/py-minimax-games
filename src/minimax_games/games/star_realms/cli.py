"""Watch one seeded game between two bots.

`python -m minimax_games.games.star_realms --seed 7 --log` prints the game's
transcript; without `--log` it prints only who won. `--bots greedy random`
picks the bots, first player first (both random by default).
"""

import argparse
from collections.abc import Callable

from minimax_games.games.star_realms.bots import Bot, GreedyBot, RandomBot
from minimax_games.games.star_realms.match import play_match

BOTS: dict[str, Callable[[int], Bot]] = {
    "random": RandomBot,
    "greedy": lambda seed: GreedyBot(),
}
"""Each bot by name, built from a seed (which only `random` uses)."""


def main(argv: list[str] | None = None) -> None:
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
