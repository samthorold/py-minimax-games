"""Watch one seeded game between two random bots.

`python -m minimax_games.games.star_realms --seed 7 --log` prints the game's
transcript; without `--log` it prints only who won.
"""

import argparse

from minimax_games.games.star_realms.bots import RandomBot
from minimax_games.games.star_realms.match import play_match


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m minimax_games.games.star_realms")
    parser.add_argument("--seed", type=int, default=0, help="the game's seed")
    parser.add_argument("--log", action="store_true", help="print the transcript")
    args = parser.parse_args(argv)

    bots = (RandomBot(args.seed), RandomBot(args.seed + 1))
    result = play_match(bots, args.seed, log=print if args.log else None)
    if not args.log:
        print(f"Player {result.winner + 1} wins on turn {result.turns}")
