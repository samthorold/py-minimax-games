import logging

from minimax_games.games.wordle.game import cli, main, WordleArgs


if __name__ == "__main__":
    print("=== PyWordle ===")
    args = WordleArgs.from_argument_parser(cli)
    logging.basicConfig(level=args.log_level)
    main(
        truth=args.truth,
        vocabulary=args.vocabulary,
        guesser=args.guesser,
        scorer=args.scorer,
    )
    print(args.truth)
