from __future__ import annotations
import argparse
import logging
import random
from typing import Protocol

from minimax_games.games.wordle.node import (
    evaluate,
    prune,
    score_evaluation,
    WordleNode,
    COMPLETELY_WRONG,
    CORRECT_GUESS,
)
from minimax_games.search.alphabeta import alphabeta


logger = logging.getLogger(__name__)


class Guesser(Protocol):
    def __call__(self, guesses: list[str], scores: list[str]) -> str:
        ...


class Scorer(Protocol):
    def __call__(self, guess: str) -> str:
        ...


class AutoScorer:
    def __init__(self, truth: str) -> None:
        self.truth = truth

    def __call__(self, guess: str) -> str:
        return evaluate(self.truth, guess)


class UserScorer:
    def __call__(self, guess: str) -> str:
        print(f"Guess: {guess}")
        while True:
            score = input("Score: ").strip().lower()
            if len(score) != 5:
                print("Score must be 5 letters long. Enter another.")
                continue
            return score


class UserGuesser:
    def __init__(self, vocabulary: list[str]) -> None:
        self.vocabulary = vocabulary

    def __call__(self, guesses: list[str], scores: list[str]) -> str:
        while True:
            guess = input("Guess: ").strip().lower()
            if len(guess) != 5:
                print("Guess must be 5 letters long. Enter another.")
                continue
            if not guess in self.vocabulary:
                print("Guess not a known word.")
                continue
            return guess


class AlphaBetaGuesser:
    def __init__(self, vocabulary: list[str]) -> None:
        self.vocabulary = vocabulary

    def __call__(self, guesses: list[str], scores: list[str]) -> str:
        if not guesses:
            return "crate"
        if len(guesses) == 1 and scores[-1] == COMPLETELY_WRONG:
            return "bogus"
        vocabulary = prune(
            words=self.vocabulary,
            guesses=guesses,
            scores=scores,
            final_only=False,
        )
        node = WordleNode(
            moves=[guesses[-1], scores[-1]],
            vocabulary=vocabulary,
            depth=1 + len(guesses) * 2,
        )
        best_guess = alphabeta(
            node,
            a=node.minimum(),
            b=node.maximum(),
            soft=True,
        )
        logger.info(
            "best node move=%s moves=%s", best_guess.moves[-2], best_guess.moves
        )
        return best_guess.moves[-2]


class Wordle:
    def __init__(
        self,
        guesser: Guesser,
        scorer: Scorer,
        vocabulary: list[str],
    ) -> None:
        self.guesser = guesser
        self.scorer = scorer
        self.vocabulary = vocabulary
        self.guess_next = True
        self.guesses: list[str] = []
        self.scores: list[str] = []

    def __str__(self) -> str:
        string = "\n".join(
            f"{guess} {score}" for guess, score in zip(self.guesses, self.scores)
        )
        if len(self.guesses) > len(self.scores):
            string += f"\n{self.guesses[-1]}"
        return string

    def guess(self, guess: str) -> None:
        if len(self.guesses) > len(self.scores):
            raise RuntimeError("Score the last guess first.")
        self.guesses.append(guess)

    def score(self) -> str:
        if len(self.scores) == len(self.guesses):
            raise RuntimeError("Make another guess first.")
        score = self.scorer(self.guesses[-1])
        self.scores.append(score)
        return score

    def move(self) -> None:
        logger.info("move %s %s", self.guesses, self.scores)
        if self.guess_next:
            self.guess(self.guesser(guesses=self.guesses, scores=self.scores))
        else:
            self.score()
        self.guess_next = not self.guess_next

    def is_terminal(self) -> bool:
        correct_guess = any(s == CORRECT_GUESS for s in self.scores)
        no_more_guesses = len(self.scores) == 6
        return correct_guess or no_more_guesses


def main(
    truth: str,
    vocabulary: list[str],
    guesser: Guesser,
    scorer: Scorer,
) -> int:
    wordle = Wordle(
        guesser=guesser,
        scorer=scorer,
        vocabulary=vocabulary,
    )
    while True:
        wordle.move()
        if len(wordle.scores) == len(wordle.guesses):
            print(wordle)
        print("---")
        if wordle.is_terminal():
            break
    return score_evaluation(wordle.scores[-1])


class WordleArgs:
    def __init__(
        self,
        truth: str,
        vocabulary: list[str],
        guesser: Guesser,
        scorer: Scorer,
        log_level: str,
    ) -> None:
        if truth not in vocabulary:
            raise ValueError(f"Target '{truth}' not in vocabulary.")
        self.truth = truth
        self.vocabulary = vocabulary
        self.guesser = guesser
        self.scorer = scorer
        self.log_level = log_level.upper()

    @classmethod
    def from_argument_parser(cls, cli: argparse.ArgumentParser) -> WordleArgs:
        args = cli.parse_args()
        vocab_path = "words/words.txt" if args.vocabulary is None else args.vocabulary
        with open(vocab_path) as f:
            vocabulary = [line.strip().lower() for line in f]
        truth = random.choice(vocabulary) if args.truth is None else args.truth
        guesser = (
            UserGuesser(vocabulary=vocabulary)
            if args.interactive_guess
            else AlphaBetaGuesser(vocabulary)
        )
        scorer = UserScorer() if args.interactive_score else AutoScorer(truth=truth)
        return WordleArgs(
            truth=truth,
            vocabulary=vocabulary,
            guesser=guesser,
            scorer=scorer,
            log_level=args.log_level,
        )


cli = argparse.ArgumentParser()
cli.add_argument("--truth")
cli.add_argument("--vocabulary")
cli.add_argument("--log-level", default="WARNING")
cli.add_argument("--interactive-guess", action="store_true")
cli.add_argument("--interactive-score", action="store_true")
