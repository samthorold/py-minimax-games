from __future__ import annotations
import functools
import logging
from typing import Iterator, Protocol, Self


logger = logging.getLogger(__name__)


CORRECT_GUESS = "====="
COMPLETELY_WRONG = "....."
MINIMUM_NODE = "_____"
MAXIMUM_NODE = "^^^^^"


@functools.cache
def present(aim: str, guess: str, guessc: str, i: int) -> str:
    if i > 0:
        count_aimc = len([c for c in aim if c == guessc])
        count_stats = len([c for c in guess[:i] if c == guessc])
        if count_aimc <= count_stats:
            return "."
    return "-"


@functools.cache
def evaluate(aim: str, guess: str) -> str:
    status = ""
    for i, (aimc, guessc) in enumerate(zip(aim, guess)):
        if aimc == guessc:
            status += "="
        elif guessc in aim:
            status += present(aim=aim, guess=guess, guessc=guessc, i=i)
        else:
            status += "."
    return status


@functools.cache
def score_evaluation(sc: str) -> int:
    logger.debug("score=%s", sc)
    if sc == MINIMUM_NODE:
        return -100
    if sc == MAXIMUM_NODE:
        return 100
    scores = {".": 0, "-": 1, "=": 2}
    return sum(scores[s] for s in sc)


def prune_correct(words: list[str], i: int, c: str) -> list[str]:
    return [w for w in words if w[i] == c]


def prune_present(words: list[str], i: int, c: str) -> list[str]:
    return [w for w in words if c in w and w[i] != c]


def prune_missing(
    words: list[str], i: int, c: str, status: str, guess: str
) -> list[str]:
    # Character with missing may be present elsewhere in the word
    # return [w for w in words if w[i] != c]
    if [s for s, c_ in zip(status, guess) if c_ == c and s != "."]:
        return [w for w in words if c in w and w[i] != c]
    return [w for w in words if c not in w]


def prune(
    words: list[str],
    guesses: list[str],
    scores: list[str],
    final_only: bool = True,
) -> list[str]:
    if len(guesses) != len(scores):
        raise ValueError(
            "Pruning with different sizes guesses and scores not supported"
        )
    for guess, score in zip(reversed(guesses), reversed(scores)):
        if score == CORRECT_GUESS:
            return [guess]
        else:
            words = [w for w in words if w != guess]

        for i, (c, s) in enumerate(zip(guess, score)):
            match s:
                case "=":
                    words = prune_correct(words, i, c)
                case "-":
                    words = prune_present(words, i, c)
                case ".":
                    words = prune_missing(words, i, c, score, guess)
                case _:
                    raise ValueError(f"Unkown evaluation.")
        if final_only:
            break
    return words


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


class WordleNode:
    """Node in a Wordle game.

    The maximising player chooses the highest scoring allowed word.
    The minimising player chooses the lowest score that guess could have attained,
    given the allowed words.

    """

    moves: list[str]

    def __init__(
        self,
        moves: list[str],
        vocabulary: list[str],
        depth: int = 1,
    ) -> None:
        self.moves = moves
        self.vocabulary = vocabulary
        self.depth = depth
        if vocabulary:
            logger.debug("create node %s %s %s", moves, depth, self.is_terminal())

    def __lt__(self, other: Self) -> bool:
        return self.score() < other.score()

    def __le__(self, other: Self) -> bool:
        return self.score() <= other.score()

    def __gt__(self, other: Self) -> bool:
        return self.score() > other.score()

    def __ge__(self, other: Self) -> bool:
        return self.score() >= other.score()

    def score(self) -> int:
        min_max_node = self.moves[-1] in [MINIMUM_NODE, MAXIMUM_NODE]
        if not self.is_maximising() or min_max_node:
            return score_evaluation(self.moves[-1])
        # And this is the crux
        # What is the score for a guess before knowing the
        # minimising player's turn
        # Returning the same score for all would _run_
        # but maybe only with soft alphabeta
        if self.depth > 4:
            return 0
        return len(set(self.moves[-1]))

    def is_maximising(self) -> bool:
        return bool(self.depth % 2)

    def is_terminal(self) -> bool:
        no_more_guesses = self.depth == 13
        correct_guess = bool(self.moves) and self.moves[-1] == CORRECT_GUESS
        return no_more_guesses or correct_guess

    def children(self) -> Iterator[Self]:
        if self.is_maximising():
            self.prune()
            for guess in self.vocabulary:
                yield WordleNode(
                    moves=self.moves + [guess],
                    vocabulary=[w for w in self.vocabulary],
                    depth=self.depth + 1,
                )
        else:
            # this only needs to be each _evaluation_
            # multiple words lead to the same evaluation.
            for guess in self.vocabulary:
                sc = evaluate(guess=self.moves[-1], aim=guess)
                logger.debug("%s %s", self.moves, sc)
                yield WordleNode(
                    moves=self.moves + [sc],
                    vocabulary=[w for w in self.vocabulary],
                    depth=self.depth + 1,
                )

    def prune(self) -> None:
        if len(self.moves) < 2:
            return
        self.vocabulary = prune(
            words=self.vocabulary,
            guesses=self.moves[-2:-1],
            scores=self.moves[-1:],
            final_only=True,
        )

    def maximum(self) -> WordleNode:
        return WordleNode(vocabulary=[], moves=[MAXIMUM_NODE])

    def minimum(self) -> WordleNode:
        return WordleNode(vocabulary=[], moves=[MINIMUM_NODE])
