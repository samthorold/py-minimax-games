from __future__ import annotations

import functools
import logging
from collections import Counter
from collections.abc import Iterator
from typing import Self

logger = logging.getLogger(__name__)


CORRECT_GUESS = "====="
COMPLETELY_WRONG = "....."
MINIMUM_NODE = "_____"
MAXIMUM_NODE = "^^^^^"
# Base score for a win, less one per guess used. It is above any feedback
# score and below the sentinels' 100.
WIN_SCORE = 50


@functools.cache
def evaluate(aim: str, guess: str) -> str:
    """Score ``guess`` against the answer ``aim`` using Wordle's rules.

    Exact matches (``=``) are assigned first. Then, left to right, a letter
    is present (``-``) while the answer still has unclaimed copies of it,
    and missing (``.``) otherwise.

    >>> evaluate("abcde", "eeeee")
    '....='
    """
    unclaimed = Counter(a for a, g in zip(aim, guess, strict=True) if a != g)
    status = ""
    for aimc, guessc in zip(aim, guess, strict=True):
        if aimc == guessc:
            status += "="
        elif unclaimed[guessc]:
            unclaimed[guessc] -= 1
            status += "-"
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
    if [s for s, c_ in zip(status, guess, strict=False) if c_ == c and s != "."]:
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
    for guess, score in zip(reversed(guesses), reversed(scores), strict=False):
        if score == CORRECT_GUESS:
            return [guess]
        else:
            words = [w for w in words if w != guess]

        for i, (c, s) in enumerate(zip(guess, score, strict=False)):
            match s:
                case "=":
                    words = prune_correct(words, i, c)
                case "-":
                    words = prune_present(words, i, c)
                case ".":
                    words = prune_missing(words, i, c, score, guess)
                case _:
                    raise ValueError("Unkown evaluation.")
        if final_only:
            break
    return words


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
        if min_max_node:
            return score_evaluation(self.moves[-1])
        # A maximising node's last move is the feedback for the latest
        # guess, a minimising node's last move is a guess.
        if self.is_maximising():
            if self.moves[-1] == CORRECT_GUESS:
                return WIN_SCORE - self.guesses_used()
            return score_evaluation(self.moves[-1])
        # And this is the crux
        # What is the score for a guess before knowing the
        # minimising player's turn
        # Returning the same score for all would _run_
        # but maybe only with alphabeta(prune_on_ties=True)
        if self.depth > 4:
            return 0
        return len(set(self.moves[-1]))

    def guesses_used(self) -> int:
        return self.depth // 2

    def is_maximising(self) -> bool:
        return bool(self.depth % 2)

    def is_terminal(self) -> bool:
        no_more_guesses = self.depth == 13
        correct_guess = bool(self.moves) and self.moves[-1] == CORRECT_GUESS
        return no_more_guesses or correct_guess

    def children(self) -> Iterator[WordleNode]:
        if self.is_maximising():
            self.prune()
            for guess in self.vocabulary:
                yield WordleNode(
                    moves=[*self.moves, guess],
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
                    moves=[*self.moves, sc],
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
