"""Star Realms as a search tree: the `Node` adapter `alphabeta` searches.

A node is a state seen from one player's side, `perspective`, who maximises.
Its children come from the shared move generator (`rules.legal_actions`,
then `rules.advance_legal`: `advance`, so automatic play runs as in a match,
without re-checking an action just taken from `legal_actions`), and `moves` is
the line of decisions from the root. Search depth counts turns
(docs/star-realms.md, Search): pass `ends_turn` as `alphabeta`'s `ply_cost`.
"""

from collections.abc import Hashable, Iterator
from typing import Self

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.action import Action, EndTurn
from minimax_games.games.star_realms.evaluate import LINEAR, WIN, Evaluator
from minimax_games.games.star_realms.state import GameState


class StarRealmsNode:
    """`state` seen from `perspective`, reached by `moves` from the root,
    scored by `evaluator` (its children's too): by default `evaluate`."""

    def __init__(
        self,
        state: GameState,
        perspective: int,
        moves: list[Action] | None = None,
        *,
        score: int | None = None,
        evaluator: Evaluator = LINEAR,
    ) -> None:
        self.state = state
        self.evaluator = evaluator
        self.perspective = perspective
        self.moves: list[Action] = [] if moves is None else moves
        self._score = score

    def __gt__(self, other: Self) -> bool:
        return self.score() > other.score()

    def __lt__(self, other: Self) -> bool:
        return self.score() < other.score()

    def __ge__(self, other: Self) -> bool:
        return self.score() >= other.score()

    def __le__(self, other: Self) -> bool:
        return self.score() <= other.score()

    def score(self) -> int:
        """The evaluator's score from `perspective`, computed once.

        A finished game scores `WIN` (or `-WIN`) less one per move it took,
        so the search prefers the quickest win and the slowest loss.
        """
        if self._score is None:
            score = self.evaluator(self.state, self.perspective)
            if abs(score) == WIN:
                score -= len(self.moves) if score > 0 else -len(self.moves)
            self._score = score
        return self._score

    def is_terminal(self) -> bool:
        return rules.winner(self.state) is not None

    def is_maximising(self) -> bool:
        return self.state.current == self.perspective

    def children(self) -> Iterator[Self]:
        for action in rules.legal_actions(self.state):
            yield type(self)(
                rules.advance_legal(self.state, action),
                self.perspective,
                [*self.moves, action],
                evaluator=self.evaluator,
            )

    def minimum(self) -> Self:
        """A sentinel scoring below every real node, even a lost game."""
        return type(self)(
            self.state, self.perspective, score=-WIN - 1, evaluator=self.evaluator
        )

    def maximum(self) -> Self:
        """A sentinel scoring above every real node, even a won game."""
        return type(self)(
            self.state, self.perspective, score=WIN + 1, evaluator=self.evaluator
        )


def ends_turn(node: StarRealmsNode) -> int:
    """`alphabeta`'s `ply_cost`: 1 for a node reached by ending a turn, else 0,
    so the search depth counts turns."""
    return 1 if node.moves and isinstance(node.moves[-1], EndTurn) else 0


def transposition_key(node: StarRealmsNode) -> Hashable:
    """`alphabeta`'s `key`: the state, which is hashable and includes the side
    to move, and the number of moves to it, since a finished game's score
    depends on how many moves it took (`StarRealmsNode.score`). Different
    orders of the same plays reach the same state in the same number of moves,
    so this still merges them."""
    return node.state, len(node.moves)
