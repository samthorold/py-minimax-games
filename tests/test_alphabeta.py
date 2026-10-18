from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import ClassVar, Self

import pytest

from minimax_games.games.tic_tac_toe.node import Board, Move, Player
from minimax_games.search.alphabeta import alphabeta


class ChildlessBoard(Board):
    """A board that is not terminal but has no moves to make."""

    def is_terminal(self) -> bool:
        return False

    def children(self) -> Iterator[Board]:
        yield from ()


def test_childless_node_is_treated_as_a_leaf() -> None:
    b = ChildlessBoard.from_string(None)

    result = alphabeta(b, b.minimum(), b.maximum())

    assert result is b


class ShallowBoard(Board):
    """A board whose children are childless, so the tree is one move deep."""

    def children(self) -> Iterator[Board]:
        for child in super().children():
            yield ChildlessBoard(child.board, child.player, child.moves, child.depth)


def test_childless_children_are_scored_as_leaves() -> None:
    b = ShallowBoard.from_string("xx.oo....")

    result = alphabeta(b, b.minimum(), b.maximum())

    assert result.moves == [(0, 2)]
    assert result.score() == 9


def minimax(node: Board) -> int:
    """Brute-force minimax value, for checking alphabeta against."""
    if node.is_terminal():
        return node.score()
    values = [minimax(child) for child in node.children()]
    return max(values) if node.is_maximising() else min(values)


@pytest.mark.parametrize("depth", [None, 9])
@pytest.mark.parametrize("prune_on_ties", [True, False])
@pytest.mark.parametrize(
    ("position", "player"),
    [
        ("x...o...x", Player.O),
        ("xo..x....", Player.O),
        ("x.o.x...o", Player.X),
        ("xo.x..o..", Player.X),
        ("o.x.x....", Player.O),
        ("xoxo.....", Player.X),
        (".x..o.x..", Player.O),
    ],
)
def test_tic_tac_toe_agrees_with_brute_force_minimax(
    position: str, player: Player, prune_on_ties: bool, depth: int | None
) -> None:
    """With no limit, or one past the end of the game, the search is exact."""
    b = Board.from_string(position, player)

    result = alphabeta(
        b, b.minimum(), b.maximum(), prune_on_ties=prune_on_ties, depth=depth
    )

    assert result.score() == minimax(b)
    assert minimax(b.move(result.moves[0])) == minimax(b)


@pytest.mark.parametrize("prune_on_ties", [True, False])
@pytest.mark.parametrize(
    ("position", "player", "score", "moves"),
    [
        (
            "x...o...x",
            Player.O,
            0,
            [(0, 1), (2, 1), (2, 0), (0, 2), (1, 2), (1, 0)],
        ),
        ("xo..x....", Player.O, 6, [(2, 2), (1, 0), (0, 2), (1, 2)]),
        ("x.o.x...o", Player.X, 0, [(1, 2), (1, 0), (0, 1), (2, 1), (2, 0)]),
        ("xo.x..o..", Player.X, 7, [(1, 1), (0, 2), (1, 2)]),
        (
            "o.x.x....",
            Player.O,
            0,
            [(2, 0), (1, 0), (1, 2), (0, 1), (2, 1), (2, 2)],
        ),
        ("xoxo.....", Player.X, 7, [(1, 1), (1, 2), (2, 0)]),
        (
            ".x..o.x..",
            Player.O,
            0,
            [(0, 0), (2, 2), (2, 1), (0, 2), (1, 2), (1, 0)],
        ),
    ],
)
def test_tic_tac_toe_best_line_is_pinned(
    position: str,
    player: Player,
    score: int,
    moves: list[tuple[int, int]],
    prune_on_ties: bool,
) -> None:
    b = Board.from_string(position, player)

    result = alphabeta(b, b.minimum(), b.maximum(), prune_on_ties=prune_on_ties)

    assert result.score() == score
    assert result.moves == moves


def test_result_keeps_the_concrete_node_type() -> None:
    """Board-only attributes on the result type-check (ty runs over tests)."""
    b = Board.from_string("xo.x..o..", Player.X)

    result = alphabeta(b, b.minimum(), b.maximum())

    assert result.depth == len(result.moves)
    assert result.board[1][1] == Player.X


class UnscorableBoard(Board):
    """A bound that fails if anything asks for its score."""

    def score(self) -> int:
        raise AssertionError("bound was scored")


def test_terminal_node_does_not_score_the_bounds() -> None:
    """Returning a leaf needs no comparison, so the bounds go unscored."""
    b = Board.from_string("xxxoo....", Player.X)
    a = UnscorableBoard.from_string("ooo......")
    z = UnscorableBoard.from_string("xxx......")

    assert alphabeta(b, a, z) is b


class CountingBoard(Board):
    """A board that records how often each position's lines are scanned."""

    scans: ClassVar[Counter[tuple[Move, ...]]] = Counter()

    def _scan_lines(self) -> int:
        CountingBoard.scans[tuple(self.moves)] += 1
        return super()._scan_lines()

    def children(self) -> Iterator[Board]:
        for child in super().children():
            yield CountingBoard(child.board, child.player, child.moves, child.depth)


def test_each_board_is_scored_at_most_once() -> None:
    CountingBoard.scans.clear()
    b = CountingBoard.from_string("xo.x..o..", Player.X)

    result = alphabeta(b, b.minimum(), b.maximum())

    assert result.score() == 7
    assert CountingBoard.scans
    assert max(CountingBoard.scans.values()) == 1


@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_depth_limit_stops_the_search_that_many_plies_down(depth: int) -> None:
    b = Board.from_string("x...o....", Player.X, depth=2)

    result = alphabeta(b, b.minimum(), b.maximum(), depth=depth)

    assert result.depth <= b.depth + depth


def test_depth_limit_scores_the_frontier_with_score() -> None:
    """At the limit, `score()` is the evaluation: one ply picks the best child."""
    b = Board.from_string("x...o....", Player.X)

    result = alphabeta(b, b.minimum(), b.maximum(), depth=1)

    assert len(result.moves) == 1
    assert not result.is_terminal()
    assert result.score() == max(child.score() for child in b.children())


def test_depth_zero_returns_the_node_itself() -> None:
    b = Board.from_string("x...o....", Player.X)

    assert alphabeta(b, b.minimum(), b.maximum(), depth=0) is b


@dataclass
class Tree:
    """A hand-built game tree node, with a `cost` for `ply_cost` to read.

    `value` is the node's score: the true value of a leaf, or the heuristic
    evaluation of an inner node when the search stops there.
    """

    name: str
    value: int
    kids: tuple[Self, ...] = ()
    maximising: bool = True
    cost: int = 1
    moves: list[str] = field(default_factory=list)

    def __gt__(self, other: Self) -> bool:
        return self.value > other.value

    def __lt__(self, other: Self) -> bool:
        return self.value < other.value

    def __ge__(self, other: Self) -> bool:
        return self.value >= other.value

    def __le__(self, other: Self) -> bool:
        return self.value <= other.value

    def score(self) -> int:
        return self.value

    def is_terminal(self) -> bool:
        return not self.kids

    def children(self) -> Iterator[Self]:
        for kid in self.kids:
            yield replace(kid, moves=[*self.moves, kid.name])

    def is_maximising(self) -> bool:
        return self.maximising

    def minimum(self) -> Self:
        return replace(self, name="min", value=-1000, kids=())

    def maximum(self) -> Self:
        return replace(self, name="max", value=1000, kids=())


def cost(node: Tree) -> int:
    return node.cost


def test_free_plies_do_not_use_up_the_depth_budget() -> None:
    """`a` costs nothing, so a one-ply search looks through it to its replies."""
    root = Tree(
        "root",
        0,
        kids=(
            Tree(
                "a",
                100,
                maximising=False,
                cost=0,
                kids=(Tree("a1", 3), Tree("a2", 5)),
            ),
            Tree("b", 4, maximising=False, kids=(Tree("b1", -9),)),
        ),
    )

    result = alphabeta(root, root.minimum(), root.maximum(), depth=1, ply_cost=cost)

    assert result.moves == ["b"]
    assert result.score() == 4


def chain(*costs: int) -> Tree:
    """A line of single moves, `n0` to `nk`, where `nk` costs `costs[k - 1]`.

    Each node's heuristic value is its index, and the last one is terminal.
    """
    node = Tree(f"n{len(costs)}", len(costs), cost=costs[-1])
    for i in reversed(range(1, len(costs))):
        node = Tree(f"n{i}", i, kids=(node,), cost=costs[i - 1])
    return Tree("n0", 0, kids=(node,))


@pytest.mark.parametrize(
    ("costs", "depth", "stop"),
    [
        ((0, 0, 1, 0, 1, 1), 1, "n3"),
        ((0, 0, 1, 0, 1, 1), 2, "n5"),
        ((1, 0, 0, 1, 1), 1, "n1"),
        ((2, 1, 1), 1, "n1"),
        ((0, 0, 0), 1, "n3"),
    ],
)
def test_search_stops_at_the_node_that_uses_up_the_budget(
    costs: tuple[int, ...], depth: int, stop: str
) -> None:
    root = chain(*costs)

    result = alphabeta(root, root.minimum(), root.maximum(), depth=depth, ply_cost=cost)

    assert result.name == stop


@pytest.mark.parametrize("costs", [(1, 1, 1), (0, 0, 0), (1, 0, 1), (5, 5, 5)])
def test_no_depth_limit_searches_to_terminal_nodes_whatever_the_cost(
    costs: tuple[int, ...],
) -> None:
    root = chain(*costs)

    result = alphabeta(root, root.minimum(), root.maximum(), ply_cost=cost)

    assert result.is_terminal()
    assert result.name == "n3"
