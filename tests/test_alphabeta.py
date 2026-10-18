import random
from collections import Counter
from collections.abc import Callable, Hashable, Iterator
from dataclasses import dataclass, field, replace
from typing import ClassVar, Self

import pytest

from minimax_games.games.tic_tac_toe.node import Board, Move, Player
from minimax_games.search.alphabeta import Table, alphabeta


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


def board_key(node: Board) -> Hashable:
    """A tic-tac-toe position: the board and the side to move."""
    return tuple(map(tuple, node.board)), node.player


@pytest.mark.parametrize("key", [None, board_key])
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
    position: str,
    player: Player,
    prune_on_ties: bool,
    depth: int | None,
    key: Callable[[Board], Hashable] | None,
) -> None:
    """With no limit, or one past the end of the game, the search is exact,
    with or without a transposition table."""
    b = Board.from_string(position, player)

    result = alphabeta(
        b,
        b.minimum(),
        b.maximum(),
        prune_on_ties=prune_on_ties,
        depth=depth,
        key=key,
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


class CountingTree(Tree):
    """A tree that records how often each node's children are generated."""

    expanded: ClassVar[Counter[str]] = Counter()

    def children(self) -> Iterator[Self]:
        CountingTree.expanded[self.name] += 1
        yield from super().children()


def name(node: Tree) -> str:
    return node.name


def diamond() -> CountingTree:
    """One player's turn of two plays, `p` and `q`, in either order: both
    orders reach the same node `pq`, whose replies `r` and `s` end the line."""
    pq = CountingTree("pq", 0, kids=(CountingTree("r", 3), CountingTree("s", 7)))
    p = CountingTree("p", 0, kids=(replace(pq, name="pq"),))
    q = CountingTree("q", 0, kids=(replace(pq, name="pq"),))
    return CountingTree("root", 0, kids=(p, q))


def test_a_transposition_is_searched_once_with_a_key() -> None:
    CountingTree.expanded.clear()
    root = diamond()

    alphabeta(root, root.minimum(), root.maximum(), key=name)

    assert CountingTree.expanded["pq"] == 1


def test_without_a_key_a_transposition_is_searched_again() -> None:
    CountingTree.expanded.clear()
    root = diamond()

    alphabeta(root, root.minimum(), root.maximum(), prune_on_ties=False)

    assert CountingTree.expanded["pq"] == 2


def test_a_transposed_result_keeps_the_path_that_reached_it() -> None:
    """`pq` is first searched under `p`, which the opponent spoils, then reached
    again under `q`: the best line runs through `q`, not `p`."""
    CountingTree.expanded.clear()
    pq = CountingTree("pq", 0, kids=(CountingTree("r", 3), CountingTree("s", 7)))
    p = CountingTree(
        "p", 0, maximising=False, kids=(replace(pq), CountingTree("t", -5))
    )
    q = CountingTree("q", 0, kids=(replace(pq),))
    root = CountingTree("root", 0, kids=(p, q))

    result = alphabeta(root, root.minimum(), root.maximum(), key=name)

    assert CountingTree.expanded["pq"] == 1
    assert result.moves == ["q", "pq", "s"]
    assert result.moves == alphabeta(root, root.minimum(), root.maximum()).moves


def dag(seed: int, layers: int = 6, width: int = 4, fan: int = 3) -> Tree:
    """A random game tree full of transpositions: each layer's nodes pick
    their children from the same few nodes of the next layer, so a node is
    reached by many paths, with different windows and, since some moves are
    free (`cost` 0), different depth budgets. The sides alternate by layer.
    """
    rng = random.Random(seed)
    below: list[Tree] = []
    for layer in reversed(range(1, layers + 1)):
        below = [
            Tree(
                f"{layer}.{i}",
                rng.randint(-20, 20),
                kids=tuple(rng.sample(below, rng.randint(1, fan))) if below else (),
                maximising=layer % 2 == 0,
                cost=rng.randint(0, 1),
            )
            for i in range(width)
        ]
    return Tree("root", 0, kids=tuple(below))


def depth_limited_minimax(node: Tree, depth: int | None) -> int:
    """Brute-force minimax under the same depth budget as `alphabeta`'s."""
    if node.is_terminal() or (depth is not None and depth <= 0):
        return node.score()
    values = [
        depth_limited_minimax(kid, None if depth is None else depth - kid.cost)
        for kid in node.children()
    ]
    return max(values) if node.is_maximising() else min(values)


@pytest.mark.parametrize("depth", [None, 1, 2, 3])
@pytest.mark.parametrize("prune_on_ties", [True, False])
@pytest.mark.parametrize("seed", range(100))
def test_a_keyed_search_agrees_with_brute_force_on_transposing_trees(
    seed: int, prune_on_ties: bool, depth: int | None
) -> None:
    root = dag(seed)

    result = alphabeta(
        root,
        root.minimum(),
        root.maximum(),
        prune_on_ties=prune_on_ties,
        depth=depth,
        ply_cost=cost,
        key=name,
    )

    assert result.score() == depth_limited_minimax(root, depth)
    first = next(kid for kid in root.children() if kid.name == result.moves[0])
    assert depth_limited_minimax(
        first, None if depth is None else depth - first.cost
    ) == (result.score())
    leaf = root
    for move in result.moves:
        leaf = next(kid for kid in leaf.children() if kid.name == move)
    assert leaf.score() == result.score()


def test_a_table_can_be_shared_between_searches() -> None:
    CountingTree.expanded.clear()
    root = diamond()
    table: Table[CountingTree] = {}
    first = alphabeta(root, root.minimum(), root.maximum(), key=name, table=table)

    again = alphabeta(root, root.minimum(), root.maximum(), key=name, table=table)

    assert again.moves == first.moves
    assert CountingTree.expanded["root"] == 1
