from collections.abc import Iterator

import pytest

from minimax_games.games.tic_tac_toe.node import Board, Player
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
    position: str, player: Player, prune_on_ties: bool
) -> None:
    b = Board.from_string(position, player)

    result = alphabeta(b, b.minimum(), b.maximum(), prune_on_ties=prune_on_ties)

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
