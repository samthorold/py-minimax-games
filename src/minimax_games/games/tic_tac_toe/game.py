import re

from minimax_games.games.tic_tac_toe.node import Board, IllegalMove, Move, Player
from minimax_games.search.alphabeta import alphabeta

MOVE_PATTERN = re.compile(r"\s*([0-9])\s*,?\s*([0-9])\s*")


def parse_move(text: str) -> Move:
    """Read a move as a row and a column, written `rc`, `r c` or `r,c`.

    >>> parse_move("12")
    (1, 2)
    >>> parse_move("2, 0")
    (2, 0)

    Raises `ValueError`, with a message fit to show the player, if the
    text isn't two digits or names a cell off the board.
    """
    match = MOVE_PATTERN.fullmatch(text)
    if match is None:
        msg = "Enter a row and a column, e.g. 12, 1 2 or 1,2"
        raise ValueError(msg)
    r, c = int(match[1]), int(match[2])
    if r > 2 or c > 2:
        msg = "Row and column must each be 0, 1 or 2"
        raise ValueError(msg)
    return r, c


def read_move(board: Board) -> Board:
    """Ask the player for a move and return the board after it.

    Malformed moves and occupied cells are reported and asked for again.
    """
    while True:
        try:
            return board.move(parse_move(input("Move: ")))
        except (ValueError, IllegalMove) as e:
            print(e)


def main() -> None:
    prune_on_ties = True
    board = Board.from_string("." * 9, Player.O)
    while True:
        variation = alphabeta(
            node=board,
            a=board.minimum(),
            b=board.maximum(),
            prune_on_ties=prune_on_ties,
        )
        board = board.move(variation.moves[board.depth])
        print(board.string())
        if board.is_terminal():
            print(board.score())
            break
        board = read_move(board)
        print(board.string())
        if board.is_terminal():
            print(board.score())
            break
