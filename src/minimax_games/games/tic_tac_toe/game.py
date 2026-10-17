from minimax_games.games.tic_tac_toe.node import Board, Player
from minimax_games.search.alphabeta import alphabeta


def main() -> None:
    prune_on_ties = True
    board = Board.from_string("." * 9, Player.O)
    while True:
        # r, c = [int(m) for m in input("Move: ")]
        # board = board.move((r, c))
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
        r, c = [int(m) for m in input("Move: ")]
        board = board.move((r, c))
        # variation = search.alphabeta(
        #     node=board,
        #     a=board.minimum(),
        #     b=board.maximum(),
        #     prune_on_ties=prune_on_ties,
        # )
        # board = board.move(variation.moves[board.depth])
        print(board.string())
        if board.is_terminal():
            print(board.score())
            break
