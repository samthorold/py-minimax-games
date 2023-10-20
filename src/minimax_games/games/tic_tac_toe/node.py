from __future__ import annotations
from enum import Enum
from typing import Iterator


class IllegalMove(Exception):
    """Cell contains a move already."""


class Player(Enum):
    X = "x"
    O = "o"


Move = tuple[int, int]


class Board:
    @classmethod
    def from_string(
        cls,
        string: str | None,
        player: Player = Player.X,
        depth: int = 0,
    ) -> Board:
        state = "." * 9 if string is None else string
        state = state.lower()
        b: list[list[Player | None]] = []
        for r in range(3):
            b.append([])
            for c in range(3):
                ch = state[r * 3 + c]
                b[-1].append(Player(ch) if ch != "." else None)
        return cls(b, player=player, depth=depth)

    def __init__(
        self,
        state: list[list[Player | None]],
        player: Player,
        moves: list[Move] | None = None,
        depth: int = 0,
    ):
        self.board = state
        self.player = player
        self.moves: list[Move] = [] if moves is None else moves
        self.depth = depth

    def __gt__(self, other: Board) -> bool:
        return self.score() > other.score()

    def __lt__(self, other: Board) -> bool:
        return self.score() < other.score()

    def __ge__(self, other: Board) -> bool:
        return self.score() >= other.score()

    def __le__(self, other: Board) -> bool:
        return self.score() <= other.score()

    def is_maximising(self) -> bool:
        return self.player == Player.X

    def minimum(self) -> Board:
        return Board.from_string("ooo......")

    def maximum(self) -> Board:
        return Board.from_string("xxx......")

    def string(self) -> str:
        s = f"{self.player} to play ...\n"
        for row in self.board:
            for val in row:
                s += "." if val is None else val.value
            s += "\n"
        s += f"{self.score()}\n"
        return s

    def move(self, move: Move) -> Board:
        rix, cix = move
        if self.board[rix][cix]:
            msg = f"row={rix}, col={cix} is occupied ({self.board[rix][cix]})"
            raise IllegalMove(msg)
        next_board = [[v for v in row] for row in self.board]
        next_board[rix][cix] = self.player
        return Board(
            state=next_board,
            player=self.next_player(),
            moves=self.moves + [move],
            depth=self.depth + 1,
        )

    def score(self) -> int:
        for p in Player:
            sign = 1 if p == Player.X else -1
            if any(all(c == p for c in row) for row in self.board):
                return sign * (10 - self.depth)
            if any(
                all(c == p for c in [self.board[i][j] for i in range(3)])
                for j in range(3)
            ):
                return sign * (10 - self.depth)
            if all(
                [self.board[0][0] == p, self.board[1][1] == p, self.board[2][2] == p]
            ):
                return sign * (10 - self.depth)
            if all(
                [self.board[2][0] == p, self.board[1][1] == p, self.board[0][2] == p]
            ):
                return sign * (10 - self.depth)
        return 0

    def is_terminal(self) -> bool:
        return bool(self.score()) or all(all(row) for row in self.board)

    def next_player(self) -> Player:
        return Player.X if self.player == Player.O else Player.O

    def children(self) -> Iterator[Board]:
        if self.is_terminal():
            return
        for rix, row in enumerate(self.board):
            for cix, val in enumerate(row):
                if val:
                    continue
                yield self.move((rix, cix))


