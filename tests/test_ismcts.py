import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import pytest

from minimax_games.games.tic_tac_toe.node import Board, Move, Player
from minimax_games.search.ismcts import ismcts


class TicTacToe:
    """Tic-tac-toe as a perfect-information ISMCTS game: X is player 0, and
    a leaf is valued by one uniformly random playout to the end."""

    def actions(self, state: Board) -> Sequence[Move]:
        if state.is_terminal():
            return ()
        return [
            (r, c)
            for r, row in enumerate(state.board)
            for c, v in enumerate(row)
            if not v
        ]

    def apply(self, state: Board, action: Move) -> Board:
        return state.move(action)

    def to_move(self, state: Board) -> int:
        return 0 if state.player == Player.X else 1

    def is_terminal(self, state: Board) -> bool:
        return state.is_terminal()

    def value(self, state: Board, rng: random.Random) -> float:
        while not state.is_terminal():
            moves = self.actions(state)
            state = state.move(moves[rng.randrange(len(moves))])
        score = state.score()
        return 1.0 if score > 0 else 0.0 if score < 0 else 0.5


def test_it_plays_a_winning_move_in_tic_tac_toe():
    board = Board.from_string("xx.oo....")
    found = ismcts(TicTacToe(), lambda: board, 500, random.Random(0))
    assert found.move == (0, 2)


def test_it_blocks_a_threat_in_tic_tac_toe():
    board = Board.from_string("oo.x.....")  # X to move: O threatens the top row
    found = ismcts(TicTacToe(), lambda: board, 1000, random.Random(0))
    assert found.move == (0, 2)


def test_it_is_deterministic_for_a_seed_and_spends_every_iteration():
    board = Board.from_string("x...o....")
    results = [ismcts(TicTacToe(), lambda: board, 300, random.Random(4)) for _ in "ab"]
    assert results[0].move == results[1].move
    assert results[0].visits == results[1].visits
    assert sum(results[0].visits.values()) == 300 == results[0].iterations
    assert results[0].tree.visits == 300


def test_it_needs_an_iteration():
    board = Board.from_string("x...o....")
    with pytest.raises(ValueError, match="at least 1 iteration"):
        ismcts(TicTacToe(), lambda: board, 0, random.Random(0))


@dataclass(frozen=True)
class Coin:
    """Cowling et al.'s Fig. 1 game, one player's: a hidden fair coin `h`.
    `safe` scores 0.6 at once; `gamble` is followed by a guess at the coin,
    scoring 1 if right and 0 if wrong, so it is worth 0.5."""

    h: int
    moves: tuple[str, ...] = ()


class CoinGame:
    def actions(self, state: Coin) -> Sequence[str]:
        if self.is_terminal(state):
            return ()
        return ("safe", "gamble") if not state.moves else ("guess 0", "guess 1")

    def apply(self, state: Coin, action: str) -> Coin:
        return replace(state, moves=(*state.moves, action))

    def to_move(self, state: Coin) -> int:
        return 0

    def is_terminal(self, state: Coin) -> bool:
        return state.moves[:1] == ("safe",) or len(state.moves) == 2

    def value(self, state: Coin, rng: random.Random) -> float:
        while not self.is_terminal(state):  # a uniformly random playout
            moves = self.actions(state)
            state = self.apply(state, moves[rng.randrange(len(moves))])
        if state.moves == ("safe",):
            return 0.6
        return 1.0 if state.moves[1] == f"guess {state.h}" else 0.0


def coins(rng: random.Random) -> Callable[[], Coin]:
    return lambda: Coin(rng.randrange(2))


def pimc(game: CoinGame, sample: Callable[[], Coin], worlds: int) -> str:
    """Perfect-information Monte Carlo: solve each sampled world exactly and
    play the root move most worlds chose."""

    def solve(state: Coin) -> float:
        if game.is_terminal(state):
            return game.value(state, random.Random())
        return max(solve(game.apply(state, a)) for a in game.actions(state))

    votes: Counter[str] = Counter()
    for _ in range(worlds):
        world = sample()
        votes[max(game.actions(world), key=lambda a: solve(game.apply(world, a)))] += 1
    return votes.most_common(1)[0][0]


def test_it_plays_safe_where_clairvoyant_sampling_gambles_on_a_hidden_coin():
    game = CoinGame()
    assert pimc(game, coins(random.Random(0)), 20) == "gamble"
    for seed in range(3):
        found = ismcts(game, coins(random.Random(seed)), 1000, random.Random(seed))
        assert found.move == "safe"


@dataclass(frozen=True)
class Hand:
    """Player 0 can only `go`; then player 1 plays one of the cards in its
    hidden hand, and the game ends: B is worth more to it than A."""

    hand: tuple[str, ...]
    moves: tuple[str, ...] = ()


class HandGame:
    def actions(self, state: Hand) -> Sequence[str]:
        return (
            ("go",) if not state.moves else () if len(state.moves) == 2 else state.hand
        )

    def apply(self, state: Hand, action: str) -> Hand:
        return replace(state, moves=(*state.moves, action))

    def to_move(self, state: Hand) -> int:
        return len(state.moves)

    def is_terminal(self, state: Hand) -> bool:
        return len(state.moves) == 2

    def value(self, state: Hand, rng: random.Random) -> float:
        if not self.is_terminal(state):
            return 0.5
        return 0.6 if state.moves[1] == "A" else 0.0


def test_an_action_is_available_in_every_visit_whose_world_allows_it():
    rng = random.Random(1)
    # Each card is first seen alone (from the 2nd and 3rd iterations), so it
    # is a child from the first time it is legal: a card legal beside an
    # untried other may not be the one expanded, and only children count.
    first: list[tuple[str, ...]] = [("A",), ("A",), ("B",)]
    held: list[tuple[str, ...]] = []

    def sample() -> Hand:
        if len(held) < len(first):
            hand = first[len(held)]
        else:
            hand = rng.choices([("A",), ("B",), ("A", "B")], weights=[6, 2, 2])[0]
        held.append(hand)
        return Hand(hand)

    found = ismcts(HandGame(), sample, 200, random.Random(0))
    opponent = found.tree.children["go"].children
    # The first iteration only expands `go`; every later one reaches the
    # opponent's decision, where its hand says which cards are legal.
    for card in "AB":
        assert opponent[card].available == sum(card in hand for hand in held[1:])
        assert opponent[card].visits <= opponent[card].available
    assert opponent["A"].visits + opponent["B"].visits == 199
    # Where it holds both, it learns to play B.
    assert opponent["B"].visits > sum(hand == ("B",) for hand in held)


def test_with_fewer_iterations_than_moves_it_picks_a_tried_one():
    board = Board.from_string("x...o....")
    found = ismcts(TicTacToe(), lambda: board, 3, random.Random(0))
    assert len(found.visits) == 3
    assert found.move in found.visits
