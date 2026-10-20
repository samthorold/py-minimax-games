"""The cheat: alpha-beta search on the true state.

A benchmark and an upper bound, not a legitimate player: it sees the whole
`GameState` (both hands, every deck's order, the shuffle seed), which no other
bot is given (docs/star-realms.md, Search).
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Self

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.action import Action
from minimax_games.games.star_realms.evaluate import LINEAR, Evaluator
from minimax_games.games.star_realms.node import (
    StarRealmsNode,
    ends_turn,
    transposition_key,
)
from minimax_games.games.star_realms.state import GameState
from minimax_games.search.alphabeta import Table, alphabeta

logger = logging.getLogger(__name__)


def search[N: StarRealmsNode](root: N, turns: int) -> N:
    """The cheat's search: the best leaf `turns` turns below `root`.

    A state reached by another order of the same plays is searched once, from
    a transposition table keyed by `transposition_key`.
    """
    return alphabeta(
        root,
        root.minimum(),
        root.maximum(),
        depth=turns,
        ply_cost=ends_turn,
        key=transposition_key,
    )


class _OverBudget(Exception):
    """Raised by a `_BudgetedNode` asked for a node past the budget."""


class _Budget:
    """The nodes a decision may still generate."""

    def __init__(self, nodes: int) -> None:
        self.limit = nodes
        self.used = 0

    def spend(self) -> None:
        if self.used >= self.limit:
            raise _OverBudget
        self.used += 1


class _BudgetedNode(StarRealmsNode):
    """A node that charges each child it generates to `budget`, raising
    `_OverBudget` once it is spent, so `alphabeta` itself is unchanged."""

    budget: _Budget

    def children(self) -> Iterator[Self]:
        for child in super().children():
            self.budget.spend()
            child.budget = self.budget
            yield child


@dataclass(frozen=True, slots=True)
class Decision:
    """A budgeted decision: deterministic for the same state and budget."""

    move: Action
    nodes: int
    """Nodes generated, the root included, over every iteration: at most
    the budget."""
    depth: int
    """The deepest search, in turns, that finished within the budget: 0 if
    not even the 1-turn search did."""


def decide(
    state: GameState, turns: int, budget: int, evaluator: Evaluator = LINEAR
) -> Decision:
    """The cheat's move in `state`, searching at most `budget` nodes and
    scoring positions with `evaluator` (by default the linear `evaluate`).

    Iterative deepening over turns: search 1 turn, then 2, up to `turns`,
    and play the best move of the deepest search that finished. If not even
    the 1-turn search finished, play the best of the root moves it searched,
    or failing that the first legal action.
    """
    root = _BudgetedNode(state, state.current, evaluator=evaluator)
    root.budget = _Budget(budget)
    root.budget.spend()
    move, depth = None, 0
    for horizon in range(1, turns + 1):
        best, finished = _search_root(root, horizon)
        if finished:
            move, depth = best, horizon
        else:
            if depth == 0:
                move = best
            break
    if move is None:
        move = rules.legal_actions(state)[0]
    return Decision(move, root.budget.used, depth)


def _search_root(root: StarRealmsNode, turns: int) -> tuple[Action | None, bool]:
    """`search` from `root`, one root move at a time, so a search stopped by
    the budget still has the best of the moves it finished. Returns that
    move (None if none finished) and whether the whole search did.

    The same as `alphabeta` at the root (which is maximising, since the
    cheat searches from the mover's side): each move is searched with the
    window raised by the best so far, and ties keep the earlier move.

    Each search has its own transposition table. `alphabeta` writes an entry
    only once a node's search is complete, so a stopped search leaves only
    sound entries, and nothing reads them afterwards anyway. Sharing the
    table with the next iteration would gain nothing: it is keyed by the
    remaining depth and the number of moves, and the deeper search's nodes at
    a given remaining depth are a turn later than the shallower one's.
    """
    best: StarRealmsNode | None = None
    a, b = root.minimum(), root.maximum()
    table: Table[StarRealmsNode] = {}
    finished = True
    try:
        for child in root.children():
            value = alphabeta(
                child,
                a,
                b,
                depth=turns - ends_turn(child),
                ply_cost=ends_turn,
                key=transposition_key,
                table=table,
            )
            if best is None or value > best:
                best = value
                a = max(a, best)
    except _OverBudget:
        finished = False
    return (None if best is None else best.moves[0]), finished


BUDGET = 5_000
"""The default node budget per decision (see `decide`), shared by the cheat
and the sampler: about 70 ms per cheat decision in a game, and a 500-game
search-bot tournament in about half an hour with `--jobs 3`
(docs/star-realms-results.md, Milestone 3)."""


class CheatBot:
    """Benchmark only, not a legitimate player: searches the true state.

    It runs `alphabeta` on the whole state, `turns` turns deep (by default
    2: the rest of its own turn plus the opponent's reply), scores the
    positions it reaches with `evaluator` (by default the linear
    `evaluate`; `rollout.Rollout` plays them out), and plays the
    first move of the best line. Each decision searches at most `budget`
    nodes, deepening one turn at a time (`decide`). The match runner gives it
    the true state because it is a `bots.BenchmarkBot`.
    """

    def __init__(
        self, turns: int = 2, budget: int = BUDGET, evaluator: Evaluator = LINEAR
    ) -> None:
        if turns < 1:
            msg = f"the horizon must be at least 1 turn, not {turns}"
            raise ValueError(msg)
        if budget < 1:
            msg = f"the budget must be at least 1 node, not {budget}"
            raise ValueError(msg)
        self.turns = turns
        self.budget = budget
        self.evaluator = evaluator

    def settings(self) -> str:
        return (
            f"{self.turns}-turn horizon, budget {self.budget} nodes per decision,"
            f" {self.evaluator.describe()}"
        )

    def choose_from_state(
        self, state: GameState, actions: tuple[Action, ...]
    ) -> Action:
        decision = decide(state, self.turns, self.budget, self.evaluator)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("decided %s", decision)
        assert decision.move in actions, (decision.move, actions)
        return decision.move
