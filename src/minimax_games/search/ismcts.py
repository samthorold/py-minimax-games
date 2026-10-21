"""Single-observer information-set Monte Carlo tree search (SO-ISMCTS).

Cowling, Powley and Whitehouse (2012), with availability counts: see
docs/ismcts.md. A two-player, zero-sum search for games whose moves are
fully observable, so a node is identified by the actions from the root (that
path is the searcher's information set), and no states are stored in the
tree. Every iteration samples a fresh world consistent with the root
information set and replays its path on it. Chance is not branched on ("open
loop"): it is resolved inside `apply` from the sampled world.
"""

import math
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from random import Random
from typing import Protocol

EXPLORATION = 0.7
"""The default exploration constant, for rewards in [0, 1]
(Cowling et al. 2012)."""


class Game[S, A: Hashable](Protocol):
    """A two-player, zero-sum game with fully observable moves, seen through
    determinized worlds `S`. Players are 0 and 1.

    Actions must be hashable, and equal across worlds when they mean the
    same move."""

    def actions(self, state: S) -> Sequence[A]:
        """The legal actions, in a fixed order (ties are broken by it)."""
        ...

    def apply(self, state: S, action: A) -> S:
        """The state after `action`, including any automatic play and chance,
        which the world resolves from its own seeded randomness."""
        ...

    def to_move(self, state: S) -> int:
        """The player choosing in `state`: 0 or 1."""
        ...

    def is_terminal(self, state: S) -> bool: ...

    def value(self, state: S, rng: Random) -> float:
        """Player 0's reward in [0, 1]: exact if `state` is terminal, else an
        estimate (a playout, an evaluation, or both), drawing from `rng`."""
        ...


class Node[A: Hashable]:
    """A tree node: the information set reached by its path of actions, the
    keys of `children` from the root.

    `reward` is summed from the side of `mover`, the player who chose the
    action leading here (-1 at the root). `available` counts the visits to
    its parent in whose world that action was legal (Cowling et al.'s
    availability count), which stands in for the parent's visits in UCB1."""

    __slots__ = ("available", "children", "mover", "reward", "visits")

    def __init__(self, mover: int) -> None:
        self.mover = mover
        self.children: dict[A, Node[A]] = {}
        self.visits = 0
        self.reward = 0.0
        self.available = 0

    def size(self) -> int:
        """The nodes in this subtree, this one included."""
        return 1 + sum(child.size() for child in self.children.values())


@dataclass(frozen=True, slots=True)
class Result[A: Hashable]:
    """A search's choice, and the tree it built."""

    move: A
    """The most visited root action, ties broken by mean reward, then by the
    order of the root's legal actions."""
    visits: dict[A, int]
    """Each root action's visits, in the order first expanded."""
    iterations: int
    tree: Node[A]
    """The root of the search tree."""


def ismcts[S, A: Hashable](
    game: Game[S, A],
    sample: Callable[[], S],
    iterations: int,
    rng: Random,
    exploration: float = EXPLORATION,
) -> Result[A]:
    """Search `iterations` times and return the best root action.

    `sample` returns a world consistent with the root information set: a
    fresh one per iteration. The root must not be terminal, and must have the
    same legal actions in every world (as it does when they depend only on
    what the searcher knows). Each iteration descends by UCB1 over the
    actions legal in its world, the parent's visit count replaced by each
    child's availability count, expands one random untried legal action,
    values the state reached with `game.value` and backs the reward up the
    path. `rng` draws the expansions and is passed to `game.value`.
    Deterministic for the same `sample` sequence and `rng` state.
    """
    if iterations < 1:
        msg = f"the search needs at least 1 iteration, not {iterations}"
        raise ValueError(msg)
    root: Node[A] = Node(-1)
    order: Sequence[A] = ()
    for _ in range(iterations):
        world = sample()
        node = root
        path = [root]
        first = True
        while not game.is_terminal(world):
            legal = game.actions(world)
            if first:
                order, first = legal, False
            children = node.children
            untried = []
            for action in legal:
                child = children.get(action)
                if child is None:
                    untried.append(action)
                else:
                    child.available += 1
            if untried:
                action = untried[rng.randrange(len(untried))]
                node = children[action] = Node(game.to_move(world))
                node.available = 1
                world = game.apply(world, action)
                path.append(node)
                break
            action = _select(children, legal, exploration)
            node = children[action]
            world = game.apply(world, action)
            path.append(node)
        value = game.value(world, rng)
        root.visits += 1
        for visited in path[1:]:
            visited.visits += 1
            visited.reward += value if visited.mover == 0 else 1 - value
    best = max(
        (action for action in order if action in root.children),
        key=lambda action: (
            root.children[action].visits,
            root.children[action].reward / root.children[action].visits,
        ),
    )
    return Result(
        best,
        {action: child.visits for action, child in root.children.items()},
        iterations,
        root,
    )


def _select[A: Hashable](
    children: dict[A, Node[A]], legal: Sequence[A], exploration: float
) -> A:
    """The legal action whose child has the highest UCB1 score, by
    availability; the first in `legal` on ties."""
    best = legal[0]
    best_score = -math.inf
    for action in legal:
        child = children[action]
        score = child.reward / child.visits + exploration * math.sqrt(
            math.log(child.available) / child.visits
        )
        if score > best_score:
            best, best_score = action, score
    return best
