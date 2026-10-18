import copy
import logging
import operator
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from enum import Enum, auto

from minimax_games.search.node import Node

logger = logging.getLogger(__name__)


def one_ply(node: Node) -> int:
    """The default `ply_cost`: every move uses up one ply of the budget."""
    return 1


class Bound(Enum):
    """What a transposition table entry's node says about the true value."""

    EXACT = auto()
    LOWER = auto()
    """The search failed high: the true value is at least the node's."""
    UPPER = auto()
    """The search failed low: the true value is at most the node's."""


@dataclass(frozen=True, slots=True)
class Entry[N: Node]:
    """A searched node's result: its best leaf and what that leaf bounds.

    `line` is the leaf's moves after the searched node, so the leaf can be
    returned under whatever path reaches the node next time.
    """

    leaf: N
    line: tuple[object, ...]
    bound: Bound


type Table[N: Node] = dict[tuple[Hashable, int | None], Entry[N]]
"""A transposition table: entries by the node's key and remaining depth."""


def alphabeta[N: Node](
    node: N,
    a: N,
    b: N,
    prune_on_ties: bool = True,
    depth: int | None = None,
    ply_cost: Callable[[N], int] = one_ply,
    key: Callable[[N], Hashable] | None = None,
    table: Table[N] | None = None,
) -> N:
    """Return the best leaf reachable from `node`, searching with alpha-beta.

    `a` and `b` are the initial bounds, usually `node.minimum()` and
    `node.maximum()`. A leaf is a node that is terminal or has no children:
    either way it is returned as is, so the result is always a real node,
    never the `minimum()` / `maximum()` sentinel.

    `prune_on_ties` chooses the cutoff comparison. When true (the default),
    the standard `>=` / `<=` cutoff is used, so a branch is pruned as soon as
    its value ties the bound. When false, the strict `>` / `<` is used: the
    result is still correct but fewer branches are pruned. Either way the
    search is fail-soft: it returns the best node found rather than clamping
    it to the `a` / `b` window.

    `depth` limits how many plies below `node` are searched (see
    `ply_cost`). When it reaches 0 the node is treated as a leaf, so its
    `score()` acts as a heuristic evaluation. `None` (the default) searches
    all the way to terminal nodes.

    `ply_cost` says how much of the `depth` budget moving into a node uses
    up: each child is searched with `depth - ply_cost(child)`, and a node is
    treated as a leaf once its budget is 0 or less. The default costs 1 per
    node, so `depth` counts plies. A game whose turns span several moves can
    return 1 for a node that ends a turn and 0 otherwise, so `depth` counts
    turns and the search only stops at turn boundaries. Free nodes are always
    searched through, so with a cost of 0 everywhere a positive `depth`
    is never reached. `ply_cost` is ignored when `depth` is `None`.

    `key` turns on a transposition table, so a position reached again by
    another order of moves is not searched again. It maps a node to a hashable
    key that must capture everything its subtree's result depends on, other
    than the path to it and the remaining `depth` (which the table adds): the
    position, the side to move, and anything `score()` reads, such as the
    number of moves made. `None` (the default) searches without a table. Each
    entry records whether its result is exact or, having failed high or low
    under the window it was searched with, only a lower or upper bound, and is
    reused only where that is enough for the current window, so the result is
    the same as without a table up to ties between equally good lines. A
    reused leaf is returned with its moves rebuilt from the current path, so
    `moves` is always a real line from the root: this copies the stored leaf
    (`copy.copy`) and assigns its `moves`.

    `table` is the table to use, so one can be shared between searches with
    the same `key`, `ply_cost` and `prune_on_ties` (an iterative deepening
    loop, say); by default each search starts an empty one. It is ignored
    without a `key`.
    """
    debug = logger.isEnabledFor(logging.DEBUG)
    if debug:
        logger.debug("got node=%s a=%s b=%s", node.moves, a.moves, b.moves)
    if node.is_terminal() or (depth is not None and depth <= 0):
        if debug:
            logger.debug("leaf")
            logger.debug("returned node=%s a=%s b=%s", node.moves, a.moves, b.moves)
        return node

    fails_high = operator.ge if prune_on_ties else operator.gt
    fails_low = operator.le if prune_on_ties else operator.lt

    entry_key = None
    if key is not None:
        if table is None:
            table = {}
        entry_key = (key(node), depth)
        entry = table.get(entry_key)
        if entry is not None and (
            entry.bound is Bound.EXACT
            or (entry.bound is Bound.LOWER and fails_high(entry.leaf, b))
            or (entry.bound is Bound.UPPER and fails_low(entry.leaf, a))
        ):
            if debug:
                logger.debug("transposition %s", entry.bound.name)
            return _rebase(entry, node)

    a0, b0 = a, b
    best_node = node.minimum() if node.is_maximising() else node.maximum()

    has_children = False
    for child in node.children():
        has_children = True
        value = alphabeta(
            node=child,
            a=a,
            b=b,
            prune_on_ties=prune_on_ties,
            depth=None if depth is None else depth - ply_cost(child),
            ply_cost=ply_cost,
            key=key,
            table=table,
        )
        if node.is_maximising():
            best_node = max(best_node, value)
            a = max(a, best_node)
            if fails_high(best_node, b):
                break
        else:
            best_node = min(best_node, value)
            b = min(b, best_node)
            if fails_low(best_node, a):
                break

    if not has_children:
        if debug:
            logger.debug("no children, treated as a leaf")
        return node

    if table is not None and entry_key is not None:
        # Fail-soft: a result outside the original window only bounds the
        # true value. On the window's edge it is treated as a bound too, which
        # is safe whichever cutoff comparison was used.
        if best_node >= b0:
            bound = Bound.LOWER
        elif best_node <= a0:
            bound = Bound.UPPER
        else:
            bound = Bound.EXACT
        line = tuple(best_node.moves[len(node.moves) :])
        table[entry_key] = Entry(best_node, line, bound)

    if debug:
        logger.debug("returned node=%s a=%s b=%s", node.moves, a.moves, b.moves)

    return best_node


def _rebase[N: Node](entry: Entry[N], node: N) -> N:
    """The entry's leaf as reached from `node`: `node`'s moves, then the
    entry's line. The stored leaf itself if it was reached the same way."""
    leaf = entry.leaf
    moves = [*node.moves, *entry.line]
    if leaf.moves == moves:
        return leaf
    leaf = copy.copy(leaf)
    leaf.moves = moves
    return leaf
