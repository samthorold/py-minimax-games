import logging
import operator
from collections.abc import Callable

from minimax_games.search.node import Node

logger = logging.getLogger(__name__)


def one_ply(node: Node) -> int:
    """The default `ply_cost`: every move uses up one ply of the budget."""
    return 1


def alphabeta[N: Node](
    node: N,
    a: N,
    b: N,
    prune_on_ties: bool = True,
    depth: int | None = None,
    ply_cost: Callable[[N], int] = one_ply,
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

    if debug:
        logger.debug("returned node=%s a=%s b=%s", node.moves, a.moves, b.moves)

    return best_node
