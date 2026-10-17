import logging
import operator

from minimax_games.search.node import Node

logger = logging.getLogger(__name__)


def alphabeta[N: Node](
    node: N,
    a: N,
    b: N,
    prune_on_ties: bool = True,
    depth: int | None = None,
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

    `depth` limits how many plies below `node` are searched. When it reaches
    0 the node is treated as a leaf, so its `score()` acts as a heuristic
    evaluation. `None` (the default) searches all the way to terminal nodes.
    """
    logger.debug("got node=%s a=%s b=%s", node.moves, a.moves, b.moves)
    if node.is_terminal() or depth == 0:
        logger.debug("leaf")
        logger.debug("returned node=%s a=%s b=%s", node.moves, a.moves, b.moves)
        return node

    fails_high = operator.ge if prune_on_ties else operator.gt
    fails_low = operator.le if prune_on_ties else operator.lt

    child_depth = None if depth is None else depth - 1

    best_node = node.minimum() if node.is_maximising() else node.maximum()

    has_children = False
    for child in node.children():
        has_children = True
        value = alphabeta(
            node=child, a=a, b=b, prune_on_ties=prune_on_ties, depth=child_depth
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
        logger.debug("no children, treated as a leaf")
        return node

    logger.debug("returned node=%s a=%s b=%s", node.moves, a.moves, b.moves)

    return best_node
