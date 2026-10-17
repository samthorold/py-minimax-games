import logging
import operator

from minimax_games.search.node import Node

logger = logging.getLogger(__name__)


def alphabeta(node: Node, a: Node, b: Node, soft: bool = True) -> Node:
    """Return the best leaf reachable from `node`, searching with alpha-beta.

    `a` and `b` are the initial bounds, usually `node.minimum()` and
    `node.maximum()`. A leaf is a node that is terminal or has no children:
    either way it is returned as is, so the result is always a real node,
    never the `minimum()` / `maximum()` sentinel.
    """
    logger.debug("got node=%s a=%s b=%s", node.moves, a.moves, b.moves)
    if node.is_maximising():
        logger.debug("got node=%s a=%s b=%s", node.score(), a.score(), b.score())
    if node.is_terminal():
        logger.debug("terminal")
        logger.debug("returned node=%s a=%s b=%s", node.moves, a.moves, b.moves)
        return node

    fails_high = operator.ge if soft else operator.gt
    fails_low = operator.le if soft else operator.lt

    best_node = node.minimum() if node.is_maximising() else node.maximum()

    has_children = False
    for child in node.children():
        has_children = True
        if node.is_maximising():
            best_node = max(best_node, alphabeta(node=child, a=a, b=b, soft=soft))
            a = max(a, best_node)
            if fails_high(best_node, b):
                break
        else:
            best_node = min(best_node, alphabeta(node=child, a=a, b=b, soft=soft))
            b = min(b, best_node)
            if fails_low(best_node, a):
                break

    if not has_children:
        logger.debug("no children, treated as a leaf")
        return node

    logger.debug("returned node=%s a=%s b=%s", node.moves, a.moves, b.moves)

    return best_node
