from collections.abc import Iterator
from typing import Any, Protocol, Self


class Node(Protocol):
    """Comparison operators allows users to rank on more than just score."""

    moves: list[Any]

    def __gt__(self, other: Self) -> bool: ...

    def __lt__(self, other: Self) -> bool: ...

    def __ge__(self, other: Self) -> bool: ...

    def __le__(self, other: Self) -> bool: ...

    def score(self) -> int:
        """The node's value, typically what the comparison operators rank on.

        Without a depth limit `alphabeta` only scores leaves (terminal or
        childless nodes). With `depth` set, nodes at the limit are scored
        too, so `score()` must then be a meaningful heuristic evaluation of
        non-terminal positions.
        """
        ...

    def is_terminal(self) -> bool: ...

    def children(self) -> Iterator[Self]:
        """Nodes reachable in one move.

        A non-terminal node that yields no children is treated as a leaf by
        `alphabeta`: it is scored as it stands.
        """
        ...

    def is_maximising(self) -> bool: ...

    def minimum(self) -> Self: ...

    def maximum(self) -> Self: ...
