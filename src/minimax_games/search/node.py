from typing import Any, Iterator, Protocol, Self


class Node(Protocol):
    """Comparison operators allows users to rank on more than just score."""

    moves: list[Any]

    def __gt__(self, other: Self) -> bool:
        ...

    def __lt__(self, other: Self) -> bool:
        ...

    def __ge__(self, other: Self) -> bool:
        ...

    def __le__(self, other: Self) -> bool:
        ...

    def score(self) -> int:
        ...

    def is_terminal(self) -> bool:
        ...

    def children(self) -> Iterator[Self]:
        ...

    def is_maximising(self) -> bool:
        ...

    def minimum(self) -> Self:
        ...

    def maximum(self) -> Self:
        ...
