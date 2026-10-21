"""A sequential probability ratio test on a match's wins and losses.

Wald's SPRT between two hypotheses about bot A's win rate in finished
games, H0 (`p0`) and H1 (`p1`, above it): after each game, the log of the
likelihood ratio of the record so far under H1 and H0 is compared with two
bounds set by the error rates. Above the upper bound H1 is accepted, below
the lower H0; between them the match goes on. `alpha` is the chance of
accepting H1 when H0 is true, `beta` of accepting H0 when H1 is. A clear
result settles in tens of games; a win rate between the hypotheses takes
longest, which is what the game limit (`tournament.play_sprt`) is for.
"""

import enum
import math
from dataclasses import dataclass

Z = 1.959964
"""The normal quantile of the 95% interval `wilson` gives."""


class Hypothesis(enum.Enum):
    H0 = "H0"
    H1 = "H1"


@dataclass(frozen=True, slots=True)
class Sprt:
    p0: float = 0.5
    """Bot A's win rate under H0."""
    p1: float = 0.55
    """Bot A's win rate under H1: above `p0`."""
    alpha: float = 0.05
    """The chance of accepting H1 when H0 is true."""
    beta: float = 0.05
    """The chance of accepting H0 when H1 is true."""

    def __post_init__(self) -> None:
        if not 0 < self.p0 < self.p1 < 1:
            msg = f"need 0 < p0 < p1 < 1, not p0 {self.p0}, p1 {self.p1}"
            raise ValueError(msg)
        if not (self.alpha > 0 and self.beta > 0 and self.alpha + self.beta < 1):
            msg = (
                "need positive error rates summing to under 1,"
                f" not alpha {self.alpha}, beta {self.beta}"
            )
            raise ValueError(msg)

    @property
    def lower(self) -> float:
        """At or below this log-likelihood ratio, H0 is accepted."""
        return math.log(self.beta / (1 - self.alpha))

    @property
    def upper(self) -> float:
        """At or above this log-likelihood ratio, H1 is accepted."""
        return math.log((1 - self.beta) / self.alpha)

    def llr(self, wins: int, losses: int) -> float:
        """The log-likelihood ratio, H1 over H0, of A's `wins` and `losses`."""
        return wins * math.log(self.p1 / self.p0) + losses * math.log(
            (1 - self.p1) / (1 - self.p0)
        )

    def decide(self, wins: int, losses: int) -> Hypothesis | None:
        """The hypothesis A's record accepts, or None to play on."""
        llr = self.llr(wins, losses)
        if llr >= self.upper:
            return Hypothesis.H1
        if llr <= self.lower:
            return Hypothesis.H0
        return None


def wilson(wins: int, games: int, z: float = Z) -> tuple[float, float]:
    """The Wilson score interval of the win rate `wins / games` (95% by
    default): it stays inside [0, 1], unlike the normal one, for records
    near all wins or all losses. With no games, it is all of [0, 1]."""
    if not games:
        return 0.0, 1.0
    rate = wins / games
    centre = rate + z * z / (2 * games)
    spread = z * math.sqrt(rate * (1 - rate) / games + z * z / (4 * games * games))
    scale = 1 + z * z / games
    return (centre - spread) / scale, (centre + spread) / scale
