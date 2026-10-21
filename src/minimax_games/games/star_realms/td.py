"""Temporal-difference targets for the learned value (docs/star-realms.md,
Learned values: Targets).

A Monte Carlo target labels each turn-boundary state with its game's final
result, for the player who just ended: unbiased, but one noisy draw. A
TD(λ) target is the λ-return instead: built partly from a current value of
the next turn-boundary state of the same game, and partly from that state's
own target, recursively, back from the game's end:

    G_last = y_last
    G_t    = (1 - λ) (1 - V(s_t+1)) + λ (1 - G_t+1)

The `1 - ...` is the change of side: the next state was left by the other
player (the players alternate, one `EndTurn` per turn), and V is always the
chance that the player who just ended wins. The last state of a game has no
next state, so its target is the result, whatever λ. λ = 1 gives back the
Monte Carlo targets; λ = 0 is one-step TD, the next state's value alone.
"""

import numpy as np


def targets(
    y: np.ndarray,
    game: np.ndarray,
    position: np.ndarray,
    value: np.ndarray,
    lam: float,
) -> np.ndarray:
    """The TD(`lam`) targets of rows labelled `y` (1.0 if the player who
    just ended won), each row's `game` and `position` in it (as
    `selfplay.SelfPlay` and `fit.Dataset` record them), and `value`, a
    current value's win probability of each row for its own player. The
    rows must be whole games, each in order from position 0."""
    if not 0 <= lam <= 1:
        msg = f"lam must be between 0 and 1, not {lam}"
        raise ValueError(msg)
    y = np.asarray(y, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    game, position = np.asarray(game), np.asarray(position)
    n = len(y)
    if not len(game) == len(position) == len(value) == n:
        msg = "y, game, position and value must have a row each"
        raise ValueError(msg)
    starts = np.ones(n, dtype=bool)
    starts[1:] = game[1:] != game[:-1]
    expected = np.zeros(n, dtype=np.int64)
    if n:
        # the position each row must have: its distance from its game's start
        start_index = np.maximum.accumulate(np.where(starts, np.arange(n), 0))
        expected = np.arange(n) - start_index
    if not np.array_equal(position, expected) or len(np.unique(game[starts])) != int(
        starts.sum()
    ):
        msg = "rows must be whole games, each in order from position 0"
        raise ValueError(msg)
    last = np.ones(n, dtype=bool)
    last[:-1] = starts[1:]
    found = np.empty(n)
    following = 0.0
    for t in range(n - 1, -1, -1):
        if last[t]:
            following = y[t]
        else:
            following = (1 - lam) * (1 - value[t + 1]) + lam * (1 - following)
        found[t] = following
    return found
