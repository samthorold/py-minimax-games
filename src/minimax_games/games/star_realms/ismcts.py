"""The ISMCTS bot: information-set Monte Carlo tree search from its view.

A legitimate player, like the sampler: an ordinary `bots.Bot` that sees only
its view. Each decision runs `search.ismcts` over worlds sampled from the
view (`view.sample_world`), a fresh world per iteration, so a node's value
is averaged over the hidden cards and chance instead of knowing them
(docs/star-realms.md, Search; docs/ismcts.md).

The tree's edges are the decisions `rules.legal_actions` offers, automatic
play happening inside them (`rules.advance_legal`); `EndTurn` leads into the
opponent's turn. A leaf is valued by finishing the turn in progress with a
playout policy's moves (the greedy bot's by default), each made random with
probability `epsilon` (`rollout.playout`), `extra_turns` more turns too,
then `sigmoid(evaluate / scale)` for player 0, or 1 or 0 if the game ended
(`rollout.win_probability`).
"""

import logging
import random

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.action import Action
from minimax_games.games.star_realms.rollout import (
    SCALE,
    Policy,
    greedy,
    heuristic,
    playout,
    win_probability,
)
from minimax_games.games.star_realms.state import GameState
from minimax_games.games.star_realms.view import View, sample_world
from minimax_games.search.ismcts import EXPLORATION, ismcts

logger = logging.getLogger(__name__)

ITERATIONS = 450
"""The ISMCTS bot's default iterations per decision: about the sampler's
time per decision at its defaults, on CPython (docs/star-realms-results.md,
Milestone 5: ISMCTS)."""

EPSILON = 0.25
"""The default chance a leaf's playout move is random instead of the
policy's: the best of 0, 0.1, 0.25 and 0.5 (docs/star-realms-results.md,
Milestone 5: ISMCTS)."""

POLICIES: dict[str, Policy] = {"greedy": greedy, "heuristic": heuristic}
"""The playout policies, by name."""


class StarRealms:
    """Star Realms as an ISMCTS game (`search.ismcts.Game`).

    A leaf's value is player 0's win probability after a playout of `turns`
    turn ends (the rest of the leaf's turn counts as one) by `policy`, each
    move random with probability `epsilon`, scored with `scale`
    (`rollout.playout`, `rollout.win_probability`). With `until`, a leaf is
    played out to the start of turn `until` instead, but at least `turns`
    turn ends, so leaves reached by ending a turn and leaves reached within
    it are scored at the same turn boundary."""

    def __init__(
        self,
        turns: int = 1,
        epsilon: float = EPSILON,
        scale: float = SCALE,
        policy: Policy = greedy,
        until: int | None = None,
    ) -> None:
        self.turns = turns
        self.epsilon = epsilon
        self.scale = scale
        self.policy = policy
        self.until = until

    def actions(self, state: GameState) -> tuple[Action, ...]:
        return rules.legal_actions(state)

    def apply(self, state: GameState, action: Action) -> GameState:
        return rules.advance_legal(state, action)

    def to_move(self, state: GameState) -> int:
        return state.current

    def is_terminal(self, state: GameState) -> bool:
        return rules.winner(state) is not None

    def value(self, state: GameState, rng: random.Random) -> float:
        turns = self.turns
        if self.until is not None:
            turns = max(turns, self.until - state.turn)
        end = playout(state, turns, self.epsilon, rng, self.policy)
        return win_probability(end, 0, self.scale)


class IsmctsBot:
    """Searches `iterations` worlds sampled from its view per decision.

    Exploration constant `exploration`; each leaf is played to the end of
    its turn (and `extra_turns` more), or with `horizon` to the end of the
    `horizon`-th turn from the root's (and at least to the end of its own),
    by the `policy` playout policy (`"greedy"` or `"heuristic"`) made random
    with probability `epsilon`, and scored as `sigmoid(evaluate / scale)`.
    Plays the most visited move; a lone legal action is played without
    searching. Each decision draws a seed for its worlds and one for its
    playouts from an rng seeded with `seed`, so a game replays identically,
    and changing the playouts doesn't change which worlds are sampled.
    """

    def __init__(
        self,
        seed: int,
        iterations: int = ITERATIONS,
        exploration: float = EXPLORATION,
        epsilon: float = EPSILON,
        extra_turns: int = 0,
        scale: float = SCALE,
        horizon: int | None = None,
        policy: str = "greedy",
    ) -> None:
        if iterations < 1:
            msg = f"the search needs at least 1 iteration, not {iterations}"
            raise ValueError(msg)
        if extra_turns < 0:
            msg = f"extra turns can't be negative: {extra_turns}"
            raise ValueError(msg)
        if horizon is not None and horizon < 1:
            msg = f"the horizon must be at least 1 turn, not {horizon}"
            raise ValueError(msg)
        if policy not in POLICIES:
            msg = f"the playout policy is greedy or heuristic, not {policy}"
            raise ValueError(msg)
        self.iterations = iterations
        self.exploration = exploration
        self.extra_turns = extra_turns
        self.horizon = horizon
        self.policy = policy
        self.game = StarRealms(1 + extra_turns, epsilon, scale, POLICIES[policy])
        self._rng = random.Random(seed)

    def settings(self) -> str:
        more = self.extra_turns
        extra = f" + {more} more turn{'s' if more > 1 else ''}" if more else ""
        leaf = f"the turn{extra}"
        if self.horizon is not None:
            leaf = (
                f"the {_ordinal(self.horizon)} turn from the root,"
                f" at least to the end of its own{extra}"
            )
        return (
            f"{self.iterations} iterations per decision,"
            f" exploration {self.exploration:g},"
            f" leaf: epsilon-{self.policy} playout to the end of {leaf}"
            f" (epsilon {self.game.epsilon:g}), scale {self.game.scale:g}"
        )

    def choose(self, view: View, actions: tuple[Action, ...]) -> Action:
        if len(actions) == 1:
            return actions[0]
        worlds = random.Random(self._rng.getrandbits(64))
        playouts = random.Random(self._rng.getrandbits(64))
        game = self.game
        if self.horizon is not None:
            game = StarRealms(
                game.turns,
                game.epsilon,
                game.scale,
                game.policy,
                until=view.turn + self.horizon,
            )
        found = ismcts(
            game,
            lambda: sample_world(view, worlds),
            self.iterations,
            playouts,
            self.exploration,
        )
        assert found.move in actions, (found.move, actions)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("decided %s %s", found.move, found.visits)
        return found.move


def _ordinal(n: int) -> str:
    """1st, 2nd, 3rd, 4th, ..."""
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(
        n % 10 if n % 100 not in (11, 12, 13) else 0
    )
    return f"{n}{suffix or 'th'}"
