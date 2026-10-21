"""The sampler: perfect-information Monte Carlo (PIMC) over sampled worlds.

A legitimate player, unlike the cheat: an ordinary `bots.Bot` that sees only
its view. Each decision it samples worlds consistent with the view
(`view.sample_world`), runs the cheat's budgeted search on each
(`cheat.decide`) and plays the move most of them chose
(docs/star-realms.md, Search).
"""

import logging
import random
from collections import Counter
from dataclasses import dataclass

from minimax_games.games.star_realms import cheat
from minimax_games.games.star_realms.action import Action
from minimax_games.games.star_realms.cheat import BUDGET
from minimax_games.games.star_realms.evaluate import LINEAR, Evaluator
from minimax_games.games.star_realms.view import View, sample_world

logger = logging.getLogger(__name__)

WORLDS = 4
"""The sampler's default number of worlds per decision
(docs/star-realms-results.md, Milestone 3)."""


@dataclass(frozen=True, slots=True)
class Vote:
    """A sampler's decision: deterministic for the same view, actions, rng
    state and budget."""

    move: Action
    votes: Counter[Action]
    """How many worlds chose each move."""
    nodes: int
    """Nodes generated over every world: at most the budget."""
    depths: tuple[int, ...]
    """Each world's `cheat.Decision.depth`, in the order sampled: the
    deepest search, in turns, that finished within its share."""


def vote(
    view: View,
    actions: tuple[Action, ...],
    rng: random.Random,
    worlds: int,
    turns: int,
    budget: int,
    evaluator: Evaluator = LINEAR,
) -> Vote:
    """The sampler's move among `actions`, the legal actions in `view`.

    Samples `worlds` worlds from `view` with `rng` and runs `cheat.decide` on
    each, `turns` turns deep with `budget // worlds` nodes, scoring with
    `evaluator`. The move chosen
    in most worlds wins, ties broken by the order of `actions`. Every world
    shares the view's public state, so each world's move is one of `actions`.
    """
    per_world = budget // worlds
    votes: Counter[Action] = Counter()
    nodes = 0
    depths = []
    for _ in range(worlds):
        decision = cheat.decide(sample_world(view, rng), turns, per_world, evaluator)
        assert decision.move in actions, (decision.move, actions)
        votes[decision.move] += 1
        nodes += decision.nodes
        depths.append(decision.depth)
    move = max(actions, key=lambda action: votes[action])
    return Vote(move, votes, nodes, tuple(depths))


class SamplerBot:
    """Samples `worlds` worlds from its view per decision and votes.

    On each world it runs `cheat.decide`, `turns` turns deep (by default 2),
    with `budget // worlds` nodes, scoring positions with `evaluator` (by
    default the linear `evaluate`), so a decision searches at most `budget`
    nodes in all (by default the cheat's `BUDGET`, so the two compare at
    equal compute). The move chosen in most worlds wins, ties broken by the
    order of the legal actions. The worlds are drawn from an rng seeded with
    `seed`, so a game replays identically.
    """

    def __init__(
        self,
        seed: int,
        worlds: int = WORLDS,
        turns: int = 2,
        budget: int = BUDGET,
        evaluator: Evaluator = LINEAR,
    ) -> None:
        if worlds < 1:
            msg = f"the sampler needs at least 1 world, not {worlds}"
            raise ValueError(msg)
        if turns < 1:
            msg = f"the horizon must be at least 1 turn, not {turns}"
            raise ValueError(msg)
        if budget < worlds:
            msg = (
                f"the budget must be at least 1 node per world:"
                f" {budget} nodes for {worlds} worlds"
            )
            raise ValueError(msg)
        self.worlds = worlds
        self.turns = turns
        self.budget = budget
        self.evaluator = evaluator
        self._rng = random.Random(seed)

    def settings(self) -> str:
        return (
            f"{self.worlds} worlds, {self.turns}-turn horizon,"
            f" budget {self.budget} nodes per decision"
            f" ({self.budget // self.worlds} per world), {self.evaluator.describe()}"
        )

    def choose(self, view: View, actions: tuple[Action, ...]) -> Action:
        decision = vote(
            view,
            actions,
            self._rng,
            self.worlds,
            self.turns,
            self.budget,
            self.evaluator,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("decided %s", decision)
        return decision.move
