"""Rollout evaluation: score a state by playing it out (docs/star-realms.md,
Search).

A rollout plays `turns` more turn ends from a state with the greedy bot's
moves, each replaced with probability `epsilon` by a random move other than
ending the turn, then scores the state it reached as a win probability for
the searching player: 1 or 0 if the game ended, else
`sigmoid(evaluate / scale)`. `Rollout` averages `rollouts` of them and maps
the mean into the integer scores alpha-beta compares, `-SPAN..SPAN`, so a
won or lost game in the search tree (`WIN`, far beyond) still ranks above
or below any rollout.

Deterministic: the rollouts' rng is seeded from the settings' `seed`, the
perspective and the state's own shuffle seed and turn, so the same state and
settings always give the same score, whatever order a search scores its
leaves in, in any process. Sibling leaves of the same turn usually share a
shuffle seed, so they are played out with the same random numbers, which
makes the comparison between them less noisy.
"""

import math
import random
from dataclasses import dataclass

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.action import Action, EndTurn
from minimax_games.games.star_realms.bots import greedy_move
from minimax_games.games.star_realms.evaluate import WEIGHTS, WIN, Weights, evaluate
from minimax_games.games.star_realms.state import GameState

SPAN = 10**6
"""Rollout scores lie in `-SPAN..SPAN`: a sure loss to a sure win."""

SCALE = 16.0
"""The default `Rollout.scale`: `evaluate` points per unit of log-odds,
fitted by maximum likelihood to who won from 7,141 turn starts of 300
heuristic-vs-heuristic games (docs/star-realms-results.md, Milestone 5:
rollouts)."""

MAX_TURNS = 300
"""A playout stops after this many turns of the game, whatever `turns`
says, as a tournament's games do: random moves could make it go on."""


def greedy(state: GameState, actions: tuple[Action, ...]) -> Action:
    """The greedy bot's move among `actions`, the legal actions in `state`.

    It reads only what the mover knows (the legal actions come from their
    hand and the public state), so a playout through a sampled or true world
    gives neither player any clairvoyance."""
    return greedy_move(state.must_scrap, actions)


def playout(
    state: GameState, turns: int, epsilon: float, rng: random.Random
) -> GameState:
    """`state` after `turns` more turn ends (or the end of the game), each
    move greedy's or, with probability `epsilon`, a random move other than
    ending the turn (when there is one): random play would otherwise end
    turns with trade and combat unspent."""
    end = state.turn + turns
    last = min(end, MAX_TURNS)
    while state.turn < last and rules.winner(state) is None:
        actions = rules.legal_actions(state)
        if epsilon and rng.random() < epsilon:
            moves = [a for a in actions if not isinstance(a, EndTurn)] or actions
            action = moves[rng.randrange(len(moves))]
        else:
            action = greedy(state, actions)
        state = rules.advance_legal(state, action)
    return state


def win_probability(
    state: GameState, perspective: int, scale: float, weights: Weights = WEIGHTS
) -> float:
    """How likely `perspective` is to win from `state`: 1 or 0 if the game
    is over, else `sigmoid(evaluate / scale)`."""
    score = evaluate(state, perspective, weights)
    if abs(score) == WIN:
        return 1.0 if score > 0 else 0.0
    return 1 / (1 + math.exp(-score / scale))


@dataclass(frozen=True, slots=True)
class Rollout:
    """The rollout evaluation: an evaluator a search scores its leaves with
    (`evaluator(state, perspective) -> int`), as it would with `Linear`."""

    rollouts: int = 1
    """Playouts per state, averaged."""
    turns: int = 2
    """Turn ends each playout plays (the rest of this turn counts as one)."""
    epsilon: float = 0.1
    """The chance each playout move is random instead of greedy's."""
    scale: float = SCALE
    """`evaluate` points per unit of log-odds of winning."""
    seed: int = 0
    """Seeds the playouts' random moves, with the state."""
    weights: Weights = WEIGHTS
    """The weights `evaluate` scores a playout's last state with."""

    def __call__(self, state: GameState, perspective: int) -> int:
        """`state`'s score for `perspective`: `WIN` or `-WIN` if the game is
        over, else the playouts' mean win probability in `-SPAN..SPAN`."""
        won_by = rules.winner(state)
        if won_by is not None:
            return WIN if won_by == perspective else -WIN
        rng = random.Random(
            ((self.seed * 2 + perspective) << 80)
            | (state.rng_seed << 16)
            | min(state.turn, 0xFFFF)
        )
        return round(SPAN * (2 * self.value(state, perspective, rng) - 1))

    def value(self, state: GameState, perspective: int, rng: random.Random) -> float:
        """The mean win probability for `perspective` over `rollouts`
        playouts from `state`, drawing random moves from `rng`."""
        total = 0.0
        for _ in range(self.rollouts):
            end = playout(state, self.turns, self.epsilon, rng)
            total += win_probability(end, perspective, self.scale, self.weights)
        return total / self.rollouts

    def describe(self) -> str:
        """The settings, as a phrase: all but the seed, which a tournament
        sets per game."""
        return (
            f"rollout evaluation ({self.rollouts} x {self.turns} turns,"
            f" epsilon {self.epsilon:g}, scale {self.scale:g})"
        )
