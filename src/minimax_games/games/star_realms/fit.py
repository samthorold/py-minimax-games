"""Fit learned values offline and gate them against `evaluate`
(docs/star-realms.md, Learned values).

The data are the turn-boundary states of seeded heuristic-vs-heuristic
games (`boundaries`): each state as `EndTurn` leaves it, before the next
player's automatic play, labelled with whether the player who just ended
their turn went on to win (Monte Carlo targets). With `epsilon`, each move
is, with that chance, a random one other than ending the turn, for a little
variety beyond the deal. A dataset is fully described by its seeds and
settings, so it is generated, never stored.

`run` fits logistic regression and the one-hidden-layer network to the
training games and scores each, by mean log-loss on held-out games (the
seeds after the training ones), against `sigmoid(evaluate / SCALE)` on the
same states and against a constant (the training games' share of wins by
the player who just ended). A model is worth keeping only if it beats
`evaluate`.
"""

import random
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial

import numpy as np

from minimax_games.games.star_realms import features, rules
from minimax_games.games.star_realms.action import EndTurn
from minimax_games.games.star_realms.bots import HeuristicBot
from minimax_games.games.star_realms.rollout import SCALE, win_probability
from minimax_games.games.star_realms.state import GameState
from minimax_games.games.star_realms.value import (
    BATCH,
    EPOCHS,
    HIDDEN,
    L2,
    LEARNING_RATE,
    LOGISTIC,
    Mlp,
    Model,
    fit,
    log_loss,
)
from minimax_games.games.star_realms.view import observe

MAX_TURNS = 300
"""A game past this many turns is left out of the data, as a tournament
leaves it out of the win rates."""


@dataclass(frozen=True, slots=True)
class Game:
    """One game's turn-boundary states, in order, and its winner (`None` if
    it passed `MAX_TURNS`)."""

    states: tuple[GameState, ...]
    winner: int | None


def boundaries(seed: int, epsilon: float = 0.0) -> Game:
    """Play heuristic against heuristic from `seed`, each move replaced with
    probability `epsilon` by a random one other than `EndTurn` (from an rng
    seeded with `seed`), and keep each state `EndTurn` leaves."""
    bot = HeuristicBot()
    rng = random.Random(seed)
    states = []
    state = rules.auto_play(rules.new_game(seed))
    while (winner := rules.winner(state)) is None:
        if state.turn > MAX_TURNS:
            return Game(tuple(states), None)
        actions = rules.legal_actions(state)
        if epsilon and rng.random() < epsilon:
            moves = [a for a in actions if not isinstance(a, EndTurn)] or actions
            action = moves[rng.randrange(len(moves))]
        else:
            action = bot.choose(observe(state, state.current), actions)
        state = rules.apply(state, action)
        if isinstance(action, EndTurn):
            states.append(state)
        state = rules.auto_play(state)
    return Game(tuple(states), winner)


@dataclass(frozen=True, slots=True, eq=False)
class Dataset:
    """Rows of features (`x`, float32: every feature is a small integer),
    whether the player who just ended won (`y`),
    and `sigmoid(evaluate / SCALE)` for that player (`evaluate`), one per
    turn-boundary state of the finished games, with each state's game (its
    index among the seeds) and its position in that game (0 for the first
    turn's end), so temporal-difference targets can be built (`td`)."""

    x: np.ndarray
    y: np.ndarray
    evaluate: np.ndarray
    games: int
    game: np.ndarray
    position: np.ndarray


def _rows(seeds: Sequence[int], first: int = 0, *, epsilon: float) -> Dataset:
    """The data of `seeds`, the first of them game number `first`."""
    xs: list[np.ndarray] = []
    ys: list[float] = []
    ps: list[float] = []
    index: list[int] = []
    positions: list[int] = []
    games = 0
    for i, seed in enumerate(seeds):
        game = boundaries(seed, epsilon)
        if game.winner is None:
            continue
        games += 1
        xs.append(features.encode_all(game.states).astype(np.float32))
        for position, state in enumerate(game.states):
            ended = 1 - state.current
            ys.append(float(game.winner == ended))
            ps.append(win_probability(state, ended, SCALE))
            index.append(first + i)
            positions.append(position)
    x = np.concatenate(xs) if xs else np.zeros((0, len(features.NAMES)), np.float32)
    return Dataset(
        x,
        np.array(ys),
        np.array(ps),
        games,
        np.array(index, dtype=np.int64),
        np.array(positions, dtype=np.int64),
    )


def dataset(seeds: Iterable[int], epsilon: float = 0.0, jobs: int = 1) -> Dataset:
    """The data of the games from `seeds`, played in `jobs` processes, each
    a run of consecutive seeds: the same data in any number of them."""
    seeds = list(seeds)
    size = -(-len(seeds) // jobs) if seeds else 1
    chunks = [seeds[i : i + size] for i in range(0, len(seeds), size)]
    if jobs == 1 or len(chunks) <= 1:
        return _rows(seeds, epsilon=epsilon)
    with ProcessPoolExecutor(jobs) as pool:
        parts = list(
            pool.map(
                partial(_rows, epsilon=epsilon),
                chunks,
                range(0, len(seeds), size),
            )
        )
    return Dataset(
        np.concatenate([p.x for p in parts]),
        np.concatenate([p.y for p in parts]),
        np.concatenate([p.evaluate for p in parts]),
        sum(p.games for p in parts),
        np.concatenate([p.game for p in parts]),
        np.concatenate([p.position for p in parts]),
    )


@dataclass(frozen=True, slots=True)
class Settings:
    """What `run` plays and fits: the training games from `base_seed`, then
    the held-out ones, and the fits' settings (see `value.fit`)."""

    train_games: int = 8000
    test_games: int = 2000
    base_seed: int = 20_000
    epsilon: float = 0.0
    seed: int = 0
    epochs: int = EPOCHS
    batch: int = BATCH
    learning_rate: float = LEARNING_RATE
    l2: float = L2
    hidden: int = HIDDEN
    jobs: int = 1

    @property
    def train_seeds(self) -> range:
        return range(self.base_seed, self.base_seed + self.train_games)

    @property
    def test_seeds(self) -> range:
        start = self.base_seed + self.train_games
        return range(start, start + self.test_games)


@dataclass(frozen=True, slots=True, eq=False)
class Run:
    """A run's held-out log-loss of each predictor by name (`constant`,
    `evaluate`, and each model's form), its data's sizes, and the models."""

    losses: dict[str, float]
    train_states: int
    test_states: int
    train_games: int
    test_games: int
    models: dict[str, Model] = field(default_factory=dict)


def run(settings: Settings) -> Run:
    """Generate the data, fit both forms and score every predictor."""
    train = dataset(settings.train_seeds, settings.epsilon, settings.jobs)
    test = dataset(settings.test_seeds, settings.epsilon, settings.jobs)
    fitted = partial(
        fit,
        x=train.x,
        y=train.y,
        seed=settings.seed,
        epochs=settings.epochs,
        batch=settings.batch,
        learning_rate=settings.learning_rate,
        l2=settings.l2,
        names=features.NAMES,
    )
    models = {
        "logistic": fitted(LOGISTIC),
        "mlp": fitted(Mlp(hidden=settings.hidden)),
    }
    losses = {
        "constant": log_loss(np.full(len(test.y), train.y.mean()), test.y),
        "evaluate": log_loss(test.evaluate, test.y),
        **{name: log_loss(m.predict(test.x), test.y) for name, m in models.items()},
    }
    return Run(
        losses, len(train.y), len(test.y), train.games, test.games, models=models
    )


def report(result: Run, settings: Settings) -> list[str]:
    """The run as plain-text lines: the data, then each log-loss."""
    train, test = settings.train_seeds, settings.test_seeds
    gate = result.losses["evaluate"]
    lines = [
        f"Train: {result.train_games} games (seeds {train.start}-{train.stop - 1}),"
        f" {result.train_states} states",
        f"Test: {result.test_games} games (seeds {test.start}-{test.stop - 1}),"
        f" {result.test_states} states",
        f"Fit: seed {settings.seed}, {settings.epochs} epochs, batch"
        f" {settings.batch}, learning rate {settings.learning_rate:g},"
        f" l2 {settings.l2:g}, {settings.hidden} hidden units,"
        f" epsilon {settings.epsilon:g}",
        "",
        "Held-out log-loss:",
    ]
    for name, loss in result.losses.items():
        verdict = ""
        if name in result.models:
            verdict = (
                " (beats evaluate)" if loss < gate else " (does not beat evaluate)"
            )
        label = f"sigmoid(evaluate / {SCALE:g})" if name == "evaluate" else name
        lines.append(f"  {label}: {loss:.4f}{verdict}")
    return lines
