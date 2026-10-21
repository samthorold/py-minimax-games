"""Learned values: models of how likely the player who just ended their turn
is to win, fitted to game results (docs/star-realms.md, Learned values).

Two forms, hand-written in numpy: `LOGISTIC`, logistic regression on the
features (`features.encode`), and `Mlp`, a network with one hidden layer of
tanh units. Each gives its loss, the mean log-loss plus `l2 / 2` times the
sum of its squared weights (not biases), with the loss's gradient worked out
by hand (the tests check it against finite differences). `fit` trains one
with Adam on shuffled minibatches, deterministic given its seed.

A `Model` is a form's parameters with the features' names and the mean and
spread it standardises them by. It is saved as an `.npz` of plain arrays,
never a pickle (which runs code when loaded), and models in the package's
`models/` directory load by name (`load`). `ValueEvaluator` wraps one as an
`evaluate.Evaluator`.
"""

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from minimax_games.games.star_realms import features, rules
from minimax_games.games.star_realms.evaluate import WIN
from minimax_games.games.star_realms.rollout import SPAN
from minimax_games.games.star_realms.state import GameState

type Params = dict[str, np.ndarray]
"""A model's parameters by name."""

EPOCHS = 10
"""`fit`'s passes over the training data."""
BATCH = 256
"""`fit`'s minibatch size."""
LEARNING_RATE = 1e-3
"""Adam's step size in `fit`."""
L2 = 3e-3
"""The loss's weight penalty in `fit`."""
HIDDEN = 32
"""`Mlp`'s hidden units unless told otherwise."""


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + np.tanh(0.5 * z))


def _log_loss_of_logits(z: np.ndarray, y: np.ndarray) -> float:
    """The mean log-loss of predictions `sigmoid(z)` for labels `y`, computed
    from the logits so that confident predictions don't overflow."""
    return float(np.mean(np.logaddexp(0, z) - y * z))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    """The mean log-loss of probabilities `p` for 0/1 labels `y`."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class Form(Protocol):
    """A kind of model: how its parameters start, predict and learn."""

    @property
    def name(self) -> str:
        """The form's name, as a model file records it."""
        ...

    def init(self, columns: int, rng: np.random.Generator) -> Params:
        """Starting parameters for rows of `columns` features."""
        ...

    def predict(self, params: Params, x: np.ndarray) -> np.ndarray:
        """The win probability for each row of `x`."""
        ...

    def loss_and_gradient(
        self, params: Params, x: np.ndarray, y: np.ndarray, l2: float
    ) -> tuple[float, Params]:
        """The loss on rows `x` with labels `y`, and its gradient."""
        ...


@dataclass(frozen=True, slots=True)
class Logistic:
    """Logistic regression: `sigmoid(x @ w + b)`."""

    name: str = "logistic"

    def init(self, columns: int, rng: np.random.Generator) -> Params:
        return {"w": np.zeros(columns), "b": np.zeros(1)}

    def predict(self, params: Params, x: np.ndarray) -> np.ndarray:
        return _sigmoid(x @ params["w"] + params["b"][0])

    def loss_and_gradient(
        self, params: Params, x: np.ndarray, y: np.ndarray, l2: float
    ) -> tuple[float, Params]:
        w = params["w"]
        z = x @ w + params["b"][0]
        loss = _log_loss_of_logits(z, y) + 0.5 * l2 * float(w @ w)
        dz = (_sigmoid(z) - y) / len(y)
        return loss, {"w": x.T @ dz + l2 * w, "b": np.array([dz.sum()])}


@dataclass(frozen=True, slots=True)
class Mlp:
    """One hidden layer of tanh units: `sigmoid(tanh(x @ w1 + b1) @ w2 + b2)`.
    `hidden` only sizes new parameters; a model's own say how many it has."""

    hidden: int = HIDDEN
    name: str = "mlp"

    def init(self, columns: int, rng: np.random.Generator) -> Params:
        # Glorot-style scales keep the tanh units off their flat tails.
        return {
            "w1": rng.normal(scale=1 / np.sqrt(columns), size=(columns, self.hidden)),
            "b1": np.zeros(self.hidden),
            "w2": rng.normal(scale=1 / np.sqrt(self.hidden), size=self.hidden),
            "b2": np.zeros(1),
        }

    def predict(self, params: Params, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ params["w1"] + params["b1"])
        return _sigmoid(h @ params["w2"] + params["b2"][0])

    def loss_and_gradient(
        self, params: Params, x: np.ndarray, y: np.ndarray, l2: float
    ) -> tuple[float, Params]:
        w1, w2 = params["w1"], params["w2"]
        h = np.tanh(x @ w1 + params["b1"])
        z = h @ w2 + params["b2"][0]
        penalty = 0.5 * l2 * (float(np.sum(w1 * w1)) + float(w2 @ w2))
        loss = _log_loss_of_logits(z, y) + penalty
        dz = (_sigmoid(z) - y) / len(y)
        dh = np.outer(dz, w2) * (1 - h * h)
        return loss, {
            "w1": x.T @ dh + l2 * w1,
            "b1": dh.sum(axis=0),
            "w2": h.T @ dz + l2 * w2,
            "b2": np.array([dz.sum()]),
        }


LOGISTIC = Logistic()
FORMS: dict[str, Form] = {"logistic": LOGISTIC, "mlp": Mlp()}
"""Each form by name, as a model file records it."""


@dataclass(frozen=True, slots=True, eq=False)
class Model:
    """A fitted value: `form`'s `params` over standardised features
    (`(x - mean) / std`), the features named `names`."""

    form: str
    names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    params: Params

    def predict(self, x: np.ndarray) -> np.ndarray:
        """The win probability for each row of raw features `x`."""
        return FORMS[self.form].predict(self.params, (x - self.mean) / self.std)

    def save(self, path: Path) -> None:
        """Write the model to `path` as an `.npz` of plain arrays."""
        arrays: dict[str, Any] = {
            "form": np.array(self.form),
            "names": np.array(self.names),
            "mean": self.mean,
            "std": self.std,
            **{f"param_{name}": array for name, array in self.params.items()},
        }
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "Model":
        """The model saved at `path`. Never unpickles anything."""
        with np.load(path, allow_pickle=False) as data:
            return cls(
                form=str(data["form"]),
                names=tuple(str(name) for name in data["names"]),
                mean=data["mean"],
                std=data["std"],
                params={
                    key.removeprefix("param_"): data[key]
                    for key in data.files
                    if key.startswith("param_")
                },
            )


def _models() -> Path:
    return Path(str(resources.files(__package__).joinpath("models")))


def load(name: str) -> Model:
    """The packaged model `name` (`models/<name>.npz`)."""
    path = _models() / f"{name}.npz"
    if not path.is_file():
        msg = f"no model {name!r} in {path.parent}"
        raise FileNotFoundError(msg)
    return Model.load(path)


def fit(
    form: Form,
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    epochs: int = EPOCHS,
    batch: int = BATCH,
    learning_rate: float = LEARNING_RATE,
    l2: float = L2,
    names: tuple[str, ...] | None = None,
) -> Model:
    """`form` fitted to rows `x` and 0/1 labels `y` by Adam on minibatches.

    The features are standardised by their mean and spread over `x` (a
    feature that never varies is left unscaled), in `x`'s own precision:
    float32 halves a big dataset's memory. The starting parameters and each
    epoch's shuffle come from `seed`, so a fit is deterministic."""
    rng = np.random.default_rng(seed)
    mean = x.mean(axis=0, dtype=np.float64)
    std = x.std(axis=0, dtype=np.float64)
    std[std == 0] = 1
    z = (x - mean.astype(x.dtype)) / std.astype(x.dtype)
    params = form.init(x.shape[1], rng)
    first = {name: np.zeros_like(a) for name, a in params.items()}
    second = {name: np.zeros_like(a) for name, a in params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch):
            rows = order[start : start + batch]
            _, grads = form.loss_and_gradient(params, z[rows], y[rows], l2)
            step += 1
            for name, grad in grads.items():
                first[name] = beta1 * first[name] + (1 - beta1) * grad
                second[name] = beta2 * second[name] + (1 - beta2) * grad * grad
                m = first[name] / (1 - beta1**step)
                v = second[name] / (1 - beta2**step)
                params[name] -= learning_rate * m / (np.sqrt(v) + eps)
    if names is None:
        names = tuple(f"x{i}" for i in range(x.shape[1]))
    return Model(form.name, names, mean, std, params)


@dataclass(frozen=True, slots=True, eq=False)
class ValueEvaluator:
    """A learned value as an `evaluate.Evaluator`: `WIN` or `-WIN` for a
    finished game, else the model's win probability mapped to `-SPAN..SPAN`,
    as `rollout.Rollout` maps its own. The model estimates how likely the
    player who just ended their turn is to win, so the other player's score
    is its negation. It is only meant for turn boundaries, the states the
    whole-turn planner scores."""

    model: Model

    def __post_init__(self) -> None:
        if self.model.names != features.NAMES:
            msg = "the model was fitted to other features than features.NAMES"
            raise ValueError(msg)

    def __call__(self, state: GameState, perspective: int) -> int:
        won_by = rules.winner(state)
        if won_by is not None:
            return WIN if won_by == perspective else -WIN
        p = float(self.model.predict(features.encode(state)[None, :])[0])
        score = round(SPAN * (2 * p - 1))
        return score if perspective == 1 - state.current else -score

    def score_rows(
        self,
        x: np.ndarray,
        winner: np.ndarray,
        current: np.ndarray,
        perspective: int,
    ) -> np.ndarray:
        """`__call__`'s scores for many states at once, from their feature
        rows `x`, winners (-1 for none) and players to move: the whole-turn
        planner's Rust backend scores a plan's ends with it, in one call.
        A batch's products can differ from a single row's in the last bit,
        far below the rounding to whole points."""
        p = self.model.predict(x)
        score = np.round(SPAN * (2 * p - 1)).astype(np.int64)
        score = np.where(current == 1 - perspective, score, -score)
        won = np.where(winner == perspective, WIN, -WIN)
        return np.where(winner < 0, score, won)

    def describe(self) -> str:
        return f"learned value ({self.model.form})"
