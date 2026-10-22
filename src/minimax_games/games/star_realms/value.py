"""Learned values: models of how likely the player who just ended their turn
is to win, fitted to game results (docs/star-realms.md, Learned values).

Two forms, hand-written in numpy: `LOGISTIC`, logistic regression on the
features (`features.encode`), and `Mlp`, a network with one hidden layer of
tanh units. Each gives its loss, the mean log-loss plus `l2 / 2` times the
sum of its squared weights (not biases), with the loss's gradient worked out
by hand (the tests check it against finite differences). `fit` trains one
with Adam on shuffled minibatches, deterministic given its seed.

A `Model` is a form's parameters with the features' names and the mean and
spread it standardises them by. It may hold several members, fits of the
form from different seeds to the same rows (`fit_ensemble`), and its win
probability is then the mean of theirs: an ensemble, steadier than any one
fit. A model may also read `features.EXTRA`, hand-made features it
computes from the counts of each row (`features.extra`) and appends before
standardising (#173), and `features.MORE` after them (#178). It is saved
as an `.npz` of plain arrays, never a pickle (which runs code when
loaded), and models in the package's `models/` directory load by name
(`load`). `ValueEvaluator` wraps one as an `evaluate.Evaluator`.
"""

from collections.abc import Sequence
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
    """A fitted value: `form` over standardised features (`(x - mean) /
    std`), the features named `names`, with one set of parameters per
    member. Its win probability is the mean of its members' (an ensemble of
    fits from different seeds, `fit_ensemble`); a model of one member is a
    single fit. With `extra` (`features.EXTRA`), the rows it is given are
    widened by their extra features (`features.extra`) before standardising,
    so `mean` and `std` cover both."""

    form: str
    names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    members: tuple[Params, ...]
    extra: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.members:
            msg = "a model has at least 1 member"
            raise ValueError(msg)
        _check_extra(self.extra)

    @property
    def params(self) -> Params:
        """The parameters of a model of one member."""
        if len(self.members) != 1:
            msg = f"a model of {len(self.members)} members has no single params"
            raise ValueError(msg)
        return self.members[0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """The win probability for each row of raw features `x`: the mean of
        the members', summed in member order (as the Rust forward pass sums
        them)."""
        form = FORMS[self.form]
        z = (_widen(x, self.extra) - self.mean) / self.std
        if len(self.members) == 1:
            return form.predict(self.members[0], z)
        total = form.predict(self.members[0], z)
        for params in self.members[1:]:
            total = total + form.predict(params, z)
        return total / len(self.members)

    def save(self, path: Path) -> None:
        """Write the model to `path` as an `.npz` of plain arrays: a model
        of one member as `param_<name>`, as before ensembles, and one of
        several as `member_<i>_<name>`, with their count."""
        if len(self.members) == 1:
            params = {f"param_{n}": a for n, a in self.members[0].items()}
        else:
            params = {
                f"member_{i}_{n}": a
                for i, member in enumerate(self.members)
                for n, a in member.items()
            }
            params["members"] = np.array(len(self.members))
        arrays: dict[str, Any] = {
            "form": np.array(self.form),
            "names": np.array(self.names),
            "mean": self.mean,
            "std": self.std,
            **params,
        }
        if self.extra:
            arrays["extra"] = np.array(self.extra)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "Model":
        """The model saved at `path`. Never unpickles anything."""
        with np.load(path, allow_pickle=False) as data:
            if "members" in data.files:
                members = tuple(
                    {
                        key.removeprefix(f"member_{i}_"): data[key]
                        for key in data.files
                        if key.startswith(f"member_{i}_")
                    }
                    for i in range(int(data["members"]))
                )
            else:
                members = (
                    {
                        key.removeprefix("param_"): data[key]
                        for key in data.files
                        if key.startswith("param_")
                    },
                )
            return cls(
                form=str(data["form"]),
                names=tuple(str(name) for name in data["names"]),
                mean=data["mean"],
                std=data["std"],
                members=members,
                extra=(
                    tuple(str(n) for n in data["extra"])
                    if "extra" in data.files
                    else ()
                ),
            )


def ensemble(models: Sequence[Model]) -> Model:
    """One model whose members are `models`' members, in order: fits of the
    same form to the same features and standardisation (fits to the same
    rows, from different seeds)."""
    if not models:
        msg = "an ensemble needs at least 1 member"
        raise ValueError(msg)
    first = models[0]
    for model in models[1:]:
        if model.form != first.form:
            msg = f"members of one form only: {first.form} and {model.form}"
            raise ValueError(msg)
        if (
            model.names != first.names
            or not np.array_equal(model.mean, first.mean)
            or not np.array_equal(model.std, first.std)
            or model.extra != first.extra
        ):
            msg = "members share their features and standardisation"
            raise ValueError(msg)
    members = tuple(params for model in models for params in model.members)
    return Model(first.form, first.names, first.mean, first.std, members, first.extra)


EXTRAS: tuple[tuple[str, ...], ...] = (
    (),
    features.EXTRA,
    features.EXTRA + features.MORE,
)
"""The extra features a model may read: none, `features.EXTRA` (#173), or
those and `features.MORE` after them (#178)."""


def _check_extra(extra: tuple[str, ...]) -> None:
    if extra not in EXTRAS:
        msg = (
            "the extra features are features.EXTRA, those and features.MORE,"
            f" or none, not {extra}"
        )
        raise ValueError(msg)


def reads_more(extra: tuple[str, ...]) -> bool:
    """Whether a model of extra features `extra` reads `features.MORE`."""
    return extra == EXTRAS[2]


def _widen(x: np.ndarray, extra: tuple[str, ...]) -> np.ndarray:
    """Rows `x` with their extra features appended, in `x`'s precision."""
    if not extra:
        return x
    widened = features.extra(x, more=reads_more(extra))
    return np.hstack([x, widened.astype(x.dtype)])


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
    extra: tuple[str, ...] = (),
) -> Model:
    """`form` fitted to rows `x` and 0/1 labels `y` by Adam on minibatches.

    The features are standardised by their mean and spread over `x` (a
    feature that never varies is left unscaled), in `x`'s own precision:
    float32 halves a big dataset's memory. The starting parameters and each
    epoch's shuffle come from `seed`, so a fit is deterministic. With
    `extra` (`features.EXTRA`, or `features.EXTRA + features.MORE`), the
    model reads the rows' extra features too, standardised alongside them."""
    _check_extra(extra)
    names_of_x = names
    x = _widen(x, extra)
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
    if names_of_x is None:
        names_of_x = tuple(f"x{i}" for i in range(x.shape[1] - len(extra)))
    return Model(form.name, names_of_x, mean, std, (params,), extra)


def fit_ensemble(
    form: Form,
    x: np.ndarray,
    y: np.ndarray,
    *,
    seeds: Sequence[int],
    epochs: int = EPOCHS,
    batch: int = BATCH,
    learning_rate: float = LEARNING_RATE,
    l2: float = L2,
    names: tuple[str, ...] | None = None,
    extra: tuple[str, ...] = (),
) -> Model:
    """An ensemble of `form` fitted to `x` and `y` once per seed (`fit`),
    the members in `seeds`' order. One seed is `fit` itself. The fits differ
    only in their starting parameters and shuffles, so averaging them
    steadies the value against the luck of one fit."""
    return ensemble(
        [
            fit(
                form,
                x,
                y,
                seed=seed,
                epochs=epochs,
                batch=batch,
                learning_rate=learning_rate,
                l2=l2,
                names=names,
                extra=extra,
            )
            for seed in seeds
        ]
    )


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
