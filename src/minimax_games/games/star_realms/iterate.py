"""Policy iteration for the learned value (docs/star-realms.md, Learned
values: policy iteration), unattended and resumable.

Each round:

1. **Plays** `train_games` training games and `held_out_games` held-out
   games in Rust (`selfplay.play`), each set mixed as `mix` mixes it: half
   the current best's planner against itself, a quarter against the
   heuristic, a quarter against an earlier champion chosen uniformly from
   the pool (which holds champions only, the current best included), seats
   alternating.
2. **Fits** both forms from scratch (`fit_candidates`) to the training games
   of the last `window` rounds, and keeps the one with the lower log-loss on
   this round's held-out games: the candidate.
3. **Gates** it offline (`pregate`): its held-out log-loss must be below the
   current best's on the same states. Then in a match (`decide`), the
   candidate's planner against the current best's, same settings, played in
   Rust in batches and stopped by SPRT at the game that decides it, H0 50%
   against H1 55%, at most `gate_games` games; undecided at the limit fails.
4. On promotion, the new champion joins the pool and plays a fixed
   `ladder_games` games against the heuristic on seeds 0-499: recorded, not
   gating. A failed candidate is dropped; its round's games stay in the
   window.

It stops after `patience` rounds in a row without a promotion, after
`max_rounds` rounds, or once the rounds have taken `hours` (`stop_reason`).

**Seeds** come from disjoint ranges (`Settings`): ladder games 0-499,
round `r`'s training games from `TRAIN_BASE + STRIDE * (r - 1)`, its
held-out games from `HELD_OUT_BASE + ...` and its gating games from
`GATE_BASE + ...`, so no value is judged on games it was trained on.

**Outputs**, in the run's directory, are written as they complete:
`state.json` (the settings and every finished round's record, rewritten
after each round), `data/round-NN.npz` (a round's rows, so an interrupted
round doesn't replay its games; never committed), `candidates/` (both
forms' fits per round), `models/<name>.npz` (each promoted model), and
per-seed results of every gating and ladder match, `gate-rNN.csv` and
`ladder-rNN.csv` (seed, first, second, winner, turns), for later bots to be
paired against. Running again with the same directory resumes after the
last finished round; the settings that change games must be the same.
"""

import csv
import json
import os
import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from minimax_games.games.star_realms import features, selfplay, value
from minimax_games.games.star_realms.fit import MAX_TURNS
from minimax_games.games.star_realms.planner import BUDGET, WORLDS
from minimax_games.games.star_realms.sprt import Hypothesis, Sprt
from minimax_games.games.star_realms.value import (
    BATCH,
    EPOCHS,
    HIDDEN,
    L2,
    LEARNING_RATE,
    LOGISTIC,
    Mlp,
    Model,
    log_loss,
)

HEURISTIC = "heuristic"

STRIDE = 10_000
"""Seeds set aside per round for each kind of game."""
MAX_ROUNDS = 100
"""The most rounds the seed ranges have room for."""
TRAIN_BASE = 1_000_000
HELD_OUT_BASE = 2_000_000
GATE_BASE = 3_000_000
"""Round `r` takes its training, held-out and gating seeds from these plus
`STRIDE * (r - 1)`; the ladder's are 0 up, below them all."""

KEPT = ("threads", "hours", "max_rounds", "patience")
"""Settings that may change when a run is resumed: they don't change any
game, fit or decision, only when the run stops and how fast it goes."""


@dataclass(frozen=True, slots=True)
class Settings:
    """A run's settings: the defaults are the agreed design's."""

    train_games: int = 4000
    """Training games per round; with the held-out ones, 5,000."""
    held_out_games: int = 1000
    window: int = 4
    """Rounds of training games each fit sees."""
    gate_games: int = 1000
    """The match's limit."""
    gate_batch: int = 128
    """Games played per batch of the match, ahead of the decision."""
    ladder_games: int = 500
    h0: float = 0.5
    h1: float = 0.55
    alpha: float = 0.05
    beta: float = 0.05
    max_rounds: int = 10
    patience: int = 3
    """Stop after this many rounds in a row without a promotion."""
    hours: float = 2.0
    """Start no round once the rounds so far have taken this long."""
    worlds: int = WORLDS
    budget: int = BUDGET
    max_turns: int = MAX_TURNS
    fit_seed: int = 0
    epochs: int = EPOCHS
    batch: int = BATCH
    learning_rate: float = LEARNING_RATE
    l2: float = L2
    hidden: int = HIDDEN
    start: str = "mlp"
    """The starting champion, a packaged model."""
    prefix: str = "pi"
    """Round `r`'s candidate is named `<prefix><r:02d>`."""
    threads: int = 8

    def __post_init__(self) -> None:
        for name in ("train_games", "held_out_games", "gate_games"):
            if not 0 < getattr(self, name) <= STRIDE:
                msg = f"{name} must be 1 to {STRIDE}, the seeds a round has room for"
                raise ValueError(msg)
        if not 0 < self.ladder_games <= TRAIN_BASE:
            msg = f"ladder_games must be 1 to {TRAIN_BASE}, below the rounds' seeds"
            raise ValueError(msg)
        if not 0 < self.max_rounds <= MAX_ROUNDS:
            msg = f"max_rounds must be 1 to {MAX_ROUNDS}: the seed ranges' rounds"
            raise ValueError(msg)
        if self.window < 1 or self.patience < 1 or self.gate_batch < 1:
            msg = "window, patience and gate_batch must be at least 1"
            raise ValueError(msg)

    def train_seeds(self, round: int) -> range:
        return _seeds(TRAIN_BASE, round, self.train_games)

    def held_out_seeds(self, round: int) -> range:
        return _seeds(HELD_OUT_BASE, round, self.held_out_games)

    def gate_seeds(self, round: int) -> range:
        return _seeds(GATE_BASE, round, self.gate_games)

    @property
    def ladder_seeds(self) -> range:
        return range(self.ladder_games)

    @property
    def test(self) -> Sprt:
        return Sprt(self.h0, self.h1, self.alpha, self.beta)


def _seeds(base: int, round: int, games: int) -> range:
    start = base + STRIDE * (round - 1)
    return range(start, start + games)


type Named = tuple[int, str, str]
"""A game by its players' names: its seed, then seats 0 and 1."""


def mix(
    best: str, pool: Sequence[str], seeds: Iterable[int], rng: random.Random
) -> list[Named]:
    """Games from `seeds`: the first half `best` against itself, the next
    quarter against the heuristic, the rest against a champion chosen
    uniformly from `pool` by `rng` for each game; `best` moves first in
    alternate games against each opponent."""
    seeds = list(seeds)
    selfish = len(seeds) // 2
    heuristic = selfish + len(seeds) // 4
    games = [(seed, best, best) for seed in seeds[:selfish]]
    for opponents, chunk in (
        (None, seeds[selfish:heuristic]),
        (pool, seeds[heuristic:]),
    ):
        for i, seed in enumerate(chunk):
            other = HEURISTIC if opponents is None else rng.choice(opponents)
            games.append((seed, best, other) if i % 2 == 0 else (seed, other, best))
    return games


def window(round: int, size: int) -> list[int]:
    """The rounds whose training games round `round` fits to."""
    return list(range(max(1, round - size + 1), round + 1))


def pregate(candidate: float, best: float) -> bool:
    """The offline gate: the candidate's held-out log-loss is below the
    current best's on the same states."""
    return candidate < best


def decide(
    outcomes: Iterable[int | None], test: Sprt, limit: int
) -> tuple[Hypothesis | None, int]:
    """The SPRT's decision on the candidate's games in seed order (1 won, 0
    lost, None unfinished: neither), and the games it took; `None` and the
    games seen, at most `limit`, if none decided it."""
    wins = losses = played = 0
    for outcome in outcomes:
        if played == limit:
            break
        played += 1
        if outcome is None:
            continue
        if outcome:
            wins += 1
        else:
            losses += 1
        verdict = test.decide(wins, losses)
        if verdict is not None:
            return verdict, played
    return None, played


def promotes(verdict: Hypothesis | None) -> bool:
    """Only an accepted H1 promotes: H0 or undecided at the limit fails."""
    return verdict is Hypothesis.H1


def stop_reason(
    promotions: Sequence[bool], seconds: float, settings: Settings
) -> str | None:
    """Why the run stops before another round, given each finished round's
    promotion and the time they took, or None to play on."""
    if len(promotions) >= settings.max_rounds:
        return f"{settings.max_rounds} rounds played"
    trailing = 0
    for promoted in reversed(promotions):
        if promoted:
            break
        trailing += 1
    if trailing >= settings.patience:
        return f"{settings.patience} rounds in a row without a promotion"
    if seconds >= settings.hours * 3600:
        return f"the time cap ({settings.hours:g} hours)"
    return None


def fit_candidates(
    x: np.ndarray, y: np.ndarray, settings: Settings
) -> dict[str, Model]:
    """Both forms, fitted from scratch to rows `x` and labels `y`."""
    forms = {"logistic": LOGISTIC, "mlp": Mlp(hidden=settings.hidden)}
    return {
        name: value.fit(
            form,
            x,
            y,
            seed=settings.fit_seed,
            epochs=settings.epochs,
            batch=settings.batch,
            learning_rate=settings.learning_rate,
            l2=settings.l2,
            names=features.NAMES,
        )
        for name, form in forms.items()
    }


@dataclass(slots=True)
class Round:
    """One finished round's record, as `state.json` keeps it."""

    round: int
    champion: str
    """The current best when the round began."""
    pool: list[str]
    window: list[int]
    train: dict[str, int]
    """Finished and unfinished training games, and their states."""
    held_out: dict[str, int]
    window_states: int
    losses: dict[str, float]
    """Held-out log-loss of each form's fit and of the champion."""
    candidate: str
    """The form that went forward."""
    name: str
    pregate: bool
    sprt: dict[str, Any] | None
    """The match: decision (H0, H1 or undecided), games, wins, losses and
    log-likelihood ratio; None if the offline gate failed."""
    promoted: str | None
    ladder: dict[str, Any] | None
    seconds: dict[str, float]


def _write_json(path: Path, data: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)


def _write_games(
    path: Path,
    games: Sequence[Named],
    winners: Sequence[int | None],
    turns: Sequence[int],
) -> None:
    """Each game's seed, players, winner (by name, empty if unfinished) and
    turns, as CSV."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "first", "second", "winner", "turns"])
        for (seed, first, second), winner, turn in zip(
            games, winners, turns, strict=True
        ):
            name = "" if winner is None else (first, second)[winner]
            writer.writerow([seed, first, second, name, turn])
    os.replace(temporary, path)


def _comparable(settings: Settings) -> dict[str, Any]:
    return {k: v for k, v in asdict(settings).items() if k not in KEPT}


def _load_state(out: Path, settings: Settings) -> dict[str, Any]:
    path = out / "state.json"
    if not path.is_file():
        state = {"settings": asdict(settings), "rounds": [], "stopped": None}
        _write_json(path, state)
        return state
    state = json.loads(path.read_text())
    stored = {k: v for k, v in state["settings"].items() if k not in KEPT}
    if stored != _comparable(settings):
        differ = sorted(k for k in stored if stored[k] != _comparable(settings).get(k))
        msg = f"{path} was run with other settings: {', '.join(differ)}"
        raise ValueError(msg)
    state["settings"] = asdict(settings)
    return state


def _save_rows(path: Path, rows: dict[str, np.ndarray]) -> None:
    """Write a round's rows to `path` whole or not at all."""
    temporary = path.with_name(path.stem + ".tmp.npz")
    with temporary.open("wb") as f:
        np.savez(f, allow_pickle=False, **rows)
    os.replace(temporary, path)


def _alternating(a: str, b: str, seeds: Iterable[int]) -> list[Named]:
    """`selfplay.alternating`, by name: `a` first in the 1st, 3rd, ..."""
    return [
        (seed, a, b) if i % 2 == 0 else (seed, b, a) for i, seed in enumerate(seeds)
    ]


def _model(name: str, out: Path) -> Model:
    path = out / "models" / f"{name}.npz"
    return Model.load(path) if path.is_file() else value.load(name)


def _outcomes(
    games: Sequence[Named], played: selfplay.SelfPlay, candidate: str
) -> list[int | None]:
    found: list[int | None] = []
    for (_, first, second), winner in zip(games, played.winners, strict=True):
        found.append(
            None if winner is None else int((first, second)[winner] == candidate)
        )
    return found


def run(
    settings: Settings,
    out: Path,
    log: Callable[[str], None] = print,
    clock: Callable[[], float] | None = None,
) -> list[Round]:
    """Run rounds into `out` until a stopping rule holds, resuming after
    the rounds already recorded there; every round's records."""
    clock = clock or time.perf_counter
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("data", "candidates", "models"):
        (out / sub).mkdir(exist_ok=True)
    state = _load_state(out, settings)
    rounds = [Round(**r) for r in state["rounds"]]
    test = settings.test
    while True:
        reason = stop_reason(
            [r.promoted is not None for r in rounds],
            sum(r.seconds["total"] for r in rounds),
            settings,
        )
        if reason is not None:
            state["stopped"] = reason
            _write_json(out / "state.json", state)
            log(f"Stopped: {reason}")
            return rounds
        number = len(rounds) + 1
        champion, pool = settings.start, [settings.start]
        for r in rounds:
            if r.promoted is not None:
                champion = r.promoted
                pool.append(r.promoted)
        rounds.append(_round(settings, out, number, champion, pool, test, log, clock))
        state["rounds"] = [asdict(r) for r in rounds]
        state["stopped"] = None
        _write_json(out / "state.json", state)


def _round(
    settings: Settings,
    out: Path,
    number: int,
    champion: str,
    pool: list[str],
    test: Sprt,
    log: Callable[[str], None],
    clock: Callable[[], float],
) -> Round:
    start = clock()
    seconds: dict[str, float] = {}
    planners: dict[str, selfplay.Player] = {
        name: selfplay.Planner(_model(name, out), settings.worlds, settings.budget)
        for name in pool
    }
    planners[HEURISTIC] = HEURISTIC

    def pairings(games: Sequence[Named]) -> list[selfplay.Pairing]:
        return [(seed, planners[a], planners[b]) for seed, a, b in games]

    # 1. play, unless an interrupted run already did
    data = out / "data" / f"round-{number:02d}.npz"
    reused = data.is_file()
    if reused:
        with np.load(data, allow_pickle=False) as found:
            rows = {key: found[key] for key in found.files}
        log(f"Round {number}: reusing the games in {data}")
    else:
        train_seeds = settings.train_seeds(number)
        held_seeds = settings.held_out_seeds(number)
        games = mix(champion, pool, train_seeds, random.Random(train_seeds.start))
        games += mix(champion, pool, held_seeds, random.Random(held_seeds.start))
        played = selfplay.play(pairings(games), settings.threads, settings.max_turns)
        train = played.game < settings.train_games
        unfinished = [w is None for w in played.winners]
        rows = {
            "x_train": played.x[train],
            "y_train": played.y[train],
            "x_held": played.x[~train],
            "y_held": played.y[~train],
            "train_unfinished": np.array(sum(unfinished[: settings.train_games])),
            "held_unfinished": np.array(sum(unfinished[settings.train_games :])),
            "seconds": np.array(clock() - start),
        }
        _save_rows(data, rows)
    seconds["play"] = float(rows["seconds"])
    train_info = {
        "games": settings.train_games - int(rows["train_unfinished"]),
        "unfinished": int(rows["train_unfinished"]),
        "states": len(rows["y_train"]),
    }
    held_info = {
        "games": settings.held_out_games - int(rows["held_unfinished"]),
        "unfinished": int(rows["held_unfinished"]),
        "states": len(rows["y_held"]),
    }
    log(
        f"Round {number}: champion {champion}, pool {', '.join(pool)};"
        f" {train_info['games']} training games ({train_info['states']} states),"
        f" {held_info['games']} held out ({held_info['states']}),"
        f" {seconds['play']:.0f} s"
    )

    # 2. fit both forms to the window
    began = clock()
    rounds = window(number, settings.window)
    xs, ys = [rows["x_train"]], [rows["y_train"]]
    for earlier in rounds[:-1]:
        with np.load(out / "data" / f"round-{earlier:02d}.npz") as found:
            xs.insert(-1, found["x_train"])
            ys.insert(-1, found["y_train"])
    x, y = np.concatenate(xs), np.concatenate(ys)
    del xs, ys
    models = fit_candidates(x, y, settings)
    window_states = len(y)
    del x, y
    for form, model in models.items():
        model.save(out / "candidates" / f"round-{number:02d}-{form}.npz")
    held_x, held_y = rows["x_held"], rows["y_held"]
    losses = {
        form: log_loss(model.predict(held_x), held_y) for form, model in models.items()
    }
    best = _model(champion, out)
    losses["champion"] = log_loss(best.predict(held_x), held_y)
    form = min(models, key=lambda f: losses[f])
    candidate = models[form]
    name = f"{settings.prefix}{number:02d}"
    passed = pregate(losses[form], losses["champion"])
    seconds["fit"] = clock() - began
    log(
        f"Round {number}: fitted to rounds {rounds[0]}-{rounds[-1]}"
        f" ({window_states} states); held-out log-loss logistic"
        f" {losses['logistic']:.4f}, mlp {losses['mlp']:.4f}, champion"
        f" {losses['champion']:.4f}: {name} ({form})"
        f" {'passes' if passed else 'fails'} the offline gate,"
        f" {seconds['fit']:.0f} s"
    )

    # 3. the match
    began = clock()
    sprt = None
    verdict = None
    if passed:
        planners[name] = selfplay.Planner(candidate, settings.worlds, settings.budget)
        named = _alternating(name, champion, settings.gate_seeds(number))
        # batches played ahead; the decision is the one game by game would take
        winners: list[int | None] = []
        turns: list[int] = []
        outcomes: list[int | None] = []
        while True:
            done = len(outcomes)
            batch = named[done : done + settings.gate_batch]
            played = selfplay.play(
                pairings(batch), settings.threads, settings.max_turns
            )
            winners += played.winners
            turns += played.turns
            outcomes += _outcomes(batch, played, name)
            verdict, count = decide(outcomes, test, settings.gate_games)
            if verdict is not None or count >= settings.gate_games:
                break
        _write_games(
            out / f"gate-r{number:02d}.csv",
            named[:count],
            winners[:count],
            turns[:count],
        )
        wins = sum(o == 1 for o in outcomes[:count])
        losses_ = sum(o == 0 for o in outcomes[:count])
        sprt = {
            "decision": "undecided" if verdict is None else verdict.value,
            "games": count,
            "wins": wins,
            "losses": losses_,
            "llr": test.llr(wins, losses_),
        }
        log(
            f"Round {number}: {name} vs {champion}: {sprt['decision']} after"
            f" {count} games, {wins}-{losses_} (LLR {sprt['llr']:.2f})"
        )
    seconds["gate"] = clock() - began

    # 4. promotion and the ladder
    promoted = None
    ladder = None
    if promotes(verdict):
        promoted = name
        candidate.save(out / "models" / f"{name}.npz")
        began = clock()
        named = _alternating(name, HEURISTIC, settings.ladder_seeds)
        played = selfplay.play(pairings(named), settings.threads, settings.max_turns)
        _write_games(
            out / f"ladder-r{number:02d}.csv", named, played.winners, played.turns
        )
        outcomes = _outcomes(named, played, name)
        wins = sum(o == 1 for o in outcomes)
        finished = sum(o is not None for o in outcomes)
        ladder = {
            "games": len(named),
            "finished": finished,
            "wins": wins,
            "rate": wins / finished if finished else 0.0,
        }
        seconds["ladder"] = clock() - began
        log(
            f"Round {number}: {name} promoted; against the heuristic"
            f" {wins}/{finished} ({100 * ladder['rate']:.1f}%)"
        )
    # an earlier invocation's play time counts, as if it had run on
    seconds["total"] = clock() - start + (seconds["play"] if reused else 0.0)
    return Round(
        round=number,
        champion=champion,
        pool=list(pool),
        window=rounds,
        train=train_info,
        held_out=held_info,
        window_states=window_states,
        losses=losses,
        candidate=form,
        name=name,
        pregate=passed,
        sprt=sprt,
        promoted=promoted,
        ladder=ladder,
        seconds=seconds,
    )
