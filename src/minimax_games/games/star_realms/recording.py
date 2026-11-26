"""One recording of the champion's games, read by every instrument that
needs them (docs/star-realms.md, One recording of the champion's games).

Almost all of an instrument's time is in playing games, not in reading them,
and games are deterministic given the players, seeds and settings, so the
champion's games on a seed range are played once (`record`), with
everything any instrument reads (every decision, each end's judgement, and
what the diagnostics need: `selfplay.play`'s `record`, `judge` and
`diagnose`), saved (`save`), and each instrument reads the recording
(`read`) instead of playing its own. A recording names its champion, seeds,
mix and settings (`Settings`), and the digest of every model that played
(`digest`), so it is valid until the champion changes: `read` refuses one
whose settings don't match what the instrument was asked for, or whose
models have changed since.

Two mixes: `"self-play"`, the champion against itself with seats
alternating (`selfplay.alternating`), as the probes play; and
`"policy-iteration"`, as a policy-iteration round mixes them
(`iterate.mix`, with `Settings.pool`), as the exploitation gap and the
diagnostics play.
"""

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from minimax_games.games.star_realms import value
from minimax_games.games.star_realms.fit import MAX_TURNS
from minimax_games.games.star_realms.iterate import HEURISTIC, mix
from minimax_games.games.star_realms.planner import BUDGET, WORLDS
from minimax_games.games.star_realms.selfplay import (
    Examined,
    Pairing,
    Planner,
    Player,
    SelfPlay,
    play,
)

type Mix = Literal["self-play", "policy-iteration"]
MIXES: tuple[Mix, ...] = ("self-play", "policy-iteration")


@dataclass(frozen=True, slots=True)
class Settings:
    """What a recording's games are: `champion` (a committed model's name
    or a saved model's path, `.npz`) on `seeds`, in `mix` (with `pool`'s
    champions for the policy-iteration mix), the planners at `worlds`
    worlds and `budget` states per plan, each game stopped unfinished past
    `max_turns`."""

    champion: str
    seeds: range
    mix: Mix = "self-play"
    pool: tuple[str, ...] = ()
    worlds: int = WORLDS
    budget: int = BUDGET
    max_turns: int = MAX_TURNS

    def __post_init__(self) -> None:
        if self.mix not in MIXES:
            msg = f"no mix {self.mix!r}: {', '.join(MIXES)}"
            raise ValueError(msg)
        if self.mix == "self-play" and self.pool:
            msg = "self-play has no pool"
            raise ValueError(msg)
        if self.mix == "policy-iteration" and not self.pool:
            msg = "the policy-iteration mix needs a pool"
            raise ValueError(msg)


def model(name: str) -> value.Model:
    """The model `name`: a saved model's path (`.npz`) or a committed one."""
    return value.Model.load(Path(name)) if name.endswith(".npz") else value.load(name)


def digest(found: value.Model) -> str:
    """A digest of everything that decides how `found` judges: two models
    with the same digest play the same games."""
    h = hashlib.sha256()
    h.update(json.dumps([found.form, found.names, found.extra]).encode())
    for a in (found.mean, found.std):
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    for params in found.members:
        for key in sorted(params):
            h.update(key.encode())
            h.update(np.ascontiguousarray(params[key], dtype=np.float64).tobytes())
    return h.hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class Recording:
    """The games `settings` names, as played."""

    settings: Settings
    digests: dict[str, str]
    """Each model that played, by name, and its `digest`."""
    names: tuple[tuple[str, str], ...]
    """Each game's players by name, seat 0 first (`"heuristic"` for the
    heuristic)."""
    played: SelfPlay
    """The games, with every decision, each end's judgement and what the
    diagnostics need."""

    def players(self) -> dict[str, Player]:
        """Each player by name (`_players`)."""
        return _players(self.settings, {name: model(name) for name in self.digests})

    def pairings(self, players: dict[str, Player] | None = None) -> list[Pairing]:
        """The games as `selfplay.play` takes them, by `players` (by
        default, `players()`)."""
        return _pairings(self.settings, self.names, players or self.players())


def _players(settings: Settings, models: dict[str, value.Model]) -> dict[str, Player]:
    """A planner at the settings' worlds and budget for each model, by
    name, and the heuristic."""
    players: dict[str, Player] = {
        name: Planner(m, settings.worlds, settings.budget) for name, m in models.items()
    }
    players[HEURISTIC] = HEURISTIC
    return players


def _pairings(
    settings: Settings,
    names: tuple[tuple[str, str], ...],
    players: dict[str, Player],
) -> list[Pairing]:
    """The games `names` names, one player per name."""
    return [
        (seed, players[a], players[b])
        for seed, (a, b) in zip(settings.seeds, names, strict=True)
    ]


def thinned(played: SelfPlay, every: int) -> SelfPlay:
    """Every `every`th game of `played` (the 1st, then the `every + 1`th,
    ...), as `selfplay.play` returns those games played alone: their rows
    and plans, each game numbered by its place among them."""
    if every == 1:
        return played
    kept = range(0, len(played.winners), every)
    row = played.game % every == 0

    def rows(a: np.ndarray | None) -> np.ndarray | None:
        return None if a is None else a[row]

    examined = played.examined
    if examined is not None:
        plan = examined.game % every == 0
        examined = Examined(
            game=examined.game[plan] // every,
            turn=examined.turn[plan],
            seat=examined.seat[plan],
            complete=examined.complete[plan],
        )
    actions = played.actions
    return SelfPlay(
        x=played.x[row],
        y=played.y[row],
        game=played.game[row] // every,
        position=played.position[row],
        seat=played.seat[row],
        winners=tuple(played.winners[g] for g in kept),
        turns=tuple(played.turns[g] for g in kept),
        plans=played.plans[::every],
        diverged=played.diverged[::every],
        actions=None if actions is None else [actions[g] for g in kept],
        judgement=rows(played.judgement),
        left=rows(played.left),
        examined=examined,
    )


def _names(settings: Settings) -> tuple[tuple[str, str], ...]:
    seeds = settings.seeds
    if settings.mix == "self-play":
        return tuple((settings.champion, settings.champion) for _ in seeds)
    named = mix(
        settings.champion, list(settings.pool), seeds, random.Random(seeds.start)
    )
    return tuple((a, b) for _, a, b in named)


def record(settings: Settings, threads: int = 1) -> Recording:
    """Play the games `settings` names on `threads` threads, recording
    everything any instrument reads."""
    names = _names(settings)
    models = {
        name: model(name)
        for name in sorted({settings.champion, *settings.pool} - {HEURISTIC})
    }
    played = play(
        _pairings(settings, names, _players(settings, models)),
        threads=threads,
        max_turns=settings.max_turns,
        record=True,
        judge=True,
        diagnose=True,
    )
    digests = {name: digest(m) for name, m in models.items()}
    return Recording(settings, digests, names, played)


_ROWS = ("x", "y", "game", "position", "seat", "judgement", "left")
_GAMES = ("plans", "diverged")


def save(found: Recording, path: Path) -> None:
    """Write `found` to `path` (an `.npz` file; its directory is made)."""
    s, p = found.settings, found.played
    if p.examined is None or p.actions is None:
        msg = "a recording keeps every decision and what the diagnostics need"
        raise ValueError(msg)
    meta = {
        "champion": s.champion,
        "seeds": [s.seeds.start, s.seeds.stop],
        "mix": s.mix,
        "pool": list(s.pool),
        "worlds": s.worlds,
        "budget": s.budget,
        "max_turns": s.max_turns,
        "digests": found.digests,
        "names": [list(n) for n in found.names],
        "winners": [-1 if w is None else w for w in p.winners],
        "turns": list(p.turns),
    }
    arrays = {name: getattr(p, name) for name in (*_ROWS, *_GAMES)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        meta=np.array(json.dumps(meta)),
        actions=np.array(json.dumps(p.actions)),
        plan_game=p.examined.game,
        plan_turn=p.examined.turn,
        plan_seat=p.examined.seat,
        plan_complete=p.examined.complete,
        **arrays,
    )


def _action(found: list) -> tuple:
    return tuple(_action(a) if isinstance(a, list) else a for a in found)


def load(path: Path) -> Recording:
    """The recording `save` wrote to `path`."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        actions = json.loads(str(data["actions"]))
        arrays = {name: data[name] for name in data.files}
    settings = Settings(
        champion=meta["champion"],
        seeds=range(*meta["seeds"]),
        mix=meta["mix"],
        pool=tuple(meta["pool"]),
        worlds=meta["worlds"],
        budget=meta["budget"],
        max_turns=meta["max_turns"],
    )
    played = SelfPlay(
        **{name: arrays[name] for name in (*_ROWS, *_GAMES)},
        winners=tuple(None if w < 0 else w for w in meta["winners"]),
        turns=tuple(meta["turns"]),
        actions=[[_action(a) for a in game] for game in actions],
        examined=Examined(
            game=arrays["plan_game"],
            turn=arrays["plan_turn"],
            seat=arrays["plan_seat"],
            complete=arrays["plan_complete"],
        ),
    )
    return Recording(
        settings, meta["digests"], tuple(tuple(n) for n in meta["names"]), played
    )


class Mismatch(ValueError):
    """A recording isn't of the games an instrument was asked for."""


def read(path: Path, settings: Settings) -> Recording:
    """The recording at `path`, if it is of the games `settings` names,
    played by the models as they are now; `Mismatch` otherwise, naming
    what differs."""
    found = load(path)
    differ = [
        f"{field} {getattr(found.settings, field)!r}, asked for"
        f" {getattr(settings, field)!r}"
        for field in Settings.__dataclass_fields__
        if getattr(found.settings, field) != getattr(settings, field)
    ]
    differ += [
        f"model {name} has changed since it was recorded"
        for name, recorded in found.digests.items()
        if digest(model(name)) != recorded
    ]
    if differ:
        msg = f"{path} isn't a recording of the games asked for: " + "; ".join(differ)
        raise Mismatch(msg)
    return found
