"""TD(λ) targets against Monte Carlo targets, offline (#165).

One fixed dataset of pi03's games under policy iteration's mix (half pi03
against itself, a quarter against the heuristic, a quarter against a
champion from {mlp, pi03}): 4,000 training games from seed 6,000,000 and
1,000 held out from 6,100,000. On it, a 4-member mlp ensemble
(`value.fit_ensemble`, seeds 0-3, `value.fit` defaults) per target: TD(λ)
bootstrapping from pi03's value, for a few λ, and Monte Carlo (λ = 1). Each
candidate's planner then plays pi03's over the same 300 seeds from
6,200,000, seats alternating, in Rust.

    uv run python docs/td-targets/td.py runs/td-165 [match games [training games]]
"""

import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from minimax_games.games.star_realms import features, iterate, selfplay, td, value

out = Path(sys.argv[1])
games = int(sys.argv[2]) if len(sys.argv) > 2 else 300
train_games = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
THREADS = 8
TRAIN = range(6_000_000, 6_000_000 + train_games)
HELD = range(6_100_000, 6_100_000 + train_games // 4)
MATCH = range(6_200_000, 6_200_000 + games)
LAMBDAS = (0.0, 0.5, 0.8, 0.95, 1.0)
POOL = ("mlp", "pi03")
out.mkdir(parents=True, exist_ok=True)

models = {name: value.load(name) for name in POOL}
planners: dict[str, selfplay.Player] = {
    name: selfplay.Planner(m) for name, m in models.items()
}
planners["heuristic"] = "heuristic"
pi03 = models["pi03"]

data = out / "dataset.npz"
if not data.is_file():
    named = iterate.mix("pi03", POOL, TRAIN, random.Random(TRAIN.start))
    named += iterate.mix("pi03", POOL, HELD, random.Random(HELD.start))
    t = time.perf_counter()
    played = selfplay.play(
        [(s, planners[a], planners[b]) for s, a, b in named], threads=THREADS
    )
    seconds = time.perf_counter() - t
    iterate._write_games(out / "dataset-games.csv", named, played.winners, played.turns)
    train = played.game < len(TRAIN)
    np.savez(
        data,
        allow_pickle=False,
        x=played.x,
        y=played.y,
        game=played.game,
        position=played.position,
        train=train,
    )
    record = {
        "train_seeds": [TRAIN.start, TRAIN.stop - 1],
        "held_out_seeds": [HELD.start, HELD.stop - 1],
        "pool": POOL,
        "games": len(named),
        "unfinished": sum(w is None for w in played.winners),
        "train_states": int(train.sum()),
        "held_out_states": int((~train).sum()),
        "seconds": round(seconds, 1),
    }
    (out / "dataset.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"dataset: {record}", flush=True)

with np.load(data, allow_pickle=False) as found:
    x, y, game, position, train = (
        found[k] for k in ("x", "y", "game", "position", "train")
    )
held = ~train
current = pi03.predict(x)
print(
    f"pi03 held-out log-loss {value.log_loss(current[held], y[held]):.4f}", flush=True
)

champion = planners["pi03"]
rows = []
for lam in LAMBDAS:
    name = f"td{lam:g}" if lam < 1 else "mc"
    target = td.targets(y[train], game[train], position[train], current[train], lam)
    t = time.perf_counter()
    model = value.fit_ensemble(
        value.Mlp(), x[train], target, seeds=range(4), names=features.NAMES
    )
    fit_seconds = time.perf_counter() - t
    model.save(out / f"{name}.npz")
    loss = value.log_loss(model.predict(x[held]), y[held])
    named = iterate._alternating(name, "pi03", MATCH)
    players = {name: selfplay.Planner(model), "pi03": champion}
    t = time.perf_counter()
    played = selfplay.play(
        [(s, players[a], players[b]) for s, a, b in named], threads=THREADS
    )
    seconds = time.perf_counter() - t
    iterate._write_games(out / f"{name}.csv", named, played.winners, played.turns)
    outcomes = iterate._outcomes(named, played, name)
    wins = sum(o == 1 for o in outcomes)
    finished = sum(o is not None for o in outcomes)
    first = sum(o == 1 for o in outcomes[::2])
    rows.append(
        (
            name,
            lam,
            round(loss, 4),
            wins,
            finished,
            first,
            round(fit_seconds, 1),
            round(seconds, 1),
        )
    )
    print(
        f"{name}: held-out log-loss {loss:.4f}, {wins}/{finished}"
        f" ({100 * wins / finished:.1f}%), as 1st {first}/{len(outcomes[::2])},"
        f" fit {fit_seconds:.0f} s, match {seconds:.0f} s",
        flush=True,
    )
with (out / "summary.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "candidate",
            "lambda",
            "log_loss",
            "wins",
            "finished",
            "wins_as_first",
            "fit_seconds",
            "match_seconds",
        ]
    )
    writer.writerows(rows)
