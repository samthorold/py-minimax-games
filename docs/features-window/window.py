"""The hand-made features at pi03's data size, offline (#175).

#173's capacity test (`docs/capacity/capacity.py`) at the window's size:
the training states of run 2's saved rounds 1-4 (`runs/run-2/data/`, pi03's
games under policy iteration's mix, 369,419 states, the size pi03's window
was), held out on those rounds' held-out states (92,354). Each candidate is
a 4-member mlp ensemble (`value.fit_ensemble`, seeds 0-3, the `value.fit`
defaults: width 32, 10 epochs, batch 256, Adam at 0.001, `l2` 0.003) fitted
to game results (Monte Carlo targets):

- `b32`: today's form, the baseline (the fit run 2's round 4 made);
- `f32`: the same with `features.EXTRA`.

Each one's planner plays pi03's (4 worlds, 20,000 states per plan, both in
Rust) on the same 300 seeds from 8,000,000, seats alternating. If `f32`
clearly beats pi03 there (Wilson interval above 50%, or over 55% won), the
gate is `iterate`'s SPRT (H0 50% against H1 55%, alpha = beta = 0.05, at
most 1,000 games, undecided failing) from 8,100,000, played in batches of
128; if it accepts H1, the checks: 500 seeds from 8,200,000 against pi03,
and seeds 0-499 against the heuristic.

    uv run python docs/features-window/window.py runs/features-175 runs/run-2/data

A candidate already recorded in the output directory is not refitted or
replayed; `summary.csv` is rebuilt from the records.
"""

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from minimax_games.games.star_realms import features, iterate, selfplay, value
from minimax_games.games.star_realms.sprt import Sprt, wilson

out = Path(sys.argv[1])
data = Path(sys.argv[2])
ROUNDS = range(1, 5)
CANDIDATES: dict[str, tuple[str, ...]] = {"b32": (), "f32": features.EXTRA}
BASELINE, FEATURES = "b32", "f32"
THREADS = 8
MATCH = range(8_000_000, 8_000_300)
GATE = range(8_100_000, 8_101_000)
CONFIRM = range(8_200_000, 8_200_500)
LADDER = range(500)
BATCH = 128
TEST = Sprt(0.5, 0.55, 0.05, 0.05)
out.mkdir(parents=True, exist_ok=True)

parts: dict[str, list[np.ndarray]] = {
    k: [] for k in ("x_train", "y_train", "x_held", "y_held")
}
for r in ROUNDS:
    with np.load(data / f"round-{r:02d}.npz", allow_pickle=False) as found:
        for key, rows in parts.items():
            rows.append(found[key])
x_train, y_train, x_held, y_held = (np.concatenate(parts[k]) for k in parts)
del parts
print(
    f"rounds {ROUNDS.start}-{ROUNDS.stop - 1}: {len(y_train)} training states,"
    f" {len(y_held)} held out",
    flush=True,
)
pi03 = value.load("pi03")
champion = selfplay.Planner(pi03)
pi03_losses = {
    "train_log_loss": round(value.log_loss(pi03.predict(x_train), y_train), 4),
    "held_out_log_loss": round(value.log_loss(pi03.predict(x_held), y_held), 4),
}
print(f"pi03 log-loss: {pi03_losses}", flush=True)


def match(
    name: str, players: dict[str, selfplay.Player], seeds: range, csv_name: str
) -> tuple[list[int | None], float]:
    """`name` against the other player over `seeds`, seats alternating; its
    outcomes (1 won, 0 lost, None unfinished) and the seconds it took."""
    other = next(p for p in players if p != name)
    named = iterate._alternating(name, other, seeds)
    t = time.perf_counter()
    played = selfplay.play(
        [(s, players[a], players[b]) for s, a, b in named], threads=THREADS
    )
    seconds = time.perf_counter() - t
    iterate._write_games(out / csv_name, named, played.winners, played.turns)
    return iterate._outcomes(named, played, name), seconds


def tally(outcomes: list[int | None], seconds: float) -> dict[str, object]:
    wins = sum(o == 1 for o in outcomes)
    finished = sum(o is not None for o in outcomes)
    low, high = wilson(wins, finished)
    return {
        "games": len(outcomes),
        "wins": wins,
        "finished": finished,
        "win_pct": round(100 * wins / finished, 1),
        "wilson_low": round(100 * low, 1),
        "wilson_high": round(100 * high, 1),
        "wins_as_first": sum(o == 1 for o in outcomes[::2]),
        "seconds": round(seconds, 1),
    }


models: dict[str, value.Model] = {}
for name, extra in CANDIDATES.items():
    record_path = out / f"{name}.json"
    if record_path.is_file():
        models[name] = value.Model.load(out / f"{name}.npz")
        continue
    t = time.perf_counter()
    model = value.fit_ensemble(
        value.Mlp(hidden=32),
        x_train,
        y_train,
        seeds=range(4),
        names=features.NAMES,
        extra=extra,
    )
    fit_seconds = time.perf_counter() - t
    model.save(out / f"{name}.npz")
    models[name] = model
    outcomes, seconds = match(
        name, {name: selfplay.Planner(model), "pi03": champion}, MATCH, f"{name}.csv"
    )
    record = {
        "candidate": name,
        "extra": len(extra),
        "train_log_loss": round(value.log_loss(model.predict(x_train), y_train), 4),
        "held_out_log_loss": round(value.log_loss(model.predict(x_held), y_held), 4),
        "fit_seconds": round(fit_seconds, 1),
        "match": tally(outcomes, seconds),
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(record, flush=True)
del x_train, y_train, x_held, y_held


def wins_by_seed(name: str) -> dict[int, bool]:
    with (out / f"{name}.csv").open() as f:
        return {int(r["seed"]): r["winner"] == name for r in csv.DictReader(f)}


records = {n: json.loads((out / f"{n}.json").read_text()) for n in CANDIDATES}
base, feat = wins_by_seed(BASELINE), wins_by_seed(FEATURES)
only_f = sum(feat[s] and not base[s] for s in feat)
only_b = sum(base[s] and not feat[s] for s in feat)
z = (only_f - only_b) / math.sqrt(only_f + only_b) if only_f + only_b else 0.0
paired = {"only_features": only_f, "only_baseline": only_b, "sign_z": round(z, 2)}
print(f"paired {FEATURES} vs {BASELINE}: {paired}", flush=True)
with (out / "summary.csv").open("w", newline="") as f:
    rows = [
        {
            "candidate": "pi03",
            "extra": 0,
            **pi03_losses,
        }
    ] + [
        {k: v for k, v in {**r, **r["match"]}.items() if k not in ("match", "seconds")}
        | {"match_seconds": r["match"]["seconds"]}
        for r in records.values()
    ]
    writer = csv.DictWriter(f, fieldnames=list(rows[-1]))
    writer.writeheader()
    writer.writerows(rows)

result = records[FEATURES]["match"]
clearly = result["wilson_low"] > 50 or result["win_pct"] > 55
summary: dict[str, object] = {
    "pi03": pi03_losses,
    "paired": paired,
    "clearly_beats_pi03": clearly,
}
print(f"{FEATURES} clearly beats pi03: {clearly}", flush=True)

if clearly:
    players: dict[str, selfplay.Player] = {
        FEATURES: selfplay.Planner(models[FEATURES]),
        "pi03": champion,
        "heuristic": "heuristic",
    }
    named = iterate._alternating(FEATURES, "pi03", GATE)
    winners: list[int | None] = []
    turns: list[int] = []
    outcomes: list[int | None] = []
    t = time.perf_counter()
    while True:
        batch = named[len(outcomes) : len(outcomes) + BATCH]
        played = selfplay.play(
            [(s, players[a], players[b]) for s, a, b in batch], threads=THREADS
        )
        winners += played.winners
        turns += played.turns
        outcomes += iterate._outcomes(batch, played, FEATURES)
        verdict, count = iterate.decide(outcomes, TEST, len(GATE))
        if verdict is not None or count >= len(GATE):
            break
    iterate._write_games(
        out / "gate.csv", named[:count], winners[:count], turns[:count]
    )
    wins = sum(o == 1 for o in outcomes[:count])
    losses = sum(o == 0 for o in outcomes[:count])
    summary["gate"] = {
        "decision": "undecided" if verdict is None else verdict.value,
        "games": count,
        "wins": wins,
        "losses": losses,
        "llr": round(TEST.llr(wins, losses), 2),
        "seconds": round(time.perf_counter() - t, 1),
    }
    print(f"gate: {summary['gate']}", flush=True)
    if iterate.promotes(verdict):
        confirm, seconds = match(
            FEATURES,
            {FEATURES: players[FEATURES], "pi03": champion},
            CONFIRM,
            f"confirm-{FEATURES}-pi03.csv",
        )
        summary["confirm"] = tally(confirm, seconds)
        print(f"confirmation: {summary['confirm']}", flush=True)
        ladder, seconds = match(
            FEATURES,
            {FEATURES: players[FEATURES], "heuristic": "heuristic"},
            LADDER,
            f"ladder-{FEATURES}.csv",
        )
        summary["ladder"] = tally(ladder, seconds)
        print(f"ladder: {summary['ladder']}", flush=True)

(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
