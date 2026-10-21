"""The learned value's capacity and hand-made features, offline (#173).

On #165's fixed dataset of pi03's games (`runs/td-165/dataset.npz`: 4,000
training games from seed 6,000,000 and 1,000 held out from 6,100,000), each
candidate is a 4-member mlp ensemble (`value.fit_ensemble`, seeds 0-3, the
`value.fit` defaults but for the width) fitted to the training states'
game results (Monte Carlo targets). For each: its training and held-out
log-loss, its planner against pi03's over the same 300 seeds from 7,000,000
(seats alternating, in Rust), and its planner's cost, ms per plan, on one
thread against the heuristic over seeds 600-639 (the ensemble
measurement's games, #164).

Candidates: `w32` (today's form, the baseline), `w64`, `w128`, and `f32`
(width 32 with `features.EXTRA`, computed from the 190 counts).

    uv run python docs/capacity/capacity.py runs/capacity-173 DATASET [candidate ...]

A candidate already recorded in the output directory is not refitted or
replayed; `summary.csv` is rebuilt from every recorded candidate.
"""

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from minimax_games.games.star_realms import features, iterate, selfplay, value
from minimax_games.games.star_realms.sprt import wilson

out = Path(sys.argv[1])
data = Path(sys.argv[2])
CANDIDATES: dict[str, tuple[int, tuple[str, ...]]] = {
    "w32": (32, ()),
    "w64": (64, ()),
    "w128": (128, ()),
    "f32": (32, features.EXTRA),
}
chosen = sys.argv[3:] or list(CANDIDATES)
THREADS = 8
MATCH = range(7_000_000, 7_000_300)
TIMING = range(600, 640)
BASELINE = "w32"
out.mkdir(parents=True, exist_ok=True)

with np.load(data, allow_pickle=False) as found:
    x, y, train = (found[k] for k in ("x", "y", "train"))
held = ~train
pi03 = value.load("pi03")
champion = selfplay.Planner(pi03)
print(
    f"pi03 log-loss: training {value.log_loss(pi03.predict(x[train]), y[train]):.4f},"
    f" held-out {value.log_loss(pi03.predict(x[held]), y[held]):.4f}",
    flush=True,
)

for name in chosen:
    record_path = out / f"{name}.json"
    if record_path.is_file():
        continue
    hidden, extra = CANDIDATES[name]
    t = time.perf_counter()
    model = value.fit_ensemble(
        value.Mlp(hidden=hidden),
        x[train],
        y[train],
        seeds=range(4),
        names=features.NAMES,
        extra=extra,
    )
    fit_seconds = time.perf_counter() - t
    model.save(out / f"{name}.npz")
    train_loss = value.log_loss(model.predict(x[train]), y[train])
    held_loss = value.log_loss(model.predict(x[held]), y[held])
    parameters = sum(a.size for member in model.members for a in member.values())

    named = iterate._alternating(name, "pi03", MATCH)
    players = {name: selfplay.Planner(model), "pi03": champion}
    t = time.perf_counter()
    played = selfplay.play(
        [(s, players[a], players[b]) for s, a, b in named], threads=THREADS
    )
    match_seconds = time.perf_counter() - t
    iterate._write_games(out / f"{name}.csv", named, played.winners, played.turns)
    outcomes = iterate._outcomes(named, played, name)

    timing = selfplay.alternating(players[name], "heuristic", TIMING)
    t = time.perf_counter()
    timed = selfplay.play(timing, threads=1)
    timing_seconds = time.perf_counter() - t
    plans = int(timed.plans.sum())

    record = {
        "candidate": name,
        "hidden": hidden,
        "extra": len(extra),
        "parameters": parameters,
        "train_log_loss": round(train_loss, 4),
        "held_out_log_loss": round(held_loss, 4),
        "wins": sum(o == 1 for o in outcomes),
        "finished": sum(o is not None for o in outcomes),
        "wins_as_first": sum(o == 1 for o in outcomes[::2]),
        "ms_per_plan": round(1000 * timing_seconds / plans, 2),
        "fit_seconds": round(fit_seconds, 1),
        "match_seconds": round(match_seconds, 1),
        "timing_seconds": round(timing_seconds, 1),
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(record, flush=True)


def wins_by_seed(name: str) -> dict[int, bool]:
    with (out / f"{name}.csv").open() as f:
        return {int(r["seed"]): r["winner"] == name for r in csv.DictReader(f)}


rows = []
baseline = wins_by_seed(BASELINE) if (out / f"{BASELINE}.csv").is_file() else None
for name in CANDIDATES:
    path = out / f"{name}.json"
    if not path.is_file():
        continue
    r = json.loads(path.read_text())
    low, high = wilson(r["wins"], r["finished"])
    only_it = only_base = 0
    if baseline is not None and name != BASELINE:
        mine = wins_by_seed(name)
        only_it = sum(mine[s] and not baseline[s] for s in mine)
        only_base = sum(baseline[s] and not mine[s] for s in mine)
    differ = only_it + only_base
    z = (only_it - only_base) / math.sqrt(differ) if differ else 0.0
    rows.append(
        {
            **r,
            "win_pct": round(100 * r["wins"] / r["finished"], 1),
            "wilson_low": round(100 * low, 1),
            "wilson_high": round(100 * high, 1),
            "only_it": only_it,
            "only_baseline": only_base,
            "sign_z": round(z, 2),
        }
    )
    print(
        f"{name}: train {r['train_log_loss']:.4f}"
        f" held-out {r['held_out_log_loss']:.4f},"
        f" {r['wins']}/{r['finished']} ({rows[-1]['win_pct']}%,"
        f" {rows[-1]['wilson_low']}-{rows[-1]['wilson_high']}),"
        f" vs {BASELINE} +{only_it}/-{only_base} (z {z:.2f}),"
        f" {r['ms_per_plan']} ms/plan",
        flush=True,
    )
if rows:
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
