"""More hand-made features, and a cheaper pf01, offline (#178).

#175's test (`docs/features-window/window.py`) again, on the same data: the
training states of run 2's saved rounds 1-4 (`runs/run-2/data/`, 369,419
states), held out on those rounds' held-out states (92,354). Every fit is
an mlp (`value.fit`'s defaults: width 32, 10 epochs, batch 256, Adam at
0.001, `l2` 0.003) fitted to game results (Monte Carlo targets), from seeds
4 on: pf01 is #175's fit from seeds 0-3 on these rows, so a refit from
those seeds would be pf01 itself.

- `p32`: pf01's form, `features.EXTRA`, 4 members (seeds 4-7): a
  pf01-equivalent refit, the baseline the others are paired with;
- `m32`: the same with `features.MORE` too (seeds 4-7);
- `p32x1`, `p32x2`: pf01's form with 1 and 2 members (seeds 4, and 4-5).
  `fit_ensemble` fits each seed on its own under one standardisation, so
  these are `p32`'s first 1 and 2 members, taken rather than refitted.

Each one's planner plays pf01's (4 worlds, 20,000 states per plan, both in
Rust) on the same 300 seeds from 9,000,000, seats alternating, compared
seed by seed with `p32`. Planner cost, as #173 measured it: ms per plan on
one thread against the heuristic over seeds 600-639 (pf01's too). If `m32`
clearly beats pf01 (Wilson interval above 50%, or over 55% won), the gate
is `iterate`'s SPRT (H0 50% against H1 55%, alpha = beta = 0.05, at most
1,000 games, undecided failing) from 9,100,000, in batches of 128; if it
accepts H1, the checks: 500 seeds from 9,200,000 against pf01, and seeds
0-499 against the heuristic. The cheaper models are only recorded.

    uv run python docs/more-features/more.py runs/features-178 runs/run-2/data

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
SEEDS = range(4, 8)
BASELINE, FEATURES = "p32", "m32"
CANDIDATES: dict[str, tuple[str, ...]] = {
    BASELINE: features.EXTRA,
    FEATURES: features.EXTRA + features.MORE,
}
FEWER = {"p32x1": 1, "p32x2": 2}
"""The cheaper models: `p32`'s first members."""
THREADS = 8
MATCH = range(9_000_000, 9_000_300)
GATE = range(9_100_000, 9_101_000)
CONFIRM = range(9_200_000, 9_200_500)
LADDER = range(500)
TIMING = range(600, 640)
BATCH = 128
TEST = Sprt(0.5, 0.55, 0.05, 0.05)
EARLIER = [
    range(4_000_000),  # iterate.Settings, up to 100 rounds, and the ladder
    range(4_000_000, 4_000_500),
    range(5_000_000, 5_000_300),
    range(6_000_000, 6_200_300),
    range(7_000_000, 7_000_300),
    range(8_000_000, 8_200_500),
]
for fresh in (MATCH, GATE, CONFIRM):
    for used in EARLIER:
        assert fresh.stop <= used.start or used.stop <= fresh.start, (fresh, used)
assert MATCH.stop <= GATE.start and GATE.stop <= CONFIRM.start
assert iterate.GATE_BASE + iterate.STRIDE * iterate.MAX_ROUNDS <= 4_000_000
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


def losses(model: value.Model) -> dict[str, float]:
    return {
        "train_log_loss": round(value.log_loss(model.predict(x_train), y_train), 4),
        "held_out_log_loss": round(value.log_loss(model.predict(x_held), y_held), 4),
    }


pf01 = value.load("pf01")
champion = selfplay.Planner(pf01)
pf01_losses = losses(pf01)
print(f"pf01 log-loss: {pf01_losses}", flush=True)


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


def ms_per_plan(model: value.Model) -> float:
    """The planner's ms per plan with `model`, on one thread, against the
    heuristic over `TIMING`."""
    timing = selfplay.alternating(selfplay.Planner(model), "heuristic", TIMING)
    t = time.perf_counter()
    timed = selfplay.play(timing, threads=1)
    seconds = time.perf_counter() - t
    return round(1000 * seconds / int(timed.plans.sum()), 2)


timing_path = out / "pf01-timing.json"
if not timing_path.is_file():
    timing_path.write_text(json.dumps({"ms_per_plan": ms_per_plan(pf01)}) + "\n")
pf01_ms = json.loads(timing_path.read_text())["ms_per_plan"]
print(f"pf01: {pf01_ms} ms/plan", flush=True)


def record(
    name: str, model: value.Model, extra: int, fit_seconds: float | None
) -> None:
    """Play `name` against pf01, time its planner, and record it."""
    outcomes, seconds = match(
        name, {name: selfplay.Planner(model), "pf01": champion}, MATCH, f"{name}.csv"
    )
    found = {
        "candidate": name,
        "extra": extra,
        "members": len(model.members),
        **losses(model),
        "fit_seconds": fit_seconds,
        "ms_per_plan": ms_per_plan(model),
        "match": tally(outcomes, seconds),
    }
    (out / f"{name}.json").write_text(json.dumps(found, indent=2) + "\n")
    print(found, flush=True)


models: dict[str, value.Model] = {}
for name, extra in CANDIDATES.items():
    if (out / f"{name}.json").is_file():
        models[name] = value.Model.load(out / f"{name}.npz")
        continue
    t = time.perf_counter()
    model = value.fit_ensemble(
        value.Mlp(hidden=32),
        x_train,
        y_train,
        seeds=SEEDS,
        names=features.NAMES,
        extra=extra,
    )
    fit_seconds = round(time.perf_counter() - t, 1)
    model.save(out / f"{name}.npz")
    models[name] = model
    record(name, model, len(extra), fit_seconds)
base = models[BASELINE]
for name, members in FEWER.items():
    if (out / f"{name}.json").is_file():
        continue
    fewer = value.Model(
        base.form, base.names, base.mean, base.std, base.members[:members], base.extra
    )
    fewer.save(out / f"{name}.npz")
    record(name, fewer, len(base.extra), None)


def wins_by_seed(name: str) -> dict[int, bool]:
    with (out / f"{name}.csv").open() as f:
        return {int(r["seed"]): r["winner"] == name for r in csv.DictReader(f)}


names = [*CANDIDATES, *FEWER]
records = {n: json.loads((out / f"{n}.json").read_text()) for n in names}
baseline_wins = wins_by_seed(BASELINE)
for name in names:
    if name == BASELINE:
        continue
    wins = wins_by_seed(name)
    only = sum(wins[s] and not baseline_wins[s] for s in wins)
    only_base = sum(baseline_wins[s] and not wins[s] for s in wins)
    z = (only - only_base) / math.sqrt(only + only_base) if only + only_base else 0
    records[name]["paired"] = {
        "only_it": only,
        "only_baseline": only_base,
        "sign_z": round(z, 2),
    }
    print(f"paired {name} vs {BASELINE}: {records[name]['paired']}", flush=True)
with (out / "summary.csv").open("w", newline="") as f:
    rows = [
        {
            "candidate": "pf01",
            "extra": len(pf01.extra),
            "members": len(pf01.members),
            **pf01_losses,
            "ms_per_plan": pf01_ms,
        }
    ] + [
        {
            k: v
            for k, v in {**r, **r["match"], **r.get("paired", {})}.items()
            if k not in ("match", "paired", "seconds")
        }
        | {"match_seconds": r["match"]["seconds"]}
        for r in records.values()
    ]
    fields = list(rows[-1])
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

result = records[FEATURES]["match"]
clearly = result["wilson_low"] > 50 or result["win_pct"] > 55
summary: dict[str, object] = {
    "pf01": {**pf01_losses, "ms_per_plan": pf01_ms},
    "paired": {n: r["paired"] for n, r in records.items() if "paired" in r},
    "clearly_beats_pf01": clearly,
}
print(f"{FEATURES} clearly beats pf01: {clearly}", flush=True)

if clearly:
    players: dict[str, selfplay.Player] = {
        FEATURES: selfplay.Planner(models[FEATURES]),
        "pf01": champion,
        "heuristic": "heuristic",
    }
    named = iterate._alternating(FEATURES, "pf01", GATE)
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
    lost = sum(o == 0 for o in outcomes[:count])
    summary["gate"] = {
        "decision": "undecided" if verdict is None else verdict.value,
        "games": count,
        "wins": wins,
        "losses": lost,
        "llr": round(TEST.llr(wins, lost), 2),
        "seconds": round(time.perf_counter() - t, 1),
    }
    print(f"gate: {summary['gate']}", flush=True)
    if iterate.promotes(verdict):
        confirm, seconds = match(
            FEATURES,
            {FEATURES: players[FEATURES], "pf01": champion},
            CONFIRM,
            f"confirm-{FEATURES}-pf01.csv",
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
