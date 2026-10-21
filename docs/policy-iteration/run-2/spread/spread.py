"""How much candidates fitted to the same data vary against pi03: single
fits against ensembles of k fits (#164). Fits the mlp to run 2's round 1
training rows (pi03's games) from many seeds, then plays each candidate's
planner against pi03's on the same fixed seeds, in Rust."""

import csv
import sys
import time
from pathlib import Path

import numpy as np

from minimax_games.games.star_realms import features, iterate, selfplay, value

out = Path(sys.argv[1])
games = int(sys.argv[2]) if len(sys.argv) > 2 else 300
with np.load("runs/run-2/data/round-01.npz", allow_pickle=False) as found:
    x, y = found["x_train"], found["y_train"]
    hx, hy = found["x_held"], found["y_held"]
print(f"{len(y)} training states, {len(hy)} held out", flush=True)

form = value.Mlp()
fits: dict[int, value.Model] = {}
t = time.perf_counter()
for seed in range(24):
    fits[seed] = value.fit(form, x, y, seed=seed, names=features.NAMES)
print(f"24 fits in {time.perf_counter() - t:.0f} s", flush=True)

candidates: dict[str, value.Model] = {}
for seed in range(8):
    candidates[f"k1-s{seed}"] = fits[seed]
for i in range(4):
    seeds = range(8 + 4 * i, 12 + 4 * i)
    candidates[f"k4-s{seeds[0]}-{seeds[-1]}"] = value.ensemble([fits[s] for s in seeds])
for i in range(2):
    seeds = range(8 + 8 * i, 16 + 8 * i)
    candidates[f"k8-s{seeds[0]}-{seeds[-1]}"] = value.ensemble([fits[s] for s in seeds])

pi03 = value.load("pi03")
champion = selfplay.Planner(pi03)
seeds = range(5_000_000, 5_000_000 + games)
out.mkdir(parents=True, exist_ok=True)
print(f"pi03 held-out log-loss {value.log_loss(pi03.predict(hx), hy):.4f}")
rows = []
for name, model in candidates.items():
    loss = value.log_loss(model.predict(hx), hy)
    planner = selfplay.Planner(model)
    named = iterate._alternating(name, "pi03", seeds)
    players = {name: planner, "pi03": champion}
    t = time.perf_counter()
    played = selfplay.play(
        [(s, players[a], players[b]) for s, a, b in named], threads=8
    )
    seconds = time.perf_counter() - t
    iterate._write_games(out / f"{name}.csv", named, played.winners, played.turns)
    outcomes = iterate._outcomes(named, played, name)
    wins = sum(o == 1 for o in outcomes)
    finished = sum(o is not None for o in outcomes)
    rows.append((name, len(model.members), loss, wins, finished, seconds))
    print(
        f"{name}: log-loss {loss:.4f}, {wins}/{finished}"
        f" ({100 * wins / finished:.1f}%), {seconds:.0f} s",
        flush=True,
    )
with (out / "summary.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["candidate", "members", "log_loss", "wins", "finished", "seconds"])
    writer.writerows(rows)
for k in (1, 4, 8):
    rates = [100 * w / n for _, m, _, w, n, _ in rows if m == k]
    print(
        f"k={k}: {len(rates)} candidates, mean {np.mean(rates):.1f}%,"
        f" sd {np.std(rates, ddof=1):.1f}, range {min(rates):.1f}-{max(rates):.1f}"
    )
