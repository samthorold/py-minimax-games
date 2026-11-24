"""The Trade Row as it bears on whoever buys next: the fits (#209).

pf01's recipe exactly (#175, `docs/features-window/window.py`): the
training states of run 2's saved rounds 1-4 (`runs/run-2/data/`, 369,419
states), a 4-member mlp ensemble (`value.fit_ensemble`, seeds 0-3, the
`value.fit` defaults: width 32, 10 epochs, batch 256, Adam at 0.001, `l2`
0.003) fitted to game results, reading `features.EXTRA`:

- `control`: the recipe as it is, which should reproduce `pf01`;
- `t32`: the same reading `features.TRADE_ROW` after `EXTRA`, the
  candidate.

Each fit is timed (wall seconds; the training half of *learns at the speed
the feedback loop needs*), both are saved to the output directory, and the
control is compared with `pf01` parameter by parameter and by its
predictions on the held-out states. Plays no games: the population run and
the denial probes are the CLI's (`population`, `probe denial`), run on the
saved models.

    uv run python docs/trade-row/trade_row.py runs/trade-row-209 runs/run-2/data

Run it on 2 threads (`VECLIB_MAXIMUM_THREADS=2 OPENBLAS_NUM_THREADS=2
OMP_NUM_THREADS=2`), as the speed hypothesis reads.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from minimax_games.games.star_realms import features, value

out = Path(sys.argv[1])
data = Path(sys.argv[2])
ROUNDS = range(1, 5)
SEEDS = range(4)
CANDIDATES: dict[str, tuple[str, ...]] = {
    "control": features.EXTRA,
    "t32": features.EXTRA + features.TRADE_ROW,
}
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
summary: dict[str, object] = {"pf01": losses(pf01)}
print(f"pf01: {summary['pf01']}", flush=True)
models: dict[str, value.Model] = {}
for name, extra in CANDIDATES.items():
    t = time.perf_counter()
    model = value.fit_ensemble(
        value.Mlp(hidden=32),
        x_train,
        y_train,
        seeds=SEEDS,
        names=features.NAMES,
        extra=extra,
    )
    seconds = time.perf_counter() - t
    model.save(out / f"{name}.npz")
    models[name] = model
    summary[name] = {
        "extra": len(extra),
        **losses(model),
        "fit_seconds": round(seconds, 1),
    }
    print(f"{name}: {summary[name]}", flush=True)

control = models["control"]
same = all(
    np.array_equal(mine[k], theirs[k])
    for mine, theirs in zip(control.members, pf01.members, strict=True)
    for k in theirs
) and (
    np.array_equal(control.mean, pf01.mean) and np.array_equal(control.std, pf01.std)
)
largest = max(
    float(np.max(np.abs(mine[k] - theirs[k])))
    for mine, theirs in zip(control.members, pf01.members, strict=True)
    for k in theirs
)
predicted = float(np.max(np.abs(control.predict(x_held) - pf01.predict(x_held))))
summary["control_against_pf01"] = {
    "parameters_identical": same,
    "largest_parameter_difference": largest,
    "largest_held_out_prediction_difference": predicted,
}
print(f"control against pf01: {summary['control_against_pf01']}", flush=True)
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
