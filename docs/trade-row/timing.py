"""The planner's cost with the Trade Row features (#209), as #173 and #178
measured it: ms per plan on one thread against the heuristic over seeds
600-639, for `pf01` and each saved model given, interleaved twice.

    uv run python docs/trade-row/timing.py \
        runs/trade-row-209/control.npz runs/trade-row-209/t32.npz
"""

import sys
import time
from pathlib import Path

from minimax_games.games.star_realms import selfplay, value

TIMING = range(600, 640)
models = {"pf01": value.load("pf01")} | {
    Path(p).stem: value.Model.load(Path(p)) for p in sys.argv[1:]
}
for repeat in range(2):
    for name, model in models.items():
        games = selfplay.alternating(selfplay.Planner(model), "heuristic", TIMING)
        t = time.perf_counter()
        played = selfplay.play(games, threads=1)
        seconds = time.perf_counter() - t
        ms = 1000 * seconds / int(played.plans.sum())
        print(f"run {repeat + 1} {name}: {ms:.2f} ms per plan", flush=True)
