"""The cost of a rollout at several rollout players (#229): from one turn
of each of 6 `pf01` games against itself (seeds 24,000,000-24,000,005),
the turn `pf01` chooses rolled out 8 times by `pf01`'s judgement at each
setting of worlds and states per plan, on 2 threads.

    uv run python docs/regret/cost.py
"""

import time

import numpy as np

from minimax_games.games.star_realms import regret
from minimax_games.games.star_realms.value import load

model = load("pf01")
base = regret.Settings(threads=2)
found, game_turns = regret.decisions(
    model, range(24_000_000, 24_000_006), base, per_game=1, seed=1
)
print("decision turns", [d.turn for d in found], flush=True)
for worlds, budget in ((4, 20_000), (1, 5_000), (1, 2_000)):
    s = regret.Settings(threads=2, rollout_worlds=worlds, rollout_budget=budget)
    t = time.perf_counter()
    n = 0
    lengths = []
    for d in found:
        option = regret.shortlist(model, d.state, base, regret.decision_seed(d))[0]
        r = regret.roll(model, d.state, [option.line], range(8), s)
        n += 8
        lengths += (r.turns[0] - d.turn).tolist()
    dt = time.perf_counter() - t
    print(
        f"{worlds} worlds {budget}: {1000 * dt / n:.0f} ms per rollout wall on"
        f" 2 threads, {np.mean(lengths):.1f} turns left",
        flush=True,
    )
