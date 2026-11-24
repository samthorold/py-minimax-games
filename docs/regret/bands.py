"""Which bands of judged gap a regret run's options came from (#232): the
run's decisions and each one's options recomputed as the run drew them
(the same games, shortlists and draws, all fixed by the seeds; no
rollouts), counted by band, beside how many candidates each band held.

    uv run python docs/regret/bands.py GAMES BASE_SEED [PER_GAME]
"""

import itertools
import math
import random
import sys

from minimax_games.games.star_realms import regret
from minimax_games.games.star_realms.value import load

games, base = int(sys.argv[1]), int(sys.argv[2])
per_game = int(sys.argv[3]) if len(sys.argv) > 3 else 2
model = load("pf01")
settings = regret.Settings(threads=4)
strata = regret.Strata()
found, _ = regret.decisions(
    model, range(base, base + games), settings, per_game=per_game, seed=base
)
edges = (0.0, *strata.edges, math.inf)
names = [f"{lo:.1f}-{hi:.1f}" if hi < math.inf else f"{lo:.1f}+" for lo, hi in
         itertools.pairwise(edges)]  # fmt: skip
held = dict.fromkeys(names, 0)
drawn = dict.fromkeys(names, 0)
candidates = []
for d in found:
    seed = regret.decision_seed(d)
    options = regret.shortlist(model, d.state, settings, seed)
    picked = regret.widened(options, strata, random.Random(seed))
    candidates.append(len(options))
    if options[0].judged >= 1.0:
        continue
    for counts, these in ((held, options[1:]), (drawn, picked[1:])):
        seen = set()
        for o in these:
            gap = options[0].judged - o.judged
            for name, (lo, hi) in zip(names, itertools.pairwise(edges), strict=True):
                if (lo == 0 or lo < gap) and gap <= hi:
                    seen.add(name)
        for name in seen:
            counts[name] += 1
print(f"{len(found)} decisions; candidates per decision: median"
      f" {sorted(candidates)[len(candidates) // 2]}, min {min(candidates)},"
      f" max {max(candidates)}")  # fmt: skip
print("Band     Decisions with a candidate there  Decisions with an option rolled out")
for name in names:
    print(f"{name:8} {held[name]:<33} {drawn[name]}")
