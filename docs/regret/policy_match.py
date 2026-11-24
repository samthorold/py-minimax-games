"""A rollout player against the champion as it plays (#229): `pf01` at
WORLDS worlds and BUDGET states per plan against `pf01` at its own 4 worlds
and 20,000 states, on SEEDS seeds from 24,050,000, each in both seats, in
one Rust batch on 2 threads.

    uv run python docs/regret/policy_match.py WORLDS BUDGET SEEDS
"""

import sys
import time

from minimax_games.games.star_realms import selfplay
from minimax_games.games.star_realms.reply import sign_z
from minimax_games.games.star_realms.sprt import wilson
from minimax_games.games.star_realms.value import load

worlds, budget = int(sys.argv[1]), int(sys.argv[2])
seeds = range(24_050_000, 24_050_000 + int(sys.argv[3]))
model = load("pf01")
cheap = selfplay.Planner(model, worlds, budget)
full = selfplay.Planner(model)
games = [g for s in seeds for g in ((s, cheap, full), (s, full, cheap))]
start = time.perf_counter()
played = selfplay.play(games, threads=2)
won = [None if w is None else w == (i % 2) for i, w in enumerate(played.winners)]
wins = sum(w is True for w in won)
finished = sum(w is not None for w in won)
pairs = list(zip(won[::2], won[1::2], strict=True))
both = sum(a is True and b is True for a, b in pairs)
neither = sum(a is False and b is False for a, b in pairs)
low, high = wilson(wins, finished)
print(
    f"pf01 at {worlds} worlds, {budget} states against pf01 at 4, 20,000 on seeds"
    f" {seeds.start}-{seeds.stop - 1} in both seats: {wins}/{finished}"
    f" ({100 * wins / finished:.1f}%, {100 * low:.1f}-{100 * high:.1f}%);"
    f" seeds won both / lost both {both} / {neither},"
    f" sign z {sign_z(both, neither):.2f}; {time.perf_counter() - start:.0f} s"
)
