# Star Realms: tournament results

Seeded bot-vs-bot results, one entry per milestone (see [the design](star-realms.md)). Each entry is reproducible: the same command gives the same wins, seats, turns and unfinished games; only the timings vary.

How to read them: game `i` uses seed `base_seed + i`; bot A moves first in even games and bot B in odd ones, so each bot has half the first-player games. "As 1st" and "As 2nd" are a bot's wins as first and second player. A game that passes `--max-turns` is **unfinished**: it counts in no win rate, and win rates are out of the finished games. ms/move is the mean time inside the bot's `choose`, per decision (automatic plays aren't decisions). Any single game replays with `python -m minimax_games.games.star_realms --seed <seed> --bots <first> <second> --log`.

## Milestone 1: engine core and baselines

- Date: 2026-10-18
- Commit: `561b0a3` plus the tournament command (#47); first card slice only (trade, combat, Authority, draw and ally abilities limited to those).
- Budgets: n/a (baselines, no search).
- Machine: Apple M1 MacBook, 8 GB, macOS (Darwin 25.2), CPython 3.14.0.

### greedy vs random

`python -m minimax_games.games.star_realms tournament --bots greedy random --games 500`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| greedy | 500/500 | 100.0% | 250/250 | 250/250 | 0.0026 |
| random | 0/500 | 0.0% | 0/250 | 0/250 | 0.0003 |

500 games, max 300 turns, 0 unfinished, mean 22.3 turns, 3.5 s wall time.

Greedy wins every game.

### greedy vs greedy

`python -m minimax_games.games.star_realms tournament --bots greedy greedy --games 500`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| greedy-A | 252/500 | 50.4% | 127/250 | 125/250 | 0.0026 |
| greedy-B | 248/500 | 49.6% | 125/250 | 123/250 | 0.0026 |

500 games, max 300 turns, 0 unfinished, mean 25.5 turns, 4.9 s wall time.

The first player won 252/500 (50.4%): no measurable seat advantage for greedy play.

### random vs random

`python -m minimax_games.games.star_realms tournament --bots random random --games 500 --max-turns 1000`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| random-A | 253/492 | 51.4% | 124/245 | 129/247 | 0.0003 |
| random-B | 239/492 | 48.6% | 118/247 | 121/245 | 0.0003 |

500 games, max 1000 turns, 8 unfinished (seeds 191, 260, 300, 314, 339, 441, 454, 476), mean 106.6 turns over the 492 finished, 14.8 s wall time.

The first player won 242/492 (49.2%). Random games are long (finished games ran to over 700 turns) and some stall: once the Trade Row is empty, the Authority automatic play gains outpaces the damage of bots that attack only at random. Hence the higher turn cap; at the default 300, 8 of the first 100 seeds are unfinished.

## Milestone 2: cheat search benchmark (nodes/sec)

- Date: 2026-10-18
- Commit: `2d64f17` plus the benchmark command (#62).
- Budget: at most 200,000 nodes per search; a search past it is stopped and its nodes read `>200000`.
- Machine: Apple M1 MacBook, 8 GB, macOS (Darwin 25.2), CPython 3.14.0.

`python -m minimax_games.games.star_realms bench` (or `task bench`) searches each position with the cheat's search (`cheat.search`, alpha-beta with `ends_turn`) at horizons of 1 and 2 turns. A position `seed:turn` is the first decision of `turn` in the greedy-vs-greedy game from `seed` (seed 1's game lasts 23 turns, seed 2's 27), so the node counts are the same on every run; only the timings vary. ms is the time of that one decision (or until it was stopped). `--states` also counts distinct states (it hashes every state, so the timings below are from the run without it), plus an exact count of each position's whole first turn.

| Position | Horizon | Nodes | Distinct states | nodes/s | ms |
|---|---|---|---|---|---|
| 1:3 | 1 | 106 | 56 | 12,879 | 8 |
| 1:3 | 2 | 625 | 315 | 14,436 | 43 |
| 1:7 | 1 | 64 | 64 | 9,185 | 7 |
| 1:7 | 2 | >200,000 | 5,321 | 11,763 | >17,002 |
| 1:11 | 1 | 36,652 | 918 | 11,077 | 3,309 |
| 1:11 | 2 | 171,722 | 5,468 | 13,211 | 12,999 |
| 1:15 | 1 | 17,158 | 724 | 12,199 | 1,407 |
| 1:15 | 2 | >200,000 | 9,250 | 13,626 | >14,678 |
| 1:19 | 1 | 6,780 | 848 | 7,844 | 864 |
| 1:19 | 2 | >200,000 | 57,104 | 13,303 | >15,034 |
| 2:5 | 1 | >200,000 | 2,693 | 15,828 | >12,636 |
| 2:5 | 2 | >200,000 | 10,771 | 18,804 | >10,636 |
| 2:13 | 1 | >200,000 | 1,960 | 12,358 | >16,184 |
| 2:13 | 2 | >200,000 | 5,341 | 16,109 | >12,415 |
| 2:21 | 1 | >200,000 | 12,623 | 10,294 | >19,430 |
| 2:21 | 2 | >200,000 | 3,560 | 11,270 | >17,747 |

- Horizon 1: 5/8 searches complete, 660,760 nodes in 53.8 s (12,272 nodes/s), 1,119 ms per completed decision.
- Horizon 2: 2/8 searches complete, 1,372,347 nodes in 100.6 s (13,648 nodes/s), 6,521 ms per completed decision.
- 2.6 minutes wall time (5 minutes with `--states`).

The whole first turn of each position (`bench.turn_tree`: every order of the mover's plays to the end of the turn, which is exactly what a 1-turn search visits, since nothing in it is pruned) against the distinct states in it:

| Position | One turn: nodes | Distinct states | Nodes per state |
|---|---|---|---|
| 1:3 | 106 | 56 | 1.9 |
| 1:7 | 64 | 64 | 1.0 |
| 1:11 | 36,652 | 918 | 40 |
| 1:15 | 17,158 | 724 | 24 |
| 1:19 | 6,780 | 848 | 8 |
| 2:5 | 847,468 | 3,864 | 219 |
| 2:13 | 1,039,568 | 2,330 | 446 |
| 2:21 | 174,834,920 | 44,750 | 3,907 |

### Verdict

**A 2-turn cheat is nowhere near fast enough for a few-hundred-game tournament, and neither is a 1-turn one.** Past the opening, 6 of 7 positions don't finish a 2-turn search in 200,000 nodes (15 s), and three don't finish even 1 turn: seed 2's turn-21 turn alone is 175 million nodes, about 4 hours at this speed. A game has dozens of cheat decisions, so even one game can take hours, against the design's target of 500 games in about 10 minutes.

**The bottleneck is tree size, and above all transpositions.** Per-node speed is steady at 8,000-19,000 nodes/s (about 80 µs per node) whatever the position, so the variation in decision time is all tree size. Within one turn, the same states are reached by many orders of the same plays: 2-40x in seed 1's positions, 200-450x mid-game in seed 2 and 3,900x at its turn 21, growing with how many cards and buys a turn has. Even inside the 200,000-node budget the searches reach each state 3-100 times. No plausible per-node speed-up (a leaner state, PyPy: #67, #68) closes a factor of thousands.

Follow-up, in order:

1. **Merge transpositions within a turn**: a transposition table keyed on the state, or generating each turn's plays in one canonical order (play all cards before buying, for instance). Merged, seed 2's turn 21 is 44,750 states, about 4 s at today's speed, instead of 4 hours.
2. **Bound what's left**: the 2-turn tree is still each of the mover's turn outcomes (up to tens of thousands) times the opponent's turn, too big to search in full. A deterministic node budget, and move ordering so alpha-beta prunes the opponent's turn, are the next levers.
3. **Only then per-node speed** (#67 profiling, #68 PyPy), which multiplies whatever the first two leave.
