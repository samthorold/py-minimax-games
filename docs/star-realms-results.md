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

## Milestone 2b: cheat search with a transposition table

- Date: 2026-10-18
- Commit: `06465ae` plus the transposition table (#72).
- Budget, positions, horizons and machine: as in Milestone 2 above, so the rows compare directly.

The cheat's search now passes `alphabeta` a `key` (`node.transposition_key`: the state and the number of moves to it), so a state reached by another order of the same plays is searched once at a given depth. Nodes are still counted as they are generated, before the table is consulted, so a search's nodes are about the edges between the states it expands, not the states themselves.

| Position | Horizon | Nodes before | Nodes now | Distinct states now | nodes/s now | ms before | ms now |
|---|---|---|---|---|---|---|---|
| 1:3 | 1 | 106 | 69 | 56 | 13,704 | 8 | 5 |
| 1:3 | 2 | 625 | 361 | 315 | 14,883 | 43 | 24 |
| 1:7 | 1 | 64 | 64 | 64 | 9,228 | 7 | 7 |
| 1:7 | 2 | >200,000 | 38,590 | 28,673 | 13,741 | >17,002 | 2,808 |
| 1:11 | 1 | 36,652 | 2,057 | 918 | 13,762 | 3,309 | 150 |
| 1:11 | 2 | 171,722 | 8,192 | 5,468 | 13,847 | 12,999 | 592 |
| 1:15 | 1 | 17,158 | 1,757 | 724 | 15,535 | 1,407 | 113 |
| 1:15 | 2 | >200,000 | 20,929 | 10,149 | 15,371 | >14,678 | 1,362 |
| 1:19 | 1 | 6,780 | 1,116 | 848 | 9,234 | 864 | 121 |
| 1:19 | 2 | >200,000 | >200,000 | 152,940 | 14,667 | >15,034 | >13,636 |
| 2:5 | 1 | >200,000 | 9,039 | 3,864 | 18,354 | >12,636 | 493 |
| 2:5 | 2 | >200,000 | 28,013 | 16,194 | 18,817 | >10,636 | 1,489 |
| 2:13 | 1 | >200,000 | 4,119 | 2,330 | 14,700 | >16,184 | 280 |
| 2:13 | 2 | >200,000 | 9,018 | 6,715 | 16,383 | >12,415 | 550 |
| 2:21 | 1 | >200,000 | 107,972 | 44,750 | 11,370 | >19,430 | 9,496 |
| 2:21 | 2 | >200,000 | >200,000 | 109,203 | 13,227 | >17,747 | >15,120 |

- Horizon 1: 8/8 searches complete (5/8 before), 126,193 nodes in 10.7 s (11,834 nodes/s), 1,333 ms per completed decision.
- Horizon 2: 6/8 searches complete (2/8 before), 505,103 nodes in 35.6 s (14,196 nodes/s), 1,138 ms per completed decision.
- 46 s wall time (2.6 minutes before; 58 s with `--states`).

Each 1-turn search now expands exactly the distinct states of its turn (the "Distinct states" column matches `bench.turn_tree`'s count for every position), and seed 2's turn 21, 175 million nodes and about 4 hours before, takes 9.5 s. Per-node speed is unchanged (the table costs a hash per expanded node), so the gain is all tree size.

### Verdict

**The transposition table removes the factor of thousands; what is left is the size of the merged 2-turn tree.** Every 1-turn search now finishes, the slowest in under 10 s, and 6 of 8 2-turn searches finish, at a median of about 1 s. But a 2-turn search still costs seconds in the opening and mid-game and passes 200,000 nodes late in the game (seed 1's turn 19, seed 2's turn 21), where the mover's turn has thousands of outcomes and each is followed by a whole opponent turn. That is still far from 500 games in 10 minutes. Next: a deterministic node budget with iterative deepening (#73), then move ordering so alpha-beta prunes the opponent's turn, then per-node speed (#67, #68).

## Milestone 2c: a deterministic node budget per cheat decision

- Date: 2026-10-18
- Commit: `bdfcfa6` plus the node budget (#73).
- Budget: the cheat's default, 5,000 nodes per decision (`cheat.BUDGET`), at its default 2-turn horizon.
- Machine: Apple M1 MacBook, 8 GB, macOS (Darwin 25.2), CPython 3.14.0.

Each decision now deepens one turn at a time (`cheat.decide`): it searches 1 turn, then 2, and plays the best move of the deepest search that finished within the budget. If not even the 1-turn search finished, it plays the best of the root moves that search finished, or failing that the first legal action. Nodes are counted as they are generated, the root included, over every iteration, and the search stops at the budget exactly (no overshoot), so a decision is the same move and node count on every run.

### Budgeted decisions at the benchmark positions

`python -m minimax_games.games.star_realms bench --budget <N>` times the cheat's decision at each benchmark position (the start of a turn, the hardest decision of a turn). Depth is the deepest search that finished: 2, 1, or 0 for neither.

| Budget | 2 turns | 1 turn | Neither | nodes/s | ms per decision |
|---|---|---|---|---|---|
| 1,000 | 1/8 | 1/8 | 6/8 | 12,864 | 72 |
| 3,000 | 1/8 | 4/8 | 3/8 | 13,736 | 195 |
| **5,000** | 1/8 | 5/8 | 2/8 | 13,862 | 320 |
| 10,000 | 1/8 | 6/8 | 1/8 | 14,004 | 629 |
| 20,000 | 3/8 | 4/8 | 1/8 | 14,303 | 1,082 |

At 5,000 nodes, per position:

| Position | Depth | Nodes | nodes/s | ms |
|---|---|---|---|---|
| 1:3 | 2 | 429 | 14,853 | 29 |
| 1:7 | 1 | 5,000 | 13,839 | 361 |
| 1:11 | 1 | 5,000 | 12,701 | 394 |
| 1:15 | 1 | 5,000 | 15,231 | 328 |
| 1:19 | 1 | 5,000 | 13,025 | 384 |
| 2:5 | 0 | 5,000 | 17,946 | 279 |
| 2:13 | 1 | 5,000 | 14,843 | 337 |
| 2:21 | 0 | 5,000 | 11,252 | 444 |

### Whole games

Most decisions in a game come later in a turn, with fewer options left, so they finish deeper than the turn starts above. Every decision of two seeded games against each baseline (measured with a probe around `cheat.decide`), by budget:

| Opponent | Budget | Seed | Seconds | Turns | Decisions | ms/decision | Depth 2 / 1 / 0 |
|---|---|---|---|---|---|---|---|
| greedy | 1,000 | 0 | 2.9 | 21 | 57 | 51 | 28 / 16 / 13 |
| greedy | 1,000 | 1 | 6.2 | 39 | 113 | 54 | 48 / 47 / 18 |
| greedy | 3,000 | 0 | 6.4 | 21 | 56 | 115 | 33 / 18 / 5 |
| greedy | 3,000 | 1 | 8.1 | 25 | 62 | 130 | 33 / 24 / 5 |
| greedy | **5,000** | 0 | 9.6 | 21 | 56 | 171 | 35 / 18 / 3 |
| greedy | **5,000** | 1 | 12.1 | 25 | 65 | 186 | 42 / 18 / 5 |
| greedy | 10,000 | 0 | 25.3 | 29 | 75 | 338 | 54 / 18 / 3 |
| greedy | 10,000 | 1 | 21.9 | 22 | 59 | 372 | 42 / 13 / 4 |
| heuristic | 3,000 | 0 | 6.6 | 23 | 51 | 130 | 28 / 20 / 3 |
| heuristic | 3,000 | 1 | 8.9 | 23 | 56 | 159 | 24 / 31 / 1 |
| heuristic | **5,000** | 0 | 10.0 | 23 | 51 | 196 | 32 / 18 / 1 |
| heuristic | **5,000** | 1 | 14.8 | 23 | 56 | 264 | 25 / 31 / 0 |

At the default budget, through the tournament command (`tournament --bots cheat greedy --games 2`, and the same with `heuristic`):

- cheat vs greedy: 2 games, 23 turns on average, **182 ms per cheat decision**, 22.0 s wall time (11 s per game).
- cheat vs heuristic: 2 games, 23 turns on average, **213 ms per cheat decision**, 22.8 s wall time (11 s per game).

### Choosing the default

**5,000 nodes**: a 2-turn cheat game takes about 11 s, so a 300-game cheat tournament takes about an hour, and 60% of its decisions finish the full 2-turn search (96% finish at least 1 turn; 134, 85 and 9 of 228 decisions in the four games above). 3,000 nodes would cut the time by about a third (40 minutes for 300 games) but finish 2 turns in about half the decisions (118 of 225); 10,000 doubles the time for a few more 2-turn finishes. The budget is `CheatBot(budget=...)`.

Ordering the root moves by the previous iteration's best move (searching it first at 2 turns) was tried and dropped: over every decision of the first 12 turns of three greedy games (131 decisions) at 5,000 nodes, it finished 2 turns in 78 decisions instead of 76, with 1.3% fewer nodes, and it would change which of two equally good moves the cheat plays, so a budgeted decision would no longer always match the unbudgeted search. Sharing one transposition table across the iterations was also left out: entries are keyed by remaining depth and move count, and a 1-turn search's nodes are all in the mover's turn while the 2-turn search's same-depth nodes are in the opponent's, so no entry could be reused.

### Verdict

**Every cheat decision now finishes, deterministically, in a fraction of a second, so the milestone-2 tournaments can run (#64).** The price is depth: at the turn starts of mid and late game the 2-turn search rarely finishes in 5,000 nodes, and the cheat then plays its 1-turn move. Move ordering so alpha-beta prunes the opponent's turn, and per-node speed (#67, #68), would let the same budget finish 2 turns more often.

## Milestone 2: the ladder (heuristic, cheat, greedy)

- Date: 2026-10-18
- Commit: `b83aea5` plus `tournament --jobs` (#64).
- Budgets: the cheat at its defaults, a **2-turn horizon** and **5,000 nodes per decision** (`CheatBot()`, `cheat.BUDGET`), deepening 1 then 2 turns (Milestone 2c). Greedy and heuristic don't search.
- Machine: Apple M1 MacBook, 8 GB (4 performance and 4 efficiency cores), macOS (Darwin 25.2), CPython 3.14.0.

500 games per pairing, seeds 0-499, alternating seats, max 300 turns. The two cheat pairings ran at the same time, each as three runs of consecutive seeds (0-167, 168-333, 334-499) in its own process, combined with `tournament.combine`, which is what `--jobs 3` does (its split is 168/168/164; results are per seed, so any split gives the same wins, seats and turns). All six processes took 72-81 minutes, so the whole cheat ladder took 81 minutes of wall time. **The cheat's ms/decision below is from six processes sharing the machine's eight cores, half of them efficiency cores, so it is about twice the uncontended figure** (182-213 ms in Milestone 2c); heuristic vs greedy ran alone.

### heuristic vs greedy

`python -m minimax_games.games.star_realms tournament --bots heuristic greedy --games 500`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| heuristic | 427/500 | 85.4% | 218/250 | 209/250 | 0.0050 |
| greedy | 73/500 | 14.6% | 41/250 | 32/250 | 0.0027 |

500 games, 0 unfinished, mean 23.2 turns, 5.0 s wall time. The first player won 259/500 (51.8%).

### cheat vs greedy

`python -m minimax_games.games.star_realms tournament --bots cheat greedy --games 500 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| cheat | 411/500 | 82.2% | 220/250 | 191/250 | 393 |
| greedy | 89/500 | 17.8% | 59/250 | 30/250 | 0.0051 |

500 games, 0 unfinished, mean 27.8 turns, 36,574 cheat decisions (73 per game), 4.0 hours of cheat thinking time (81 minutes of wall time over three processes). The first player won 279/500 (55.8%).

### cheat vs heuristic

`python -m minimax_games.games.star_realms tournament --bots cheat heuristic --games 500 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| cheat | 268/500 | 53.6% | 154/250 | 114/250 | 418 |
| heuristic | 232/500 | 46.4% | 136/250 | 96/250 | 0.0098 |

500 games, 0 unfinished, mean 25.5 turns, 31,943 cheat decisions (64 per game), 3.7 hours of cheat thinking time (75 minutes of wall time over three processes). The first player won 290/500 (58.0%).

### Reading the ladder

**cheat ≈ heuristic > greedy.** Both beat greedy by a wide margin, the heuristic slightly more (85.4% against 82.2%), and the cheat beats the heuristic only narrowly: 53.6%, whose 95% interval (about ±4.4 points) still includes 50%. Seeing the true state, deck order included, and searching 2 turns buys the cheat about as much as the heuristic's hand-written buying policy (faction synergy, economy early and combat late, bases) does without any search. Two things hold the cheat back. Its budget: at 5,000 nodes the 2-turn search finishes in only about 60% of decisions (Milestone 2c), and at turn starts in mid and late game the cheat usually plays its 1-turn move. And its evaluation: `evaluate` values a buy by its average primary trade and combat per card, which knows nothing of ally abilities or of a base's value over several turns, the very things the heuristic is tuned on. Against stronger opponents the first-player advantage also grows: 51.8% with heuristic vs greedy, 55.8% and 58.0% in the cheat pairings.

**What the sampler (milestone 3) should be compared against.** PIMC runs the cheat's search on sampled worlds, so it can't be expected to beat the cheat at the same per-world settings: the cheat is its upper bound. The bar for a legitimate bot is the **heuristic**, the strongest bot that only sees its view. Compare the sampler against heuristic, greedy and the cheat with these seeds and seat alternation, at the same `evaluate`, a 2-turn horizon and an equal total node budget per decision (K worlds × nodes per world against the cheat's 5,000, or the cheat raised to match). Since the cheat itself only just beats the heuristic, a better `evaluate` (tuned weights, or rollouts) and a budget that finishes 2 turns more often (move ordering, per-node speed: #67, #68) will likely matter as much as sampling for the sampler to beat the heuristic.

## Milestone 2d: profile of the cheat's search

- Date: 2026-10-18
- Commit: `9e32e95` plus the profile command (#67).
- Workload: the cheat's real one at its defaults (2-turn horizon, 5,000 nodes per decision): its budgeted decisions at the 8 benchmark positions and its 11 decisions in turns 1-6 of seed 0's cheat-vs-heuristic game, 19 decisions and 37,838 nodes in all.
- Machine: Apple M1 MacBook, 8 GB, macOS (Darwin 25.2), CPython 3.14.0.

`python -m minimax_games.games.star_realms profile` (or `task profile`) runs it under `cProfile` and writes the report to `profiles/profile.txt` (see the README's Development section for how to read it). Without the profiler the workload takes 2.8 s (13,700 nodes/s, 73 µs per node); under it, 8.8 s (3.2x). Call counts are exact and the same on every run; the times below are the profiler's, so compare them with each other (two runs agreed to within 0.1 point on every share).

### Where the time goes

| Area | % of search | Calls | Calls per node |
|---|---|---|---|
| Generating children (`StarRealmsNode.children`) | 83.7 | | |
| &nbsp;&nbsp;Move generation (`legal_actions` of each expanded node) | 5.0 | 14,659 | 0.4 |
| &nbsp;&nbsp;`rules.advance` | 78.2 | 37,826 | 1.0 |
| &nbsp;&nbsp;&nbsp;&nbsp;Legality check in `rules.apply` (`legal_actions`) | 13.6 | 37,826 | 1.0 |
| &nbsp;&nbsp;&nbsp;&nbsp;Automatic-play scan (`_next_automatic`: `legal_actions`) | 37.3 | 100,304 | 2.7 |
| &nbsp;&nbsp;&nbsp;&nbsp;Applying actions (`rules._apply`) | 25.9 | 100,304 | 2.7 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;of which `dataclasses.replace` | 21.8 | 392,316 | 10.4 |
| `evaluate` | 7.5 | 12,953 | 0.3 |
| Transposition table (key, lookup, store, `_rebase`) | 4.9 | 24,866 | 0.7 |
| Search overhead (`alphabeta`, node comparisons, the budget) | 4.0 | | |

Cutting across those areas, **hashing and equality take 25.5%** (self time of every `__hash__` and `__eq__`, 13.9 million calls, 370 per node): `builtins.hash` 11.6%, `Enum.__hash__` 6.8% and `Card.__hash__` 5.9%. Almost all of it is hashing `Card`s, in `legal_actions`'s `dict.fromkeys` (916,740 calls, 26% cumulative) and the rules' `in` checks: a frozen dataclass's generated `__hash__` hashes a tuple of all nine fields on every call, and two of them are enums whose `__hash__` is Python code. Hashing the states for the transposition table is cheap by comparison (`GameState.__hash__` 0.1%).

**`legal_actions` is the hotspot**: 152,790 calls (4.0 per node) and 53% of the search's time cumulatively, but only one call in ten generates a node's moves. The rest are the rules re-deriving what they already know: `rules.apply` recomputes the legal actions to check an action the search took from them, and `auto_play` recomputes the whole list after every automatic action just to find the next automatic one (2.7 scans per node).

### Cheapest wins, for a later optimisation issue

In order of expected gain for the effort (not done here). Two were checked by patching them in for a run without the profiler (same decisions, best of two runs), as the profiler's share overstates calls as small as a hash:

1. **Skip the legality check inside the search** (`rules.apply` → `_apply` when the action comes from `legal_actions`, e.g. a trusted `advance` for the search): 13.6% of the profile, **13% less wall time** measured.
2. **Make `Card` cheap to hash**: cards are immutable catalogue entries, so cache the hash (or hash by name). A crude per-instance cache patched in gave **13% less wall time**; a hash stored on the card would gain more, and it also speeds `Card.__eq__` and the `in` checks.
3. **A dedicated automatic-play scan**: find the next automatic action from the hand and cards in play only (automatic actions are always `PlayCard`, `UseAlly` or `UseBase`) instead of building every legal action, Trade Row and attacks included: most of the 37.3%.
4. **Build states directly instead of `dataclasses.replace`** (10 calls per node, 21.8%): `replace` re-inspects the dataclass's fields on every call; a constructor call with the fields spelled out, or one `replace` per action instead of several, is cheaper.

`evaluate` (7.5%), the transposition table (4.9%) and the search itself (4.0%) are not worth optimising yet. Between them the four wins target most of `rules.advance`'s 78%, and the two measured ones alone save about a quarter of the wall time; at 5,000 nodes per decision, more nodes per second means more decisions finish the 2-turn search, the cheat's main handicap in the ladder above. PyPy (#68) multiplies whatever is left, and can reuse `task profile`.

## Milestone 2e: PyPy against CPython

- Date: 2026-10-18
- Commit: `cb98d4c` (#68: measurements only, no code change).
- Interpreters: CPython 3.14.0 (the project's environment; there is no 3.13 in it) against PyPy 3.12.14 (PyPy 8.0.0, `uv python install pypy-3.12.14`).
- Machine: Apple M1 MacBook, 8 GB, macOS (Darwin 25.2). Each run alone on the machine, one at a time.

PyPy has no 3.13, and the project requires `>=3.13`, so PyPy ran from a copy of the repo with `requires-python = ">=3.12"` and its own environment, outside the repo:

```
uv python install pypy-3.12.14
# in a copy of the repo, with requires-python lowered to ">=3.12"
UV_PROJECT_ENVIRONMENT=<copy>/.venv-pypy uv sync --python pypy@3.12
UV_PROJECT_ENVIRONMENT=<copy>/.venv-pypy uv run --frozen python -m minimax_games.games.star_realms <command>
```

Lowering `requires-python` is the only change PyPy needs: every dependency installs, superenums and the dev tools included (ruff, ty, actionlint and shellcheck ship platform binaries, not CPython extensions), and no code uses anything newer than 3.12. The CPython side ran the same commands with plain `uv run` in the repo.

The machine sometimes slept during the long runs; the reported times are `time.perf_counter`, which on macOS stops while asleep, and they agree with each process's CPU time to within 1%.

### Results

| Workload | CPython 3.14 | PyPy 3.12 | PyPy speed-up |
|---|---|---|---|
| Test suite (`pytest -q`) | 1,245 passed, 12.2 s | 1,243 passed, 2 failed, 20-21 s | 0.6x |
| `bench`, horizon 1 | 11.0 s, 11,430 nodes/s | 7.3 s, 17,227 nodes/s | 1.5x |
| `bench`, horizon 2 | 37.3 s, 13,554 nodes/s | 19.9 s, 25,383 nodes/s | 1.9x |
| `bench --budget 5000` | 2.6 s, 13,458 nodes/s, 329 ms/decision | 2.9 s, 12,236 nodes/s, 362 ms/decision | 0.9x |
| `tournament --bots cheat greedy --games 20` | 228 ms/decision, 323 s | 140 ms/decision, 199 s | 1.6x |
| `tournament --bots cheat greedy --games 60` | 220 ms/decision, 902 s | 129 ms/decision, 530 s | 1.7x |

**Same results on both.** Every `bench` search visits the same number of nodes, every budgeted decision finishes the same depth with the same nodes, and the tournaments give the same wins, seats and mean turns (20 games: cheat 14/20, 8/10 first and 6/10 second, 27.7 turns; 60 games: cheat 47/60, 25/30 and 22/30, 26.6 turns). No hash-order or set-iteration dependence showed up.

**The test suite passes except two profiler tests**, both in `tests/star_realms/test_profiling.py`, and both because PyPy's `cProfile` reports differently, not because the code is wrong: `test_hashing_lists_the_self_time_of_every_hash_and_equality_by_class` finds no `Card.__hash__` entry (PyPy's profiler doesn't record the dataclass-generated `__hash__` that builtin `hash` calls), and `test_the_breakdown_splits_the_search_time_into_nested_areas` finds `cheat.decide`'s cumulative time (0.1 ms) smaller than its children's (150 ms). So `task profile` would need CPython anyway. The suite as a whole is slower under PyPy: it is many short runs, which the JIT doesn't have time to compile.

**Warm-up decides the speed-up.** PyPy is slower than CPython until its JIT has compiled the search: the first `bench` searches (tens of nodes) ran 3-5x slower, and the whole 35,000-node `--budget 5000` run (under 3 s) was 10% slower. The speed-up then grows with the length of the run: 1.5x over `bench`'s horizon-1 searches, 1.9x over its horizon-2 searches (2.1x on the largest, seed 1's turn 19), 1.6x over a 20-game tournament, 1.7x over 60 games, and 1.75x over that run's last 40 games (331 s against 579 s). It levels off at about **1.7-1.9x** once warm. The non-search bots get slower (greedy 0.003 → 0.022 ms per move), which doesn't matter beside the cheat's 100+ ms.

1.7-1.9x is modest for PyPy on pure-Python code, and in line with what the profile (Milestone 2d) shows the search doing: mostly allocating short-lived frozen dataclasses and tuples (`dataclasses.replace`, 10 per node), building lists and dicts of legal actions, and hashing cards and enums into dicts and sets. The JIT speeds the interpretation but can't remove those allocations and hash-table operations, which escape into the states the search keeps.

### Recommendation

**Don't adopt PyPy now.** 1.7x on long tournaments is real (the 81-minute cheat ladder of Milestone 2 would take about 48 minutes), but:

- **It costs support.** `requires-python` must drop to `>=3.12` (no PyPy 3.13 exists), ruff's `target-version` with it, and CI needs a PyPy entry, which turns the single `pre-commit` job into a matrix and renames the required check, so the `main` ruleset must be updated in step. ty and actionlint do install under PyPy, so the hooks themselves would run.
- **It loses the profiler.** Two profiling tests fail and `task profile`'s breakdown is wrong under PyPy, so optimisation work stays on CPython either way.
- **The CPython optimisations come first and are comparable.** Milestone 2d's two measured wins (skipping the legality check in the search, a cheap `Card` hash) save about a quarter of the wall time between them, about 1.3x, and the dedicated automatic-play scan and building states without `dataclasses.replace` target most of the rest of `rules.advance`'s 78%. Together they plausibly reach PyPy's 1.7x on CPython, for everyone and without a second interpreter. They also cut the very allocation and hashing that limits PyPy, so PyPy's own gain after them is unknown.
- **Short runs get slower.** The budgeted benchmark, the tests and any single-game replay run slower under PyPy.

Re-measure after the CPython optimisations of #67's list are done (the same commands as above): if PyPy then still gives 1.5x or more on a 60-game tournament, adopting it for long tournaments is worth the support cost.

## Milestone 2f: skip the search's legality re-check, hash cards by name

- Date: 2026-10-19
- Commit: `192a49e` plus #79.
- Machine: Apple M1 MacBook, 8 GB, macOS (Darwin 25.2), CPython 3.14.0. Each run in the foreground, alone, one at a time; "before" is `192a49e`, "after" this change.

The first two wins of Milestone 2d's list. **Legality re-check:** `rules.advance_legal` is `rules.advance` for an action taken from `legal_actions(state)`, without `apply`'s check, which recomputed `legal_actions` just to find the action there; the search's children (and `bench`'s turn-tree count) use it, while `rules.apply` and `rules.advance`, which bots' actions go through, still reject illegal ones. **Card hashing:** `Card.__hash__` is the hash of its name (unique in the catalogue, which `catalogue.parse` already enforces), instead of the generated hash over all nine fields, two of them enums hashed in Python.

**Same results.** Every `bench` search visits the same nodes, every budgeted decision finishes the same depth with the same nodes, the profiled workload generates the same 37,838 nodes, and the 20-game tournament gives the same wins, seats and mean turns (cheat 14/20, 8/10 first and 6/10 second, 27.7 turns).

| Workload | Before | After | Change |
|---|---|---|---|
| `bench`, horizon 1 | 10.8 s, 11,667 nodes/s | 7.8 s, 16,113 nodes/s | 1.38x |
| `bench`, horizon 2 | 35.6 s, 14,181 nodes/s | 27.6 s, 18,333 nodes/s | 1.29x |
| `bench --budget 5000` | 13,851 nodes/s, 319.7 ms/decision | 18,444 nodes/s, 240.1 ms/decision | 1.33x |
| `tournament --bots cheat greedy --games 20` | 221 ms/decision, 313.8 s | 170 ms/decision, 241.7 s | 1.30x |
| `task profile`, unprofiled | 13,936 nodes/s | 18,406 nodes/s | 1.32x |

About a quarter less wall time, as Milestone 2d's patched measurements predicted. In `task profile` (shares of the profiled search):

- The legality check in `rules.apply` (13.5%) is gone: the search no longer calls `apply`, and `legal_actions` calls fall from 152,790 to 114,964.
- **Hashing and equality fall from 25.5% to 9.2%** (13.9 million calls to 4.8 million): `Enum.__hash__` (6.8%) is gone, `builtins.hash`'s self time falls from 11.4% to 3.0%, and `Card.__hash__` from 5.9% to 4.6% (now only the call of the one-line method).
- What's left of `rules.advance_legal` (75.0%, was 78.2% as `rules.advance`) is the other two wins: the automatic-play scan (38.3%) and applying actions (35.8%, of which `dataclasses.replace` 30.2%). `evaluate` is now 10.3%.

## Milestone 2g: a dedicated automatic-play scan

- Date: 2026-10-19
- Commit: `9b18e68` plus #80.
- Machine: Apple M1 MacBook, 8 GB, macOS (Darwin 25.2), CPython 3.14.0. Each run in the foreground, alone, one at a time; "before" is `9b18e68`, "after" this change.

The third win of Milestone 2d's list. `rules.auto_play` found each automatic action by building the whole `legal_actions` list (scraps, buys, attacks and all) and taking the first automatic one, about 2.7 times per node. It now scans only the kinds of action that can be automatic (`PlayCard`, `UseAlly`, `UseBase`, which `legal_actions` lists first, from the same generator), lazily, and stops at the first one `rules.automatic` accepts. A property test on seeded states checks it takes the same actions in the same order as the old definition.

**Same results.** Every `bench` search visits the same nodes, every budgeted decision finishes the same depth with the same nodes, the profiled workload generates the same 37,838 nodes, and the 20-game tournament gives the same wins, seats and mean turns (cheat 14/20, 8/10 first and 6/10 second, 27.7 turns).

| Workload | Before | After | Change |
|---|---|---|---|
| `bench`, horizon 1 | 7.6 s, 16,510 nodes/s | 5.7 s, 22,326 nodes/s | 1.35x |
| `bench`, horizon 2 | 27.2 s, 18,557 nodes/s | 21.0 s, 24,026 nodes/s | 1.29x |
| `bench --budget 5000` | 18,617 nodes/s, 237.9 ms/decision | 25,102 nodes/s, 176.4 ms/decision | 1.35x |
| `tournament --bots cheat greedy --games 20` | 172 ms/decision, 243.8 s | 128 ms/decision, 181.2 s | 1.35x |
| `task profile`, unprofiled | 18,517 nodes/s | 24,883 nodes/s | 1.34x |

In `task profile` (shares of the profiled search, which falls from 6.17 s to 4.49 s):

- **The automatic-play scan falls from 2.37 s (38.4%) to 0.79 s (17.5%)**, 62.6 to 20.8 us per node, for the same 100,304 scans. `legal_actions` calls fall from 114,964 to 14,660 (only the search's move generation is left).
- Applying actions (`rules._apply`) is now the hotspot: 2.13 s, 47.4%, of which `dataclasses.replace` 40.1%, the target of #81. `evaluate` is 14.0%.

## Milestone 2h: build states without `dataclasses.replace`

- Date: 2026-10-19
- Commit: `f5cdba2` plus #81.
- Machine: Apple M1 MacBook, 8 GB, macOS (Darwin 25.2), CPython 3.14.0. Each run in the foreground, alone, one at a time; "before" is `f5cdba2`, "after" this change.

`dataclasses.replace` looks up the class's fields on every call, about 10 calls per node. `GameState` and `PlayerState` now each have a `replace` method with one keyword per field (`None` keeps it: no field can be `None`) that calls the constructor directly, and `rules` uses it. Each action in `rules._apply` now builds its new player and state once, with all their changes, instead of one copy per changed field: 392,316 `replace` calls in the profiled workload become 280,438, and dataclass constructions 220,034 become 168,331. The states are still frozen, slotted and hashable dataclasses; a test checks each `replace` agrees with `dataclasses.replace` field by field, and another plays seeded games with `dataclasses.replace` patched to fail.

**Same results.** Every `bench` search visits the same nodes, every budgeted decision finishes the same depth with the same nodes, the profiled workload generates the same 37,838 nodes, and the 20-game tournament gives the same wins, seats and mean turns (cheat 14/20, 8/10 first and 6/10 second, 27.7 turns).

| Workload | Before | After | Change |
|---|---|---|---|
| `bench`, horizon 1 | 5.6 s, 22,494 nodes/s | 3.9 s, 32,582 nodes/s | 1.45x |
| `bench`, horizon 2 | 20.8 s, 24,260 nodes/s | 13.5 s, 37,340 nodes/s | 1.54x |
| `bench --budget 5000` | 24,836 nodes/s, 178.3 ms/decision | 39,720 nodes/s, 111.5 ms/decision | 1.60x |
| `tournament --bots cheat greedy --games 20` | 127 ms/decision, 180.9 s | 82.8 ms/decision, 117.5 s | 1.54x |
| `task profile`, unprofiled | 24,498 nodes/s | 39,122 nodes/s | 1.60x |

**Since Milestone 2f's "before" (#79 to #81):** 2.8x on `bench` horizon 1 (11,667 to 32,582 nodes/s), 2.6x on horizon 2, 2.9x on `bench --budget 5000` (319.7 to 111.5 ms/decision) and 2.7x on the tournament (221 to 82.8 ms/decision, 313.8 s to 117.5 s), with every result unchanged.

In `task profile` (shares of the profiled search, which falls from 4.53 s to 3.07 s):

- **Applying actions falls from 2.15 s (47.4%) to 0.70 s (22.6%)**, 56.7 to 18.4 us per node. The `dataclasses.replace` row (1.81 s, 40.0%) is gone; building states with the new methods takes 0.39 s (12.6%), of which the generated frozen `__init__`'s self time is 0.19 s (6.2%).
- The frozen `__init__` sets each field with `object.__setattr__`: in a micro-benchmark, constructing a 12-field frozen, slotted dataclass takes about 0.67 us against 0.13 us for a plain slotted class assigning its fields. Any frozen class pays that, and the states are already `slots=True`, so there is no cheap, readable win left there without giving up immutability; not pursued.
- **The automatic-play scan (`rules._next_automatic`) is now the biggest hotspot**, 0.79 s (25.6%), unchanged in time. Then applying actions (22.6%) and `evaluate` (0.63 s, 20.6%, of which `builtins.sum` 0.16 s self time); hashing and equality is 11.4%, move generation 10.2%.

## Milestone 3: budget

- Date: 2026-10-19
- Commit: `709074d` plus #88 (`bench --budget N --worlds K`, `tournament --budget N --worlds K`, `sampler.Vote.depths`).
- Machine: Apple M1 MacBook, 8 GB (4 performance and 4 efficiency cores), macOS (Darwin 25.2), CPython 3.14.0.
- Horizon: 2 turns throughout, deepening 1 then 2 (`cheat.decide`). The sampler's K worlds share the total budget (`budget // K` nodes each).

The optimisation pass (Milestones 2f-2h) made the search 2.6-2.9x faster, so the 5,000-node default, chosen in Milestone 2c when a cheat game took 11 s, was re-measured: how deep the cheat and the sampler search at 5k, 10k, 20k and 40k nodes per decision, how long that takes, and whether it wins more. The single-process timings ran in the foreground, one at a time; the tournaments ran with `--jobs 3`, so their ms/move is inflated by contention.

### Budgeted decisions at the benchmark positions

`bench --budget N` (cheat) and `bench --budget N --worlds K` (sampler, from the mover's view, worlds seeded with the position's seed) at the eight benchmark positions, turn starts, the hardest decisions of a turn. Each cell: ms per decision, then how many searches finished 2 turns and at least 1 turn (the cheat's 8 decisions; the sampler's 8 × K worlds). Raw outputs: `bench-cheat-<N>.txt`, `bench-sampler-<N>-<K>.txt`.

| Budget | cheat | sampler K=1 | K=4 | K=8 | K=16 |
|---|---|---|---|---|---|
| **5,000** | 111 ms, 1/8, 6/8 | 101 ms, 2/8, 6/8 | 118 ms, 4/32, 12/32 | 126 ms, 8/64, 19/64 | 135 ms, 11/128, 32/128 |
| 10,000 | 228 ms, 1/8, 7/8 | 188 ms, 3/8, 6/8 | 222 ms, 5/32, 22/32 | 239 ms, 8/64, 23/64 | 254 ms, 16/128, 35/128 |
| 20,000 | 388 ms, 3/8, 7/8 | 336 ms, 4/8, 7/8 | 437 ms, 6/32, 22/32 | 450 ms, 9/64, 44/64 | 478 ms, 16/128, 50/128 |
| 40,000 | 632 ms, 6/8, 7/8 | 552 ms, 5/8, 7/8 | 833 ms, 8/32, 24/32 | 869 ms, 11/64, 46/64 | 909 ms, 17/128, 87/128 |

About 40,000 nodes/s everywhere (36-40k with many worlds: each world pays for its own root and 1-turn search). The sampler with K=1 is the cheat on one sampled world. At turn starts, splitting the budget starves each world: at 5,000 nodes with K=8, 45 of 64 worlds finish not even the 1-turn search, and with 4 or more worlds at most a quarter of them finish 2 turns at any budget measured.

### Decisions in real games

Every decision of seeded games against the heuristic, the search bot moving first in even seeds (a probe around `cheat.decide` and `sampler.vote`, single process; `games-*.txt`). Most decisions come later in a turn, with fewer options, so they finish far deeper than the turn starts above. Depth: the share of searches (the sampler's: of its worlds' searches) that finished 2 turns, then at least 1.

| Bot | Budget | K | Games (seeds) | Decisions | ms/decision | 2 turns | ≥1 turn |
|---|---|---|---|---|---|---|---|
| cheat | **5,000** | - | 4 (0-3) | 254 | 70 | 63% | 99% |
| cheat | 10,000 | - | 4 (0-3) | 252 | 129 | 67% | 100% |
| cheat | 20,000 | - | 4 (0-3) | 258 | 217 | 72% | 99% |
| cheat | 40,000 | - | 4 (0-3) | 259 | 351 | 79% | 100% |
| sampler | **5,000** | **4** | 2 (0-1) | 121 | 101 | 40% | 92% |
| sampler | 5,000 | 8 | 2 (0-1) | 119 | 107 | 37% | 89% |
| sampler | 10,000 | 4 | 2 (0-1) | 123 | 177 | 50% | 95% |
| sampler | 10,000 | 8 | 2 (0-1) | 118 | 190 | 46% | 92% |
| sampler | 20,000 | 1 | 2 (0-1) | 116 | 213 | 75% | 97% |
| sampler | 20,000 | 4 | 2 (0-1) | 118 | 303 | 59% | 96% |
| sampler | 20,000 | 8 | 2 (0-1) | 115 | 350 | 52% | 95% |
| sampler | 20,000 | 16 | 2 (0-1) | 121 | 409 | 40% | 91% |
| sampler | 40,000 | 4 | 2 (0-1) | 123 | 512 | 67% | 97% |
| sampler | 40,000 | 8 | 2 (0-1) | 114 | 635 | 59% | 97% |

Doubling the budget roughly doubles the time but adds only 4-7 points of 2-turn finishes for the cheat (63% at 5,000 to 79% at 40,000, 8x the time). For the sampler, each doubling of K costs about as much depth as halving the budget.

### Tournaments against the heuristic

100 games, seeds 0-99, alternating seats, `tournament --bots <bot> heuristic --games 100 --jobs 3 --budget N [--worlds K]` (20,000 and the sampler's 10,000 as two runs of 50, seeds 0-49 and 50-99). No game was unfinished. A 100-game win rate has a 95% interval of about ±10 points.

| Bot | Budget | K | Win% | As 1st | As 2nd | ms/move (contended) | Wall time |
|---|---|---|---|---|---|---|---|
| cheat | **5,000** | - | 54.0% | 33/50 | 21/50 | 98 | 218 s |
| cheat | 10,000 | - | 55.0% | 33/50 | 22/50 | 170 | 357 s |
| cheat | 20,000 | - | 50.0% | 32/50 | 18/50 | 277 | 610 s |
| sampler | 5,000 | 1 | 46.0% | 24/50 | 22/50 | 103 | 227 s |
| sampler | **5,000** | **4** | 50.0% | 25/50 | 25/50 | 132 | 292 s |
| sampler | 5,000 | 8 | 47.0% | 21/50 | 26/50 | 148 | 331 s |
| sampler | 10,000 | 4 | 45.0% | 20/50 | 25/50 | 227 | 569 s |

**No budget or K beats the heuristic measurably better than another.** Every result is within 46-55%, inside each other's intervals, and more budget does not trend upward: the cheat wins 54%, 55% and 50% at 5k, 10k and 20k (the 500-game ladder's 53.6% at 5k agrees). Budget is not what holds the search bots back against the heuristic now; the evaluation is (Milestone 2, Reading the ladder). 40,000 nodes and 20,000 for the sampler were not run as tournaments: they would take 20-40 minutes each, and nothing from 5k to 20k suggests a gain.

### A search-bot tournament at the default

`tournament --bots sampler cheat --games 30 --budget 5000 --worlds 4`, seeds 0-29 (`t-sampler-cheat-5000-4*.txt`): the cheat won 19/30, 22.7 turns per game on average.

| Jobs | sampler ms/move | cheat ms/move | Wall time | Per game | 500 games |
|---|---|---|---|---|---|
| 3 | 109 | 76 | 108.0 s | 3.6 s | **about 30 minutes** |
| 6 | 137 | 94 | 96.5 s | 3.2 s | about 27 minutes |

Six processes gain little over three on this machine (four performance cores). Time scales with the budget, so 10,000 nodes would take about an hour for 500 games and 20,000 nearly two. **The design's 10-minute target is about 3x away even at 5,000 nodes**; reaching it would mean about 1,500-2,000 nodes, where (Milestone 2c) most turn-start decisions finish not even 1 turn.

### Choosing the defaults

**Total budget: 5,000 nodes, shared** (`cheat.BUDGET`, which `SamplerBot` defaults to). It is the cheapest budget measured, and more did not win more: 10k and 20k cost 2x and 4x the time for a few more 2-turn finishes and no measurable gain against the heuristic. Keeping it lets the milestone-3 ladder (#89) run a 500-game search-bot pairing in about 30 minutes with `--jobs 3`, and a search bot against a baseline in about 18 minutes (cheat vs heuristic: 218 s per 100 games). Going below 5,000 to approach 10 minutes would give up depth that is already thin at turn starts, so it is not proposed until move ordering or a better evaluation makes the search cheaper or more valuable.

**Sampler worlds: K = 4** (`sampler.WORLDS`, was 5). At 5,000 nodes it was the sampler's best (50.0% against 46.0% for K=1 and 47.0% for K=8, all within noise), the cheapest of the multi-world settings, and with 1,250 nodes per world 92% of its worlds' searches finish at least 1 turn in games (89% at K=8). Fewer worlds sample hidden information less; more worlds leave each world too few nodes: at 5,000 nodes, K=8 and 16 finish not even 1 turn at most turn starts. K=4 is the measured compromise.

**Re-tune when the search changes.** If the evaluation or move ordering improves (so depth starts paying), rerun this grid: the tables above say how much time each step of budget or K costs.

## Milestone 3: the ladder (sampler against heuristic, greedy and cheat)

- Date: 2026-10-19
- Commit: `969d417` (#92), no code changes.
- Budgets: the defaults, a **2-turn horizon** and a shared total of **5,000 nodes per decision** (`cheat.BUDGET`) for both search bots, the sampler splitting it over **K = 4 worlds** (`sampler.WORLDS`, 1,250 nodes each). The K sweep keeps the 5,000-node total and changes only K. Greedy and heuristic don't search.
- Machine: Apple M1 MacBook, 8 GB (4 performance and 4 efficiency cores), macOS (Darwin 25.2), CPython 3.14.0.

500 games per pairing, seeds 0-499, alternating seats, max 300 turns, `--jobs 3`. Each pairing ran as five runs of 100 consecutive seeds (0-99, ..., 400-499), one after another in the foreground with nothing else running, combined with `tournament.combine` (each run equals `tournament ... --games 100 --base-seed <first seed> --jobs 3`; results are per seed, so the split doesn't change wins, seats or turns, and the seeds 0-99 run of sampler vs heuristic reproduced #88's 50/100 exactly). Wall times are the five runs' sum. ms/move is contended by the three processes, as in the Milestone 3 budget tournaments: the sampler measured 101 ms/decision alone in real games (Milestone 3, Decisions in real games) and 125-158 ms here. No game in any pairing was unfinished.

Intervals are 95% **Wilson** score intervals on the win rate of the finished games. With 500 games a win rate has an interval of about ±4.4 points, and one seat's 250 games about ±6.2.

### sampler vs heuristic

`python -m minimax_games.games.star_realms tournament --bots sampler heuristic --games 500 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| sampler | 263/500 | 52.6% | 145/250 | 118/250 | 137 |
| heuristic | 237/500 | 47.4% | 132/250 | 105/250 | 0.0070 |

500 games, 0 unfinished, mean 25.5 turns, 32,492 sampler decisions (65 per game), 1.2 hours of sampler thinking time, 25.6 minutes of wall time. The first player won 277/500 (55.4%). The sampler's win rate: 52.6%, 95% interval **48.2-56.9%**.

### sampler vs greedy

`python -m minimax_games.games.star_realms tournament --bots sampler greedy --games 500 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| sampler | 407/500 | 81.4% | 213/250 | 194/250 | 143 |
| greedy | 93/500 | 18.6% | 56/250 | 37/250 | 0.0038 |

500 games, 0 unfinished, mean 27.9 turns, 37,688 sampler decisions (75 per game), 1.5 hours of sampler thinking time, 31.2 minutes of wall time. The first player won 269/500 (53.8%). The sampler's win rate: 81.4%, 95% interval 77.8-84.6%.

### sampler vs cheat

`python -m minimax_games.games.star_realms tournament --bots sampler cheat --games 500 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| sampler | 224/500 | 44.8% | 123/250 | 101/250 | 125 |
| cheat | 276/500 | 55.2% | 149/250 | 127/250 | 86.5 |

500 games, 0 unfinished, mean 23.2 turns, 28,039 sampler decisions (56 per game) and 27,602 cheat decisions (55 per game), 0.97 and 0.66 hours of thinking time, 33.8 minutes of wall time. The first player won 272/500 (54.4%). The sampler's win rate: 44.8%, 95% interval **40.5-49.2%**.

### K sweep against the heuristic

The same 5,000-node total, split over K worlds: `python -m minimax_games.games.star_realms tournament --bots sampler heuristic --games 500 --jobs 3 --worlds K`. K = 4 is the main ladder's pairing above. All three ran the full 500 games (no departure).

| K | Nodes per world | Wins | Win% | 95% interval | As 1st | As 2nd | ms/move | Mean turns | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5,000 | 237/500 | 47.4% | 43.1-51.8% | 131/250 | 106/250 | 108 | 25.2 | 20.2 min |
| **4** | 1,250 | 263/500 | 52.6% | 48.2-56.9% | 145/250 | 118/250 | 137 | 25.5 | 25.6 min |
| 8 | 625 | 250/500 | 50.0% | 45.6-54.4% | 136/250 | 114/250 | 158 | 25.4 | 29.6 min |

No game was unfinished; the first player won 275/500 (K=1) and 272/500 (K=8). Raw outputs (not committed): `sampler-heuristic-500.txt`, `sampler-greedy-500.txt`, `sampler-cheat-500.txt`, `sampler-k1-heuristic-500.txt`, `sampler-k8-heuristic-500.txt`, and the per-run `*-<first seed>.txt`.

### Reading the ladder

**cheat ≈ sampler ≈ heuristic > greedy, and the cheat beats the sampler head to head.**

- **The sampler is a legitimate bot, but it does not clearly beat the heuristic.** 52.6% (263/500), interval 48.2-56.9%, includes 50% (two-sided p ≈ 0.24, normal approximation): not significant at 5%. It is where the cheat was against the heuristic in Milestone 2 (268/500, 53.6%, interval 49.2-57.9%, also not significant), so seeing only its view costs the sampler nothing measurable against the heuristic.
- **Against greedy** the sampler wins 81.4%, level with the cheat (82.2%) and a little under the heuristic (85.4%): all three intervals overlap.
- **Head to head the cheat is better**: the sampler wins 44.8%, interval 40.5-49.2%, which excludes 50% (p ≈ 0.02). Hidden information costs the sampler about 5 points against its upper bound at equal compute, and it loses in both seats (49.2% first, 40.4% second). These games are the shortest of the ladder (23.2 turns), the two searchers attacking each other harder than the baselines do.
- **K barely matters at 5,000 nodes.** K = 1 (the cheat's search on one sampled world) wins 47.4%, K = 4 52.6% and K = 8 50.0%; every interval overlaps the others and includes 50%. K = 4's 5.2-point lead over K = 1 is suggestive (z ≈ 1.6 for two independent 500-game rates) but not significant. Each world costs time: 108, 137 and 158 ms per decision at K = 1, 4, 8. K = 4 stays the default.

**What holds the search back is the evaluation, not the search or the sampling.** The cheat, which sees everything, beats the heuristic by only 53.6%; doubling or quadrupling its budget did not help (Milestone 3, Tournaments against the heuristic); and the sampler reaches the same result from views alone. A search that looks 2 turns ahead with `evaluate` plays about as well as the heuristic's hand-written buying policy and no better, so a stronger bot needs a better `evaluate` before it needs more nodes, more worlds or a different search.

**What milestones 4 and 5 should compare against.**

- **Milestone 4 (card slices)**: the regression check for each slice is this ladder at these settings and seeds: sampler vs heuristic (52.6%), sampler vs greedy (81.4%), heuristic vs greedy (85.4%, Milestone 2). The new cards change the game, so the numbers will move; a slice regresses if a search bot falls clearly below the heuristic, not if it fails to beat it.
- **Milestone 5 (ISMCTS, rollout `evaluate`)**: the bar for a legitimate bot stays the **heuristic**, and the result to beat is the **sampler** at K = 4, 5,000 nodes (both at equal budgets, as the design says). Try the rollout-based `evaluate` (and tuned `Weights`) in the **cheat** first, against the heuristic: the cheat is the upper bound, so an evaluation that doesn't lift the cheat clearly above 53.6% won't lift the sampler or ISMCTS either. If it does, rerun the sampler and the K sweep with it, since deeper and better-informed searches may finally make depth and more worlds pay.

## Milestone 4: opponent discards

- Date: 2026-10-19
- Commit: the `issue-95-discard` branch (#95).
- **The deck changed.** The playable Trade Deck is now every card whose effects the rules all implement (`catalogue.PLAYABLE_TRADE_DECK`): the 30-card first slice plus the opponent-discard cards, **40 cards**: 3 Imperial Fighter, 3 Imperial Frigate, 3 Survey Ship and 1 Royal Redoubt (Battlecruiser waits for destroy-base). Every earlier result in this file was measured on the 30-card first slice, so they don't compare one to one with these.
- Budgets: the cheat at its defaults, a **2-turn horizon** and **5,000 nodes per decision**. Greedy and heuristic don't search.
- Machine: Apple M1 MacBook, 8 GB (4 performance and 4 efficiency cores), macOS (Darwin 25.2), CPython 3.14.0.

The regression check for a card slice: 500 games each of heuristic vs greedy and greedy vs random, and 100 games of cheat vs greedy, seeds from 0, alternating seats, max 300 turns, `--jobs 3`, each run alone in the foreground. Pass bar: no crashes, no unfinished games, and the ladder order holds. Intervals are 95% Wilson score intervals.

### heuristic vs greedy

`python -m minimax_games.games.star_realms tournament --bots heuristic greedy --games 500 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| heuristic | 405/500 | 81.0% | 216/250 | 189/250 | 0.0054 |
| greedy | 95/500 | 19.0% | 61/250 | 34/250 | 0.0029 |

500 games, 0 unfinished, mean 23.1 turns, 0.9 s wall time. The first player won 277/500 (55.4%). The heuristic's win rate: 81.0%, interval 77.3-84.2% (Milestone 2, first slice: 85.4%, 82.0-88.2%).

### greedy vs random

`python -m minimax_games.games.star_realms tournament --bots greedy random --games 500 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| greedy | 500/500 | 100.0% | 250/250 | 250/250 | 0.0033 |
| random | 0/500 | 0.0% | 0/250 | 0/250 | 0.0004 |

500 games, 0 unfinished, mean 21.8 turns, 0.7 s wall time.

### cheat vs greedy

`python -m minimax_games.games.star_realms tournament --bots cheat greedy --games 100 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| cheat | 80/100 | 80.0% | 38/50 | 42/50 | 86.4 |
| greedy | 20/100 | 20.0% | 8/50 | 12/50 | 0.0041 |

100 games, 0 unfinished, mean 25.5 turns, 190 s wall time. The cheat's win rate: 80.0%, interval 71.1-86.7% (Milestone 2, first slice, 500 games: 82.2%).

### Search speed

The new rules add one check per `legal_actions` and automatic-play scan (is a discard pending?) and a field to `PlayerState`. On the 30-card deck (`new_game`'s default patched back to it), `bench` searches the identical trees as before the change (the same node counts at every position) at the same speed within noise: 32,400-32,700 nodes/s at horizon 1 and 36,500-36,800 at horizon 2 before, 32,600-34,500 and 33,600-37,100 after, over two or three runs each. With the 40-card deck the default `bench` positions are different games (horizon 1: 8,470 nodes, 35,700 nodes/s; horizon 2: 171,501 nodes, 30,600 nodes/s, all 8 searches finished).

### Verdict

**Passes.** No crashes, no unfinished games, and the ladder order holds: heuristic beats greedy (81.0%), greedy beats random (100%), cheat beats greedy (80.0%). The heuristic's lead over greedy is 4.4 points lower than on the first slice, with overlapping intervals; its weights were tuned on the 30-card deck, and its buying policy doesn't value making the opponent discard beyond a card's cost. Random bots no longer stall: seed 13, which stalled on the first slice, finishes, as do seeds 0-299.

Raw outputs (not committed): `heuristic-greedy-500.txt`, `greedy-random-500.txt`, `cheat-greedy-100.txt`, `bench-*.txt`.

## Milestone 4: scrap from hand or discard pile

- Date: 2026-10-19
- Commit: the `issue-96-scrap` branch (#96).
- **The deck changed.** `ScrapHandOrDiscard` and `ScrapFromHand` join `catalogue.IMPLEMENTED`, so the playable Trade Deck grows from 40 to **52 cards**: 3 Trade Bot, 3 Missile Bot, 3 Supply Bot, 1 Battle Mech, 1 Junkyard and 1 Machine Base (Patrol Mech waits for "or" choices). These results compare with the previous slice's ("Milestone 4: opponent discards") only loosely.
- The heuristic now scraps Vipers and Scouts whenever a card lets it (step 1 of `HeuristicBot`'s docstring); greedy never scraps when it may, and makes Machine Base's forced scrap of its cheapest card.
- Budgets, machine and the check itself as in the previous slice: 500 games each of heuristic vs greedy and greedy vs random, 100 of cheat vs greedy at its defaults (2-turn horizon, 5,000 nodes per decision), seeds from 0, alternating seats, max 300 turns, `--jobs 3`, each run alone in the foreground.

### heuristic vs greedy

`python -m minimax_games.games.star_realms tournament --bots heuristic greedy --games 500 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| heuristic | 428/500 | 85.6% | 221/250 | 207/250 | 0.0060 |
| greedy | 72/500 | 14.4% | 43/250 | 29/250 | 0.0033 |

500 games, 0 unfinished, mean 23.0 turns, 1.0 s wall time. The first player won 264/500 (52.8%). The heuristic's win rate: 85.6%, interval 82.3-88.4% (opponent-discards slice: 81.0%, 77.3-84.2%). Scrapping Vipers and Scouts helps it, and greedy buys the Machine Cult cards without using their scraps.

### greedy vs random

`python -m minimax_games.games.star_realms tournament --bots greedy random --games 500 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| greedy | 500/500 | 100.0% | 250/250 | 250/250 | 0.0034 |
| random | 0/500 | 0.0% | 0/250 | 0/250 | 0.0004 |

500 games, 0 unfinished, mean 22.9 turns, 0.9 s wall time.

### cheat vs greedy

`python -m minimax_games.games.star_realms tournament --bots cheat greedy --games 100 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| cheat | 84/100 | 84.0% | 39/50 | 45/50 | 109 |
| greedy | 16/100 | 16.0% | 5/50 | 11/50 | 0.0051 |

100 games, 0 unfinished, mean 25.4 turns, 295 s wall time. The cheat's win rate: 84.0%, interval 75.6-89.9% (opponent-discards slice: 80.0%). Its time per decision rose from 86 to 109 ms: see below.

### Search speed

The rules now check, before playing a pure-gain ship automatically, whether something could scrap it from hand (`rules._hand_scrap_ahead`), and `GameState` has two more fields. The card-level parts of `automatic` are cached per card (a card hashes by its name), which more than pays for the check. On the 40-card deck (`new_game`'s default patched back to it), `bench` searches the identical trees as before the change (8,470 nodes at horizon 1 and 171,501 at horizon 2, the same at every position) at horizon 2 30,100 nodes/s after against 30,400-30,600 before, over two runs each: the same within noise. Without the caching it was 26,200.

With the 52-card deck the trees are bigger: while a scrap may be made, every decision also offers scrapping each distinct card in hand and in the discard pile. `bench` (horizons 1 and 2, default positions): horizon 1 7/8 searches finished within 200,000 nodes (seed 2 turn 21's one-turn tree passes it), 224,313 nodes at 55,500 nodes/s; horizon 2 7/8, 342,851 nodes at 47,100 nodes/s. `bench --budget 5000`: the cheat's decisions finished 2 turns at 2/8 positions, 1 turn at 3/8 and none at 3/8 (opponent-discards deck: 4/8, 4/8 and 0/8), 90.6 ms per decision. Scrapping a card from the discard pile commutes with every action but a draw and a buy, so canonicalising when it happens would shrink these trees, if search depth turns out to limit strength.

### Verdict

**Passes.** No crashes, no unfinished games, and the ladder order holds: heuristic beats greedy (85.6%), greedy beats random (100%), cheat beats greedy (84.0%). Random players can now scrap every card they own, which can deadlock a random-vs-random game (neither can deal damage); a sampled-world test met one and now accepts that dead position. Greedy never scraps when it may, so the tournaments above can't deadlock.

Raw outputs (not committed): `scrap/heuristic-greedy-500.txt`, `scrap/greedy-random-500.txt`, `scrap/cheat-greedy-100.txt`, `scrap/bench-new-deck.txt`.

## Milestone 4: destroy target base

- Date: 2026-10-19
- Commit: the `issue-97-destroy` branch (#97).
- **The deck changed.** `DestroyBase` joins `catalogue.IMPLEMENTED`, so the playable Trade Deck grows from 52 to **56 cards**: 1 Command Ship, 1 Port of Call, 1 Battlecruiser and 1 Missile Mech (Blob Destroyer waits for scrapping from the Trade Row). These results compare with the previous slice's ("Milestone 4: scrap from hand or discard pile") only loosely.
- The heuristic now uses each destroy-base ability at once on the opponent's base with the most defense (step 3 of `HeuristicBot`'s docstring); greedy never uses them.
- Budgets, machine and the check itself as in the previous slices: 500 games each of heuristic vs greedy and greedy vs random, 100 of cheat vs greedy at its defaults (2-turn horizon, 5,000 nodes per decision), seeds from 0, alternating seats, max 300 turns, `--jobs 3`, each run alone in the foreground.

### heuristic vs greedy

`python -m minimax_games.games.star_realms tournament --bots heuristic greedy --games 500 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| heuristic | 443/500 | 88.6% | 231/250 | 212/250 | 0.0060 |
| greedy | 57/500 | 11.4% | 38/250 | 19/250 | 0.0033 |

500 games, 0 unfinished, mean 22.9 turns, 1.0 s wall time. The first player won 269/500 (53.8%). The heuristic's win rate: 88.6%, interval 85.5-91.1% (scrap slice: 85.6%, 82.3-88.4%). Greedy buys the new cards without using their destroy-base abilities.

### greedy vs random

`python -m minimax_games.games.star_realms tournament --bots greedy random --games 500 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| greedy | 500/500 | 100.0% | 250/250 | 250/250 | 0.0034 |
| random | 0/500 | 0.0% | 0/250 | 0/250 | 0.0004 |

500 games, 0 unfinished, mean 23.8 turns, 0.9 s wall time.

### cheat vs greedy

`python -m minimax_games.games.star_realms tournament --bots cheat greedy --games 100 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| cheat | 85/100 | 85.0% | 46/50 | 39/50 | 95.4 |
| greedy | 15/100 | 15.0% | 11/50 | 4/50 | 0.0047 |

100 games, 0 unfinished, mean 24.8 turns, 255 s wall time. The cheat's win rate: 85.0%, interval 76.7-90.7% (scrap slice: 84.0%).

### Search speed

The rules add one check per `legal_actions` (may a base be destroyed?), a field to `GameState`, and `DestroyBase` to the pure-gain effects. On the 52-card deck (`new_game`'s default patched back to it), `bench` searches the identical trees as before the change (224,313 nodes at horizon 1 and 342,851 at horizon 2, 7/8 searches finished at each) at the same speed: 57,000 and 47,100-47,200 nodes/s after, 56,900-57,700 and 47,400-47,600 before, over two runs each.

With the 56-card deck the default `bench` positions are different games: horizon 1 7/8 searches finished within 200,000 nodes, 381,301 nodes at 51,700 nodes/s; horizon 2 5/8, 648,781 nodes at 44,300 nodes/s. `bench --budget 5000`: the cheat's decisions finished 2 turns at 3/8 positions, 1 turn at 3/8 and none at 2/8 (scrap deck: 2/8, 3/8 and 3/8), 73.8 ms per decision. A destroy adds at most one branch per distinct opponent base, and only while one may be made.

### Verdict

**Passes.** No crashes, no unfinished games, and the ladder order holds: heuristic beats greedy (88.6%), greedy beats random (100%), cheat beats greedy (85.0%).

Raw outputs (not committed): `destroy/heuristic-greedy-500.txt`, `destroy/greedy-random-500.txt`, `destroy/cheat-greedy-100.txt`, `destroy/bench-new-deck.txt`, `destroy/bench-budget-new-deck.txt`.

## Milestone 4: scrap from the Trade Row

- Date: 2026-10-19
- Commit: the `issue-98-trade-row` branch (#98).
- **The deck changed.** `ScrapTradeRow` joins `catalogue.IMPLEMENTED`, so the playable Trade Deck grows from 56 to **60 cards**: 2 Battle Pods and 2 Blob Destroyers. These results compare with the previous slice's ("Milestone 4: destroy target base") only loosely.
- The heuristic now scraps from the Trade Row, after buying, the card the opponent would most want (its buying worth, for the opponent's cards and their next turn), unless the average card left in the Trade Deck is worth more to them (step 5 of `HeuristicBot`'s docstring); greedy never uses the ability.
- Budgets, machine and the check itself as in the previous slices: 500 games each of heuristic vs greedy and greedy vs random, 100 of cheat vs greedy at its defaults (2-turn horizon, 5,000 nodes per decision), seeds from 0, alternating seats, max 300 turns, `--jobs 3`, each run alone in the foreground.

### heuristic vs greedy

`python -m minimax_games.games.star_realms tournament --bots heuristic greedy --games 500 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| heuristic | 418/500 | 83.6% | 215/250 | 203/250 | 0.0073 |
| greedy | 82/500 | 16.4% | 47/250 | 35/250 | 0.0033 |

500 games, 0 unfinished, mean 21.8 turns, 1.0 s wall time. The heuristic's win rate: 83.6%, interval 80.1-86.6% (destroy-base slice: 88.6%, 85.5-91.1%). The drop comes with the deck, not the new policy: with the Trade Row denial switched off the heuristic wins 420/500 (84.0%) on the same seeds, and on seeds 1000-2999 it wins 1,690/2,000 (84.5%) with it and 1,698/2,000 (84.9%) without. Greedy buys Battle Pods and Blob Destroyers, cheap-for-their-combat Blob ships, without scrapping from the row. Scrapping before buying (never a card it could afford) made no difference either (1,695/2,000), so the simpler policy stays.

### greedy vs random

`python -m minimax_games.games.star_realms tournament --bots greedy random --games 500 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| greedy | 500/500 | 100.0% | 250/250 | 250/250 | 0.0035 |
| random | 0/500 | 0.0% | 0/250 | 0/250 | 0.0004 |

500 games, 0 unfinished, mean 22.7 turns, 0.8 s wall time.

### cheat vs greedy

`python -m minimax_games.games.star_realms tournament --bots cheat greedy --games 100 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| cheat | 80/100 | 80.0% | 39/50 | 41/50 | 97.8 |
| greedy | 20/100 | 20.0% | 9/50 | 11/50 | 0.0046 |

100 games, 0 unfinished, mean 23.5 turns, 237 s wall time. The cheat's win rate: 80.0%, interval 71.1-86.7% (destroy-base slice: 85.0%, 76.7-90.7%).

### Search speed

The rules add one check per `legal_actions` (may the player scrap from the row?) and a field to `GameState`, and `ScrapTradeRow` joins the pure-gain effects; `Acquire` and the new action share the refill (`rules._take_from_row`). On the 56-card deck (`new_game`'s default patched back to it), `bench` searches the identical trees as before the change (381,301 nodes at horizon 1 and 648,781 at horizon 2) at much the same speed: 50,500-50,800 and 43,500-43,700 nodes/s after, 51,100-52,000 and 44,500-44,900 before, over two runs each (about 1-2% slower, the extra state field).

With the 60-card deck the default `bench` positions are different games: horizon 1 6/8 searches finished within 200,000 nodes, 446,773 nodes at 36,700 nodes/s; horizon 2 3/8, 1,016,157 nodes at 40,400 nodes/s. `bench --budget 5000`: the cheat's decisions finished 2 turns at 2/8 positions, 1 turn at 1/8 and none at 5/8 (destroy-base deck: 3/8, 3/8 and 2/8), 113.8 ms per decision. The loss is the new positions, not the new branching: with Trade Row scraps never granted (`_resolve` patched to ignore `ScrapTradeRow`; greedy, which sets the positions up, never uses them) the same positions give the same table, 2/8, 1/8 and 5/8, and only seed 2 turn 5 needs more nodes with them (4,670 instead of 1,159, still finishing 2 turns). A row scrap adds at most one branch per distinct row card, and only while one may be made.

### Verdict

**Passes.** No crashes, no unfinished games, and the ladder order holds: heuristic beats greedy (83.6%), greedy beats random (100%), cheat beats greedy (80.0%).

Raw outputs (not committed): `rowscrap/heuristic-greedy-500.txt`, `rowscrap/greedy-random-500.txt`, `rowscrap/cheat-greedy-100.txt`, `rowscrap/bench-before-{1,2}.txt`, `rowscrap/bench-after-{1,2}.txt`, `rowscrap/bench-new-deck.txt`, `rowscrap/bench-budget-new-deck.txt`.

## Milestone 4: "or" choices

- Date: 2026-10-19
- Commit: the `issue-99-choice` branch (#99).
- **The deck changed.** `Choice`, `DrawPerFactionCard` and `DiscardThenDraw` join `catalogue.IMPLEMENTED`, so the playable Trade Deck grows from 60 to **70 cards**: 2 Trading Posts, 2 Barter Worlds, Defense Center, 2 Patrol Mechs, Blob World and 2 Recycling Stations. These results compare with the previous slice's ("Milestone 4: scrap from the Trade Row") only loosely.
- Using a choice between pure gains is automatic (Patrol Mech, and Trading Post, Barter World and Defense Center on later turns); the option is a forced `Choose` decision. Greedy takes the option with the most trade plus combat; the heuristic takes trade early, combat late, Authority at 10 or less, draws with Blob World once that draws 3 or more cards, and recycles Vipers and Scouts with Recycling Station while it holds any (step 0 of `HeuristicBot`'s docstring).
- Budgets, machine and the check itself as in the previous slices: 500 games each of heuristic vs greedy and greedy vs random, 100 of cheat vs greedy at its defaults (2-turn horizon, 5,000 nodes per decision), seeds from 0, alternating seats, max 300 turns, `--jobs 3`, each run alone in the foreground.

### heuristic vs greedy

`python -m minimax_games.games.star_realms tournament --bots heuristic greedy --games 500 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| heuristic | 448/500 | 89.6% | 221/250 | 227/250 | 0.0088 |
| greedy | 52/500 | 10.4% | 23/250 | 29/250 | 0.0041 |

500 games, 0 unfinished, mean 22.9 turns, 1.1 s wall time. The heuristic's win rate: 89.6%, interval 86.6-92.0% (Trade Row slice: 83.6%, 80.1-86.6%).

### greedy vs random

`python -m minimax_games.games.star_realms tournament --bots greedy random --games 500 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| greedy | 500/500 | 100.0% | 250/250 | 250/250 | 0.0044 |
| random | 0/500 | 0.0% | 0/250 | 0/250 | 0.0005 |

500 games, 0 unfinished, mean 22.7 turns, 1.0 s wall time.

### cheat vs greedy

`python -m minimax_games.games.star_realms tournament --bots cheat greedy --games 100 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| cheat | 74/100 | 74.0% | 36/50 | 38/50 | 109 |
| greedy | 26/100 | 26.0% | 12/50 | 14/50 | 0.0062 |

100 games, 0 unfinished, mean 24.9 turns, 281 s wall time. The cheat's win rate: 74.0%, interval 64.6-81.6% (Trade Row slice: 80.0%, 71.1-86.7%): lower, but the intervals overlap widely.

### Search speed

The rules add three fields to `GameState` (`choosing`, `recycling`, `factions_played`), a per-faction count updated by every `PlayCard`, and two checks in `_forced`; `Choice` joins the pure-gain effects when its options are pure gain. The per-card checks of which abilities are implemented (`_scrappable`, `_ally_usable`, `_primary_usable`) are now cached by card, as a `Choice` makes them recurse. On the 60-card deck (`new_game`'s default patched back to it), `bench` searches the identical trees as before the change (446,773 nodes at horizon 1 and 1,016,157 at horizon 2): 35,400-35,600 and 38,900-39,000 nodes/s after, 37,000 and 40,800 before, over two runs each (about 4-5% slower: building and hashing the bigger state).

With the 70-card deck the default `bench` positions are different games: horizon 1 7/8 searches finished within 200,000 nodes, 222,942 nodes at 33,400 nodes/s; horizon 2 6/8, 498,923 nodes at 41,600 nodes/s. `bench --budget 5000`: the cheat's decisions finished 2 turns at 4/8 positions, 1 turn at 2/8 and none at 2/8 (Trade Row deck: 2/8, 1/8 and 5/8), 87.0 ms per decision. A choice adds one decision node with one branch per option, and only where the card is used; the plays it makes automatic don't branch on timing.

### Verdict

**Passes.** No crashes, no unfinished games, and the ladder order holds: heuristic beats greedy (89.6%), greedy beats random (100%), cheat beats greedy (74.0%).

Raw outputs (not committed): `choice/heuristic-greedy-500.txt`, `choice/greedy-random-500.txt`, `choice/cheat-greedy-100.txt`, `choice/bench-before-{1,2}.txt`, `choice/bench-after-{1,2}.txt`, `choice/bench-new-deck.txt`, `choice/bench-budget-new-deck.txt`.

## Milestone 4: next ship on top of the deck

- Date: 2026-10-20
- Commit: the `issue-100-on-top` branch (#100).
- **The deck changed.** `NextShipOnTop` and `AcquireShipFreeOnTop` join `catalogue.IMPLEMENTED`, so the playable Trade Deck grows from 70 to **74 cards**: 2 Freighters, Central Office and Blob Carrier. These results compare with the previous slice's ("Milestone 4: "or" choices") only loosely.
- `NextShipOnTop` (Freighter's ally ability, Central Office) is a "may" ability granted automatically (`GameState.may_on_top`); the player then chooses `AcquireOnTop` over `Acquire` for any ship bought later in the turn. Blob Carrier's ally ability is a decision, and its free ship a forced `AcquireFreeShip` choice. A card put on top of a deck is public until drawn (`PlayerView.deck_top`). Both baselines put the ship they buy on top when they may; greedy takes the most expensive free ship and the heuristic the one it would most want to buy (`GreedyBot` and `HeuristicBot` docstrings).
- Budgets, machine and the check itself as in the previous slices: 500 games each of heuristic vs greedy and greedy vs random, 100 of cheat vs greedy at its defaults (2-turn horizon, 5,000 nodes per decision), seeds from 0, alternating seats, max 300 turns, `--jobs 3`, each run alone in the foreground.

### heuristic vs greedy

`python -m minimax_games.games.star_realms tournament --bots heuristic greedy --games 500 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| heuristic | 451/500 | 90.2% | 230/250 | 221/250 | 0.0099 |
| greedy | 49/500 | 9.8% | 29/250 | 20/250 | 0.0051 |

500 games, 0 unfinished, mean 23.1 turns, 1.3 s wall time. The heuristic's win rate: 90.2%, interval 87.3-92.5% ("or" choices slice: 89.6%, 86.6-92.0%).

### greedy vs random

`python -m minimax_games.games.star_realms tournament --bots greedy random --games 500 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| greedy | 500/500 | 100.0% | 250/250 | 250/250 | 0.0056 |
| random | 0/500 | 0.0% | 0/250 | 0/250 | 0.0005 |

500 games, 0 unfinished, mean 22.9 turns, 1.1 s wall time.

### cheat vs greedy

`python -m minimax_games.games.star_realms tournament --bots cheat greedy --games 100 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| cheat | 77/100 | 77.0% | 38/50 | 39/50 | 108 |
| greedy | 23/100 | 23.0% | 11/50 | 12/50 | 0.0068 |

100 games, 0 unfinished, mean 24.8 turns, 277 s wall time. The cheat's win rate: 77.0%, interval 67.8-84.2% ("or" choices slice: 74.0%, 64.6-81.6%).

### Search speed

The rules add `may_on_top` and `free_ships` to `GameState`, `known_top` to `PlayerState` (updated by every draw), a check in `_forced`, and the `AcquireOnTop` actions while `may_on_top` is positive. On the 70-card deck (`new_game`'s default patched back to it), `bench` searches the identical trees as before the change (222,942 nodes at horizon 1 and 498,923 at horizon 2): 32,300-32,400 and 40,100-40,200 nodes/s after, 33,700-33,800 and 41,600-41,700 before, over two runs each (about 4% slower: building and hashing the bigger state).

With the 74-card deck the default `bench` positions are different games: horizon 1 5/8 searches finished within 200,000 nodes, 609,304 nodes at 38,300 nodes/s; horizon 2 4/8, 895,044 nodes at 39,400 nodes/s. `bench --budget 5000`: the cheat's decisions finished 2 turns at 1/8 positions, 1 turn at 4/8 and none at 3/8 ("or" choices deck, on this branch too: 4/8, 2/8 and 2/8), 118.7 ms per decision. The new branching isn't the cause: with `NextShipOnTop` patched to do nothing, the same eight decisions search exactly the same nodes (the positions' trees never reach an `AcquireOnTop`), so the drop is the new deck's different games, as when each earlier slice changed the deck. `AcquireOnTop` adds one action per affordable ship, only in a turn after Freighter's ally ability or Central Office is used, and Blob Carrier's free ship one decision node with a branch per ship in the row.

### Verdict

**Passes.** No crashes, no unfinished games, and the ladder order holds: heuristic beats greedy (90.2%), greedy beats random (100%), cheat beats greedy (77.0%).

Raw outputs (not committed): `ontop/heuristic-greedy-500.txt`, `ontop/greedy-random-500.txt`, `ontop/cheat-greedy-100.txt`, `ontop/bench-before-{1,2}.txt`, `ontop/bench-after-{1,2}.txt`, `ontop/bench-new-deck.txt`, `ontop/bench-budget-new-deck.txt`, `ontop/bench-budget-old-deck.txt`.

## Milestone 4: the remaining one-off abilities

- Date: 2026-10-20
- Commit: the `issue-101-one-offs` branch (#101).
- **The deck changed, for the last time.** `DrawIfBases`, `ScrapThenDraw`, `CopyShip`, `ShipsGainCombat` and `AllyForAllFactions` join `catalogue.IMPLEMENTED`, so the playable Trade Deck grows from 74 cards to the whole base set, **80 cards**: 2 Embassy Yachts, Brain World, Stealth Needle, Fleet HQ and Mech World. These results compare with the previous slice's ("Milestone 4: next ship on top of the deck") only loosely.
- Embassy Yacht draws 2 when its owner has 2 bases in play, and is played automatically while it can't. Brain World's scraps are a forced `ScrapAndDraw` choice; Stealth Needle copies a ship in play through a forced `Copy`, becoming a copy of it (faction and abilities, plus Machine Cult) until it leaves play. Fleet HQ (+1 combat per ship played) and Mech World (an ally for every faction) are always on; ships wait in hand while a Fleet HQ is there. Greedy copies the ship with the most trade plus combat and never scraps with Brain World; the heuristic copies the ship it would least want to give up and scraps up to two Vipers and Scouts with Brain World, from the discard pile first (`GreedyBot` and `HeuristicBot` docstrings).
- Budgets, machine and the check itself as in the previous slices: 500 games each of heuristic vs greedy and greedy vs random, 100 of cheat vs greedy at its defaults (2-turn horizon, 5,000 nodes per decision), seeds from 0, alternating seats, max 300 turns, `--jobs 3`, each run alone in the foreground.

### heuristic vs greedy

`python -m minimax_games.games.star_realms tournament --bots heuristic greedy --games 500 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| heuristic | 453/500 | 90.6% | 227/250 | 226/250 | 0.0106 |
| greedy | 47/500 | 9.4% | 24/250 | 23/250 | 0.0064 |

500 games, 0 unfinished, mean 23.9 turns, 1.3 s wall time. The heuristic's win rate: 90.6%, interval 87.7-92.9% (next-ship-on-top slice: 90.2%, 87.3-92.5%).

### greedy vs random

`python -m minimax_games.games.star_realms tournament --bots greedy random --games 500 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| greedy | 500/500 | 100.0% | 250/250 | 250/250 | 0.0070 |
| random | 0/500 | 0.0% | 0/250 | 0/250 | 0.0005 |

500 games, 0 unfinished, mean 23.9 turns, 1.1 s wall time.

Random against random can now stall: with Brain World beside Machine Base, Junkyard and the other scrappers, random bots sometimes scrap nearly every card they own, leaving neither able to win (seeds 51, 109, 131, 166, 254 and 274 of 0-299 pass 1,000 turns; the match tests use seeds 0-49, which all finish). No baseline does that.

### cheat vs greedy

`python -m minimax_games.games.star_realms tournament --bots cheat greedy --games 100 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| cheat | 76/100 | 76.0% | 37/50 | 39/50 | 116 |
| greedy | 24/100 | 24.0% | 11/50 | 13/50 | 0.0093 |

100 games, 0 unfinished, mean 26.2 turns, 330 s wall time. The cheat's win rate: 76.0%, interval 66.8-83.3% (next-ship-on-top slice: 77.0%, 67.8-84.2%).

### Search speed

The rules add `scrap_to_draw` and `copying` to `GameState`, a Fleet HQ check before a ship is played automatically, Mech World and the Needle's copies in the ally check (`_allied`), and a Fleet HQ bonus when a ship is played. On the 74-card deck (`new_game`'s default patched back to it), `bench` searches the identical trees as before the change (609,304 nodes at horizon 1 and 895,044 at horizon 2): 36,700-36,900 and 38,200-38,300 nodes/s after, 38,100-38,400 and 38,900-39,200 before, over two runs each (about 2-4% slower). A first version was 10% slower: looking cards up by `Card` (whose hash is a Python call) in the per-action checks; the Fleet HQ, Stealth Needle and Mech World checks now look up names instead.

With the 80-card deck the default `bench` positions are different games: horizon 1 5/8 searches finished within 200,000 nodes, 632,643 nodes at 46,100 nodes/s; horizon 2 4/8, 966,570 nodes at 37,600 nodes/s. `bench --budget 5000`: the cheat's decisions finished 2 turns at 2/8 positions, 1 turn at 2/8 and none at 4/8 (74-card deck, on this branch: 1/8, 4/8 and 3/8), 101.8 ms per decision. The new decisions are a `Copy` node with a branch per other ship in play and a `ScrapAndDraw` node with a branch per choice of up to two cards from hand and discard pile (dozens with a long discard pile), each only when Stealth Needle or Brain World is used; Embassy Yacht stays automatic unless it can draw.

### Verdict

**Passes.** No crashes, no unfinished games, and the ladder order holds: heuristic beats greedy (90.6%), greedy beats random (100%), cheat beats greedy (76.0%). Every base-set card is now in play.

Raw outputs (not committed): `oneoffs/heuristic-greedy-500.txt`, `oneoffs/greedy-random-500.txt`, `oneoffs/cheat-greedy-100.txt`, `oneoffs/bench-before-{1,2}.txt`, `oneoffs/bench-after-{1,2}.txt`, `oneoffs/bench-new-deck.txt`, `oneoffs/bench-budget-new-deck.txt`, `oneoffs/bench-budget-old-deck.txt`.

## Milestone 4: the ladder (the full base set)

- Date: 2026-10-20
- Commit: `9cb075b` (#111), no code changes.
- **The deck**: the whole 80-card base set (`catalogue.PLAYABLE_TRADE_DECK`), where the milestone-3 ladder and the milestone-2 ladder it compares with played the 30-card first slice. Every bot was taught the new decisions by its card slice (#95-#101, the "Milestone 4" sections above); `evaluate` was not changed.
- Budgets: the defaults, as in the milestone-3 ladder: a **2-turn horizon** and a shared total of **5,000 nodes per decision** (`cheat.BUDGET`) for both search bots, the sampler splitting it over **K = 4 worlds** (`sampler.WORLDS`, 1,250 nodes each). Greedy and heuristic don't search.
- Machine: Apple M1 MacBook, 8 GB (4 performance and 4 efficiency cores), macOS (Darwin 25.2), CPython 3.14.0.

500 games per pairing, seeds 0-499, alternating seats, max 300 turns, `--jobs 3`. As in the milestone-3 ladder, each search-bot pairing ran as five runs of 100 consecutive seeds (0-99, ..., 400-499), one after another in the foreground with nothing else running, combined with `tournament.combine` (each run equals `tournament ... --games 100 --base-seed <first seed> --jobs 3`; results are per seed, so the split doesn't change wins, seats or turns); heuristic vs greedy ran as one. Wall times are the five runs' sum. **ms/move is contended by the three processes**: the cheat measured 94-102 ms per decision alone at the budgeted benchmark positions below, and 97-122 ms here. All five pairings ran the full 500 games (no departure). No game in any pairing was unfinished.

Intervals are 95% **Wilson** score intervals on the win rate of the finished games; p-values are two-sided, normal approximation. With 500 games a win rate has an interval of about ±4.4 points, one seat's 250 games about ±6.2.

### heuristic vs greedy

`python -m minimax_games.games.star_realms tournament --bots heuristic greedy --games 500 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| heuristic | 453/500 | 90.6% | 227/250 | 226/250 | 0.0104 |
| greedy | 47/500 | 9.4% | 24/250 | 23/250 | 0.0063 |

500 games, 0 unfinished, mean 23.9 turns, 1.3 s wall time. The first player won 251/500 (50.2%). The heuristic's win rate: 90.6%, interval 87.7-92.9% (the same games as the last card slice's check).

### cheat vs heuristic

`python -m minimax_games.games.star_realms tournament --bots cheat heuristic --games 500 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| cheat | 164/500 | 32.8% | 96/250 | 68/250 | 118 |
| heuristic | 336/500 | 67.2% | 182/250 | 154/250 | 0.0130 |

500 games, 0 unfinished, mean 24.3 turns, 33,712 cheat decisions (67 per game), 1.1 hours of cheat thinking time, 23.2 minutes of wall time. The first player won 278/500 (55.6%). The cheat's win rate: 32.8%, interval **28.8-37.0%**.

### sampler vs heuristic

`python -m minimax_games.games.star_realms tournament --bots sampler heuristic --games 500 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| sampler | 136/500 | 27.2% | 80/250 | 56/250 | 138 |
| heuristic | 364/500 | 72.8% | 194/250 | 170/250 | 0.0132 |

500 games, 0 unfinished, mean 24.0 turns, 34,131 sampler decisions (68 per game), 1.3 hours of sampler thinking time, 27.9 minutes of wall time. The first player won 274/500 (54.8%). The sampler's win rate: 27.2%, interval **23.5-31.3%**.

### sampler vs greedy

`python -m minimax_games.games.star_realms tournament --bots sampler greedy --games 500 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| sampler | 312/500 | 62.4% | 167/250 | 145/250 | 152 |
| greedy | 188/500 | 37.6% | 105/250 | 83/250 | 0.0102 |

500 games, 0 unfinished, mean 27.2 turns, 41,854 sampler decisions (84 per game), 1.8 hours of sampler thinking time, 37.7 minutes of wall time. The first player won 272/500 (54.4%). The sampler's win rate: 62.4%, interval 58.1-66.5%.

### sampler vs cheat

`python -m minimax_games.games.star_realms tournament --bots sampler cheat --games 500 --jobs 3`

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| sampler | 200/500 | 40.0% | 122/250 | 78/250 | 135 |
| cheat | 300/500 | 60.0% | 172/250 | 128/250 | 102 |

500 games, 0 unfinished, mean 24.0 turns, 34,111 sampler decisions (68 per game) and 31,845 cheat decisions (64 per game), 1.28 and 0.90 hours of thinking time, 46.0 minutes of wall time. The first player won 294/500 (58.8%). The sampler's win rate: 40.0%, interval **35.8-44.4%**.

Raw outputs (not committed): `m4/results/heuristic-greedy-500.txt`, `cheat-heuristic-500.txt`, `sampler-heuristic-500.txt`, `sampler-greedy-500.txt`, `sampler-cheat-500.txt`, and the per-run `*-<first seed>.txt`.

### Against milestone 3

The same pairings, settings and seeds on the 30-card first slice (heuristic vs greedy and cheat vs heuristic from the milestone-2 ladder, whose budgeted decisions are the same as milestone 3's: the optimisations since changed no result). The change's p-value compares two independent 500-game rates.

| Pairing (first bot's win rate) | 30 cards | 80 cards | Change | p |
|---|---|---|---|---|
| heuristic vs greedy | 85.4% | 90.6% | +5.2 | 0.01 |
| cheat vs heuristic | 53.6% | **32.8%** | -20.8 | < 0.0001 |
| sampler vs heuristic | 52.6% | **27.2%** | -25.4 | < 0.0001 |
| sampler vs greedy | 81.4% | 62.4% | -19.0 | < 0.0001 |
| sampler vs cheat | 44.8% | 40.0% | -4.8 | 0.12 |

### Search speed on the full deck

`task bench` (`python -m minimax_games.games.star_realms bench`, the cheat's search at horizons 1 and 2, at most 200,000 nodes per search): horizon 1 5/8 searches finished, 632,643 nodes in 13.9 s (45,400 nodes/s); horizon 2 4/8, 966,570 nodes in 26.2 s (36,900 nodes/s): the same trees as the last card slice measured, at the same speed. The positions are the greedy-vs-greedy games of seeds 1 and 2, so on the full deck they are different games from milestone 3's (32,600 and 37,300 nodes/s on the 30-card deck, Milestone 2h).

The budgeted decisions at the same positions, against the milestone-3 table's 5,000-node row (Milestone 3: budget). Each cell: ms per decision, then how many searches finished 2 turns and at least 1 turn (the cheat's 8 decisions; the sampler's 8 × 4 worlds).

| Budget 5,000 | cheat | sampler K=4 |
|---|---|---|
| 30 cards (milestone 3) | 111 ms, 1/8, 6/8 | 118 ms, 4/32, 12/32 |
| 80 cards | 94 ms, 2/8, 4/8 | 103 ms, 10/32, 12/32 |

`bench --budget 5000` and `bench --budget 5000 --worlds 4`; raw outputs `m4/results/bench.txt`, `bench-cheat-5000.txt`, `bench-sampler-5000-4.txt`. On the full deck the turn starts of mid and late game (seed 1 turns 11, 15, 19 and seed 2 turns 13 and 21) finish not even the 1-turn search, for the cheat at four of the eight positions and for every sampler world at five. Closed issue #105 found why: these unfinished searches are dominated by buys and attacks taken in every order (at positions 1:7 and 2:13 the first turn has no scrap at all), so a canonical order for buys and attacks, not pruning the new scrap and discard decisions, is the lever if depth needs improving.

### Reading the ladder

**heuristic >> cheat > sampler > greedy.** On the full deck both search bots fall far behind the heuristic, which gains on greedy.

- **The heuristic got stronger, the search bots weaker.** The heuristic beats greedy 90.6% (85.4% on the first slice; p ≈ 0.01). The cheat, which sees everything, now loses to the heuristic 32.8% to 67.2%, and the sampler 27.2% to 72.8%: about 21 and 25 points below their milestone-3 results, far outside the noise, and in both seats. Against greedy the sampler wins only 62.4% (81.4% before), 28 points below the heuristic's 90.6%. The milestone-3 regression bar, "a slice regresses if a search bot falls clearly below the heuristic", is now failed, but by the milestone as a whole rather than by any one slice: each slice's check (cheat vs greedy, 100 games) stayed between 74% and 85% (drifting from 80-85% in the first four slices to 74-77% in the last three, within each check's ±9-point interval), and none ran a search bot against the heuristic.
- **The gap between the search bots barely moved.** The cheat still beats the sampler, 60.0% to 40.0% (interval 35.8-44.4%, p < 0.0001), about 5 points more than on the first slice (44.8%), a change within noise (p ≈ 0.12). Sampling hidden information still costs the sampler a few points at equal compute, and the same against the heuristic (27.2% against the cheat's 32.8%).
- **The search did not lose depth.** At 5,000 nodes the budgeted benchmark finishes as deep as on the first slice (the cheat 2/8 positions at 2 turns, the sampler 10/32 worlds), and `bench` runs at the same nodes/s. Mid- and late-game turn starts still finish not even 1 turn, as before (#105's finding: buys and attacks in every order).
- **What changed is what `evaluate` can see.** `evaluate` scores a deck by its cards' average **primary trade and combat** only, plus Authority, bases' defense and a starter-card penalty (Milestone 2, Reading the ladder). On the first slice that covered most of the deck: 19 of the 30 cards' primary abilities were only trade and combat, and the rest added only Draw and Authority. On the full deck only 26 of 80 are; 18 have no primary trade or combat at all (Trading Post, Machine Base, Brain World, Fleet HQ, Mech World, Stealth Needle, Junkyard, ...), and the new primaries, scrapping, opponent discards, "or" choices (10 cards, whose trade or combat counts as nothing), destroying bases, Trade Row scraps and copies, are worth nothing to it, as ally abilities always were. So the search buys and plays the new cards by a score that ignores what they do, while the heuristic was taught each one by its slice (which card to scrap, which option to take, what to copy). This reading is from the evaluation's terms, not an ablation: no run isolated it.
- **Random against random can stall** on the full deck: both players scrap nearly everything and neither can win (seeds 51, 109, 131, 166, 254 and 274 of 0-299 pass 1,000 turns; Milestone 4: the remaining one-off abilities). No baseline or search bot does this: every game in this ladder finished, in 24-27 turns on average, as on the first slice.

**What milestone 5 should compare against.**

- The bar for a legitimate bot stays the **heuristic**, now 90.6% against greedy and far above the search bots. The result to beat is the **sampler** at K = 4, 5,000 nodes on the full deck: **27.2%** against the heuristic (interval 23.5-31.3%), 62.4% against greedy and 40.0% against the cheat, at these seeds and seats, with equal budgets for ISMCTS (as the design says).
- **Start with the evaluation, in the cheat.** Milestone 3 already found the evaluation, not the search or the sampling, holds the search bots back; on the full deck that is plainer still. A rollout-based `evaluate` (or one that values the new effects: scraps, draws, discards, choices and ally abilities) should first lift the **cheat** clearly above its **32.8%** against the heuristic, its upper bound; one that doesn't won't lift the sampler or ISMCTS either. Then rerun the sampler, and the budget and K sweeps, since a better evaluation may finally make depth and more worlds pay.
- **Depth, if it matters again**: canonicalising the order of buys and attacks (#105) is the measured lever, not the new cards' branching.

## Milestone 5: card values

- Date: 2026-10-20
- Commit: `586d7b4` (#114): `evaluate`'s deck-economy term is now the average **card value** of the cards a player owns, counting every effect a card has (docs/star-realms.md, Evaluation), where it was the average primary trade and combat. Nothing else changed, so this is an ablation of the evaluation against the milestone-4 ladder.
- The effect values (`evaluate.Weights`, points per unit of the per-card average, as `trade` and `combat`'s 20 were): trade 20, combat 20, Authority 8, draw 30 per card, opponent discards 20 per card, destroy a base 20, scrap from hand or discard 15 per card, Trade Row scrap 6 per card, next ship on top 15, free ship on top 80, draw if bases 15 per card, Blob World's draw per Blob card 45, discard then draw 10 per card, scrap then draw 35 per card, Machine Base's forced scrap 5, copy a ship 40, Fleet HQ 60 per point, Mech World's allies 30. A choice is worth its best option. Ally abilities count times `1 - (1 - share)^4` (`ally_hand = 4`), `share` being the fraction of the player's other cards of the card's faction; scrap abilities count a quarter (`scrap_ability = 0.25`). The base-defense weights and the starter-card penalty are unchanged.
- **The values were chosen by hand, not tuned.** One check ran before the measurement, on seeds outside it: cheat vs heuristic, 100 games at seeds 1000-1099, 44/100 (44.0%). The values were not changed after it, and none were tried on seeds 0-499. Tuning is #115, on its own seeds.
- Budgets and machine as in the milestone-4 ladder: 2-turn horizon, 5,000 nodes per decision, Apple M1 MacBook, CPython 3.14.0.

### cheat vs heuristic

`python -m minimax_games.games.star_realms tournament --bots cheat heuristic --games 500 --jobs 3`, as five runs of 100 consecutive seeds (0-99, ..., 400-499) one after another in the foreground, combined with `tournament.combine`, as in the milestone-4 ladder.

| Bot | Wins | Win% | As 1st | As 2nd | ms/move |
|---|---|---|---|---|---|
| cheat | 260/500 | 52.0% | 152/250 | 108/250 | 119 |
| heuristic | 240/500 | 48.0% | 142/250 | 98/250 | 0.0143 |

500 games, 0 unfinished, mean 24.7 turns, 36,415 cheat decisions (73 per game), 1.2 hours of cheat thinking time, 25.7 minutes of wall time. The first player won 294/500 (58.8%). The cheat's win rate: 52.0%, interval **47.6-56.3%** (as first player 60.8%, 54.6-66.6%; as second 43.2%, 37.2-49.4%). The five runs: 56, 53, 53, 55 and 43 wins of 100.

Against the milestone-4 ladder's **32.8%** (28.8-37.0%) at the same seeds and seats: **+19.2 points**, p < 0.0001 (two independent 500-game rates). Against an even match (50%), p ≈ 0.37: the cheat is now level with the heuristic, not clearly above it. On the 30-card first slice it won 53.6% (milestone-2 ladder).

Raw outputs (not committed): `m5/cardvalue/cheat-heuristic-500.txt` and the per-run `cheat-heuristic-<first seed>.txt`, the check `quick-v1-1000.txt`.

### Search speed

`task bench`, before (`31d8e97`, main) and after (`586d7b4`), at the same positions, one after the other:

| | Horizon 1 | Horizon 2 |
|---|---|---|
| before | 5/8 finished, 632,643 nodes in 13.4 s (47,200 nodes/s) | 4/8, 966,570 nodes in 25.0 s (38,700 nodes/s) |
| after | 5/8 finished, 632,643 nodes in 13.6 s (46,600 nodes/s) | 4/8, 941,416 nodes in 24.3 s (38,800 nodes/s) |

The same speed, within run-to-run noise (the horizon-2 trees differ, since the evaluation orders them differently). `evaluate` alone is about twice as fast as before: 5.9-7.4 µs per call against 11.3-14.0 µs, over the 4,504 states of 30 random games (seeds 0-29, 150 decisions each), timed alternately. The card values are precomputed per `Weights`, keyed by name, and the old starter-card penalty's comparison of every owned card with Scout and Viper (`Card.__eq__`, in Python) is now part of each card's precomputed value. Raw outputs: `m5/cardvalue/bench-before.txt`, `bench-after.txt`, `microbench.txt`.

### Verdict

Valuing what every card does lifts the cheat from 32.8% to 52.0% against the heuristic, nearly all of the 20.8 points it lost on the full deck (53.6% on the first slice), at no cost in search speed. So the milestone-4 reading holds: the evaluation, blind to the new cards' abilities, was what held the search back. The cheat is now level with the heuristic, not above it, with untuned values: tuning them (#115) is next, then rerunning the sampler against the heuristic (27.2% in the milestone-4 ladder).

## Milestone 5: PyPy re-measured

- Date: 2026-10-20
- Commit: `31d8e97` (#119: measurements, plus `task pypy`; no code change).
- Interpreters: CPython 3.14.0 (the project's environment) against PyPy 3.12.14 (PyPy 8.0.0), as in Milestone 2e, now after the CPython optimisations (Milestones 2f-2h, 2.6-2.9x) and on the full 80-card deck.
- Machine: Apple M1 MacBook, 8 GB (4 performance and 4 efficiency cores), macOS (Darwin 25.2). **Shared**: another job ran CPython tournaments with `--jobs 3` throughout (load average 2-6), so **absolute timings are contended and vary by up to 1.5x between runs; read the ratios.** To keep the comparison fair, each CPython run was paired with the same PyPy run back to back, alternating which went first (CPython, PyPy, PyPy, CPython, ...), and only one process of this measurement ran at a time (`--jobs 1`). For scale, the cheat's `bench` ran 35,900 nodes/s here on CPython against 45,400 alone (Milestone 4: the ladder).

PyPy ran as in Milestone 2e, from a copy of the repo (outside it) with `requires-python = ">=3.12"` and its own environment; CPython with plain `uv run` in the repo:

```
UV_PROJECT_ENVIRONMENT=<copy>/.venv-pypy uv sync --python pypy@3.12
UV_PROJECT_ENVIRONMENT=<copy>/.venv-pypy uv run --frozen python -m pytest -q
UV_PROJECT_ENVIRONMENT=<copy>/.venv-pypy uv run --frozen python -m minimax_games.games.star_realms bench [--budget 5000 [--worlds 4]]
```

The tournaments ran in chunks of consecutive seeds (cheat vs greedy 3 × 20, sampler vs heuristic 4 × 25), each chunk equal to `tournament --bots A B --games <n> --base-seed <first seed> --jobs 1`, combined with `tournament.combine`; results are per seed, so the split changes no win, seat or turn. **Each chunk is a fresh process, so PyPy's JIT warmed up once per chunk**, which slightly understates a single warm process's speed-up. It is also what a `--jobs 3` run sees, since each worker is a separate process warming up on its own: a 60-75 game `--jobs 3` run gives each worker 20-25 games, the size of these chunks. So no separate `--jobs 3` run was made (it would have put three more processes on the shared machine). Warm-up costs PyPy about a second per process (the budgeted benchmark below), 1-2% of a chunk.

### Results

| Workload | CPython 3.14 | PyPy 3.12 | PyPy speed-up |
|---|---|---|---|
| Test suite (`pytest -q`) | 1,380 passed, 12.7 s | 1,378 passed, 2 failed, 29.2 s | 0.4x |
| `bench`, horizon 1 (two pairs) | 17.6 s, 17.7 s (35,900 nodes/s) | 11.1 s, 10.5 s (57,200, 60,300 nodes/s) | 1.6x, 1.7x |
| `bench`, horizon 2 (two pairs) | 33.1 s, 30.9 s (29,200, 31,300 nodes/s) | 19.5 s, 18.2 s (49,600, 53,100 nodes/s) | 1.7x, 1.7x |
| `bench --budget 5000` (two pairs) | 101, 94 ms/decision (0.8 s) | 215, 213 ms/decision (1.7 s) | 0.5x |
| `bench --budget 5000 --worlds 4` | 111 ms/decision (0.9 s) | 249 ms/decision (2.0 s) | 0.4x |
| `tournament --bots cheat greedy`, 60 games | 129 ms/decision, 612 s | 78.5 ms/decision, 374 s | **1.64x** |
| `tournament --bots sampler heuristic`, 100 games | 131 ms/decision, 913 s | 86.4 ms/decision, 603 s | **1.52x** |

Per chunk (CPython against PyPy, wall time), showing the contention's spread:

| Chunk | CPython | PyPy | Speed-up |
|---|---|---|---|
| cheat vs greedy, seeds 0-19 | 120 ms/decision, 189 s | 88.8 ms/decision, 140 s | 1.35x |
| cheat vs greedy, seeds 20-39 | 138 ms/decision, 233 s | 83.6 ms/decision, 141 s | 1.65x |
| cheat vs greedy, seeds 40-59 | 128 ms/decision, 190 s | 62.0 ms/decision, 92 s | 2.05x |
| sampler vs heuristic, seeds 0-24 | 155 ms/decision, 272 s | 95.2 ms/decision, 167 s | 1.62x |
| sampler vs heuristic, seeds 25-49 | 106 ms/decision, 172 s | 80.7 ms/decision, 131 s | 1.31x |
| sampler vs heuristic, seeds 50-74 | 111 ms/decision, 199 s | 81.0 ms/decision, 146 s | 1.37x |
| sampler vs heuristic, seeds 75-99 | 151 ms/decision, 270 s | 88.3 ms/decision, 158 s | 1.70x |

**Same results on both.** Every `bench` search visits the same number of nodes, every budgeted decision (the cheat's and each sampler world's) finishes the same depth with the same nodes, and every tournament game has the same seed, first player, winner and turns, with the same number of decisions per bot in every chunk. cheat vs greedy: cheat 47/60 (23/30 first, 24/30 second), 25.9 mean turns, 4,740 cheat decisions on both. sampler vs heuristic: sampler 27/100 (18/52 first, 9/48 second), 24.1 mean turns, 6,942 sampler decisions on both. The sampler's random worlds draw from its own seeded `random.Random`, which gives the same sequence on both interpreters.

**The test suite fails the same two profiler tests as in Milestone 2e** (`tests/star_realms/test_profiling.py`: PyPy's `cProfile` has no entry for the dataclass-generated `GameState.__hash__`, and reports `cheat.decide`'s cumulative time, 2 ms, below its children's, 129 ms), so `task profile` still needs CPython. Everything else passes.

**The speed-up held up after the CPython optimisations.** On warm searches PyPy is still **1.6-1.7x** faster (`bench`; 1.5x and 1.9x in Milestone 2e, 1.8x on the largest search now, seed 1's turn 19 at horizon 1), and over long tournaments **1.5-1.65x** (1.7x in Milestone 2e, measured alone). The CPython optimisations cut allocation and hashing that PyPy's JIT couldn't remove either, so they sped up both interpreters by about as much. The two tournaments' combined ratios differ by about what the chunks scatter (1.31-2.05x), so the difference between the cheat and the sampler is not clearly real; both are at or above the 1.5x Milestone 2e set as the bar for using PyPy on long tournaments, the sampler only just. The contention probably costs the ratio a little: a later check of 6 cheat-vs-greedy games, back to back, gave 1.7x (141 against 82 ms/decision).

**Short runs got relatively worse.** CPython's decisions are about 3x faster than in Milestone 2e, so a short run is over before PyPy's JIT has paid back its warm-up: the 8 budgeted decisions take 0.8 s on CPython and 1.7 s on PyPy (0.5x; 0.9x in Milestone 2e), and the test suite runs at 0.4x (0.6x). The non-search bots are slower too (greedy 0.010 → 0.078 ms per move, the heuristic 0.013 → 0.10), negligible beside the search bots' 80+ ms. PyPy also uses more memory: 146 MB peak for a 6-game cheat-vs-greedy run against CPython's 37 MB, fine for `--jobs 3`.

Raw outputs (not committed): `m5/pypy/results/` (`bench*-{cpython,pypy}.txt`, `budget-*`, `sampler-budget-*`, and each chunk's `<bots>-<first seed>-<games>-<interpreter>.txt` and `.pkl`).

### Recommendation

**Run long tournaments under PyPy, from a separate environment, without changing the project's supported Python or CI.** At 1.5-1.65x, the milestone-4 ladder's 23-46 minute pairings would take about 15-30 minutes, and milestone 5 needs many of them. The results are identical, so a tournament can run on either interpreter and be compared with the other's.

`task pypy -- <command>` (README, Development) does it: it creates `.venv-pypy` (git-ignored) with PyPy 3.12, installs the package from the working tree into it, and runs `python -m minimax_games.games.star_realms <command>` there:

```
uv venv --allow-existing --python pypy@3.12 .venv-pypy
uv pip install --python .venv-pypy --python-version 3.13 --reinstall-package minimax-games .
.venv-pypy/bin/python -m minimax_games.games.star_realms tournament --bots sampler heuristic --games 500 --jobs 3
```

`--python-version 3.13` makes uv resolve as if the interpreter were 3.13, so the project's `requires-python = ">=3.13"` doesn't block the install, without a copy of the repo; only the package and superenums are installed (no dev tools), from the working tree on every run. Checked by running `task pypy -- tournament --bots cheat greedy --games 2 --jobs 2` (`--jobs` works under PyPy) and that a second install rebuilds the package.

What stays on CPython: the tests and pre-commit, `task profile` (the profiler's breakdown is wrong under PyPy), `bench --budget` and anything short (single games, replays), which PyPy runs slower. Nothing checks that the code still runs on PyPy 3.12: code using a 3.13-only feature would make `task pypy` fail at import, not give different results, and the identical-results check above can be repeated with a short tournament on both.

**Full support is not proposed.** It would mean lowering `requires-python` and ruff's `target-version` to 3.12, a CI matrix with a PyPy entry (renaming the required check, so the `main` ruleset changes with it), and the two profiler tests skipped on PyPy, for a speed-up only tournaments use. Revisit it only if PyPy becomes the main way tournaments run and a silent break becomes a real cost.

## Milestone 5: tuning

- Date: 2026-10-20
- Commit: the `issue-115-tune` branch (#115): `fed34f7` adds `tune` (`python -m minimax_games.games.star_realms tune`, `task tune`; README, Development) and lets the cheat search with any `Weights` (`CheatBot(weights=...)`, `cheat.decide(..., weights)`). The runs below used that code: the screen and the first two trials on CPython 3.14.0 (`uv run`), everything after on PyPy 3.12.14 (`task pypy`'s `.venv-pypy`, from the merge `f7cb72d`, which only adds `task pypy`). Results are identical on both interpreters (Milestone 5: PyPy re-measured); only the timings differ.
- Budgets: the cheat at its defaults, a 2-turn horizon and **5,000 nodes per decision**, in every run: no reduced budget, so nothing rests on a ranking transferring from a cheaper search.
- Machine: Apple M1 MacBook, 8 GB, shared: another job measured PyPy against CPython at the same time for part of the run, and a system process (`ANECompilerService`) took a full core for part of it, so the timings below are contended.
- Seeds: tuning on **3000-3199** (and 3200-3399 for the replication), confirmation on **1000-1499**, clear of the ladder's 0-499. Seeds 1000-1099 were #114's single check of the untuned values, not used for choosing anything here. Alternating seats, max 300 turns, `--jobs 3`, as in the ladder.

### Method

The budget was about 3-4 hours of machine time, and a 200-game cheat-vs-heuristic run takes about 11 minutes on CPython or 6 on PyPy with `--jobs 3`. A full coordinate pass over all 23 weights, two trials each, would have taken 4.5-8.5 hours, so the search was narrowed in three ways:

1. **A cheap screen picked the weights to tune.** `tune --screen --base-seed 3000 --games 20 --every 10 --jobs 3`: at 138 of the cheat's decisions (every 10th in its games against the heuristic at seeds 3000-3019), the share whose move changes when one weight is doubled or halved. A weight that never changes a decision can't change a game. 7.5 minutes (CPython).

   | Weight | Decisions changed |
   |---|---|
   | combat | 15.9% |
   | trade | 15.2% |
   | authority, draw, scrap_ability | 6.5% each |
   | ally_hand | 5.1% |
   | opponent_discards | 4.3% |
   | scrap_hand_or_discard | 3.6% |
   | destroy_base, copy_ship | 2.2% each |
   | base_defense, outpost_defense, next_ship_on_top, draw_if_bases | 1.4% each |
   | scrap_trade_row, acquire_ship_free_on_top, discard_then_draw | 0.7% each |
   | draw_per_faction_card, scrap_then_draw, scrap_from_hand, ships_gain_combat, ally_for_all_factions, starter_card | 0% |

   The six on top were tuned. The rest change a decision in under 5% of positions; the ones at 0% value cards rarely owned (Blob World, Brain World, Machine Base, Fleet HQ, Mech World), and the starter-card penalty (-1 Authority per Scout or Viper) is too small beside the card values (about 20 per card) for doubling it to matter: a bigger step than 2x would be needed to test it.
2. **Paired comparisons.** Every candidate plays the same 200 seeds as the incumbent. The cheat and the heuristic are deterministic, so a game the weight change doesn't affect is the same game, and only the **discordant** seeds (one won, the other lost) carry information. A change is kept if its extra wins exceed 1.64 standard deviations of noise, `1.64 * sqrt(discordant)` (a one-sided 5% sign test, normal approximation). Pairing removes the luck of the seeds whose game doesn't change: 34-86 of 200 seeds were discordant, so the noise was 6-9 games, against about 10 for two independent 200-game runs.
3. **Big steps, one pass.** Each weight is doubled and halved (`--factor 2`), since smaller changes would move the win rate by less than 200 games can see; one pass, in the screen's order.

`task pypy -- tune --fields combat trade authority draw scrap_ability ally_hand --base-seed 3000 --games 200 --jobs 3 --cache trials.jsonl` (the first two trials ran on CPython with `uv run` and the same `--cache`; the PyPy run replayed them from the file and carried on):

```
start: 102/200 wins
pass 1
  combat 20 -> 40: 110/200 wins, +8 on 68 discordant (needs > 13.5)
  combat 20 -> 10: 75/200 wins, -27 on 81 discordant (needs > 14.8)
  trade 20 -> 40: 58/200 wins, -44 on 86 discordant (needs > 15.2)
  trade 20 -> 10: 100/200 wins, -2 on 72 discordant (needs > 13.9)
  authority 8 -> 16: 110/200 wins, +8 on 42 discordant (needs > 10.6)
  authority 8 -> 4: 97/200 wins, -5 on 43 discordant (needs > 10.8)
  draw 30 -> 60: 96/200 wins, -6 on 58 discordant (needs > 12.5)
  draw 30 -> 15: 96/200 wins, -6 on 48 discordant (needs > 11.4)
  scrap_ability 0.25 -> 0.5: 99/200 wins, -3 on 55 discordant (needs > 12.2)
  scrap_ability 0.25 -> 0.125: 102/200 wins, +0 on 34 discordant (needs > 9.6)
  ally_hand 4 -> 8: 107/200 wins, +5 on 53 discordant (needs > 11.9)
  ally_hand 4 -> 2: 95/200 wins, -7 on 43 discordant (needs > 10.8)
end: 102/200 wins; unchanged
```

**No change passed the noise bar**, so the search kept every weight. Two moves hurt clearly: valuing trade at twice combat (-44) or combat at half trade (-27); the cheat does better valuing combat at least as much as trade. The best moves, doubling combat or Authority, gained 8 games each, under the bar.

The two near misses were then **replicated** on 200 fresh tuning seeds (3200-3399), with both changes together, a step the command doesn't take (a scratch script calling `tune.play` and `tune.compare`, PyPy):

```
incumbent: 89/200 wins
combat=40: 97/200 wins, +8 on 72 discordant (needs > 13.9)
authority=16: 96/200 wins, +7 on 51 discordant (needs > 11.7)
combat=40, authority=16: 102/200 wins, +13 on 71 discordant (needs > 13.8)
```

The gains repeated, but still under the bar, alone or pooled over the 400 seeds (combat +16 on 140 discordant, needs > 19.4; Authority +15 on 93, needs > 15.8).

Timings: CPython about 11-12 minutes per 200-game trial, PyPy about 6 (one PyPy trial took 33 minutes while the system process above held a core). The whole tuning took about 2.5 hours: screen 7.5 minutes, the pass 85 minutes, the replication 50.

### Confirmation on fresh seeds

The best candidate, `combat=40, authority=16`, against the untuned values, each 500 games at seeds 1000-1499, as two runs of 250 seeds (1000-1249, 1250-1499) combined with `tournament.combine`, on PyPy (`confirm_part.py`, the tournament's own games with the cheat given the weights):

| Weights | Cheat wins | Win% | 95% interval | As 1st | As 2nd | ms/move (PyPy) |
|---|---|---|---|---|---|---|
| untuned (#114) | 219/500 | 43.8% | 39.5-48.2% | 132/250 | 87/250 | 62.1 |
| combat 40, Authority 16 | 231/500 | 46.2% | 41.9-50.6% | 130/250 | 101/250 | 71.8 |

0 unfinished, mean 24.7 and 25.0 turns; 802 s and 876 s of wall time (PyPy, `--jobs 3`). **Paired** on the same seeds: the candidate won 97 seeds the untuned values lost and lost 85 they won, **+12 on 182 discordant, p = 0.41** (exact two-sided sign test), well under the bar (+22.1). Unpaired, +2.4 points, p = 0.45.

**The untuned values do worse on these seeds than on the ladder's**: 43.8% here against 52.0% at seeds 0-499 (p = 0.009, two independent 500-game rates), and #114's check on 1000-1099 gave 44/100. Nothing about the bots differs between the seed ranges, so either the ladder's 52.0% was on the lucky side of its interval (47.6-56.3%) or these seeds' deals favour the heuristic; both runs here were on the same seeds, so the comparison between them is unaffected. Against an even match the untuned cheat is below 50% on these seeds (p = 0.006) and the candidate not clearly (p = 0.09).

### Verdict

**The weights stay as they are.** Tuning by tournament found no change that beats the noise: the six most influential weights, doubled and halved over 200 paired games each, then the two best moves replicated on 200 more seeds and confirmed together on 500 fresh ones. Every comparison of the best moves (more value on combat and Authority) came out positive: combat +8 and +8, Authority +8 and +7 on the two tuning seed sets, both together +13 and then +12 on the fresh seeds, so a real gain of a point or two is plausible; showing it would take a few thousand paired games, and it is small beside what the cheat lacks against the heuristic. So hand-chosen card values were already close to what this evaluation can do at a 2-turn, 5,000-node search: the next lift has to come from the evaluation's form (rollouts, the next contender in docs/star-realms.md) or the search, not its weights. The remaining weights (17 untested) change under 5% of decisions each when doubled or halved, so tuning them would cost hours for less.

Limits: one pass and one step size (2x), the 17 weights screened out untested, the screen measured at the default weights only (an interaction between weights wouldn't show), and the tuning seeds 3000-3399 come from the same distribution as the rest, so nothing beyond the paired noise bar guards against tuning to them.

Raw outputs (not committed): `m5/tune/` (`screen-5000.txt`, `tune-log.txt` (CPython), `tune-log-pypy.txt`, `trials.jsonl` (every trial's per-seed outcomes), `replicate-3200.txt`, `confirm-{untuned,tuned}-{1000,1250}.{txt,pkl}`, `confirm-500.txt`, `bench-{before,after}-1.txt`, `scripts/`).

### Search speed

The cheat's nodes now carry their `Weights`. `bench` (CPython, contended), before (`origin/main`'s `node.py` and `cheat.py`) and after: the same trees (632,643 nodes at horizon 1, 941,416 at horizon 2) at the same speed, 49,600 and 49,200 nodes/s at horizon 1, 39,400 and 39,200 at horizon 2.

## Milestone 5: rollouts

- Date: 2026-10-21
- Commit: `8ce2a04` (#116, the `issue-116-rollout` branch). The cheat and the sampler take an **evaluator** (`CheatBot(evaluator=...)`, `SamplerBot(..., evaluator=...)`, `cheat.decide(..., evaluator)`): `evaluate.Linear(weights)` (the default, `LINEAR`, the tuned linear evaluation of Milestone 5: tuning, unchanged) or `rollout.Rollout(...)`. The tournament command takes `--eval linear|rollout`, `--rollouts`, `--rollout-turns` and `--epsilon`, and prints the evaluator in each search bot's settings.
- Interpreters: the tournaments ran on **PyPy 3.12.14** (`task pypy`'s `.venv-pypy`, installed from the commit above; a scratch script calling `tournament.play_tournament` with `cli.configured`'s bots, `--jobs 3`, so each chunk's per-seed results could be paired), the ms/decision used to match compute on **CPython 3.14.0** (`uv run`). Results are identical on both (Milestone 5: PyPy re-measured).
- Machine: Apple M1 MacBook, 8 GB, **shared**: other jobs ran at times (load average 2.5-5 during the CPython timings), so timings are contended; the CPython ones below were taken back to back to keep the comparison fair.
- Seeds: settings chosen on **4000-4199**, measured once on **2000-2499**, clear of the ladder's 0-499 and of #115's 1000-1499 and 3000-3399. Alternating seats, max 300 turns, `--jobs 3`.

### The rollout evaluation

`rollout.Rollout(rollouts, turns, epsilon, scale, seed, weights)` scores a leaf by `rollouts` playouts, each playing `turns` more turn ends (the rest of the current turn counts as one) with the greedy bot's moves, then scoring the state reached as a win probability for the searching player: 1 or 0 if the game ended, else `sigmoid(evaluate / scale)`. The mean is mapped into the integer scores alpha-beta compares, `round(SPAN * (2p - 1))` with `SPAN = 10**6`, strictly inside `-WIN..WIN`, so a game won or lost inside the search tree still ranks beyond any rollout. A finished leaf scores `WIN` or `-WIN`, as with `evaluate`.

- **The playout policy** (`rollout.greedy`) is `GreedyBot`'s, read from the state: it needs only the legal actions and the scraps owed (`bots.greedy_move`), so it reads only the mover's hand and the public state, not the opponent's hand or deck order, and builds no view. With probability `epsilon` a move is instead uniformly random among the moves other than `EndTurn` (when there are any): random play would otherwise end turns with trade and combat unspent (#117's research, §2.5). A playout also stops at turn 300.
- **Determinism**: the playouts' rng is seeded from the settings' `seed`, the perspective, and the state's own shuffle seed and turn, so the same state and settings give the same score in any order of evaluation and in any process (no `hash`, no sets). Leaves of the same turn usually share a shuffle seed, so they are played out with common random numbers. The tournament command seeds the rollouts with each game's seed. `Rollout.value(state, perspective, rng)` takes an rng instead, for a search outside alpha-beta (ISMCTS, #117).
- **`scale` = 16** `evaluate` points per unit of log-odds, by one-dimensional maximum likelihood of `P(win) = sigmoid(evaluate(s, 0) / scale)` over the turn starts of 300 heuristic-vs-heuristic games (seeds 5000-5299, 7,141 states, best 15.7, mean log-loss 0.606 against 0.650 at half and 0.624 at double). Greedy-vs-greedy games (seeds 5300-5599, 7,995 states) gave 19.0. A scratch script (`fit_scale.py`), seconds on CPython. Refit it if the weights change.

### Cost, and matching compute

ms per cheat decision against the heuristic on CPython. First 4 games (seeds 4000-4003) at a 500-node budget, to see what each setting costs:

| Rollouts × turns, epsilon | ms/decision at 500 nodes |
|---|---|
| 1 × 1, 0.1 | 29.5 |
| 1 × 2, 0 | 48.0 |
| 1 × 2, 0.1 | 52.8 |
| 1 × 4, 0.1 | 85.3 |
| 4 × 2, 0.1 | 158 |

The linear cheat at its default 5,000 nodes took 64.6 ms on the same games. Each setting was then given the budget that brings it to about that: 1 × 1 at 1,100 nodes, 1 × 2 at 650-700, 1 × 4 at 380 (4 × 2 would leave only about 200 nodes, too few to search 2 turns, so it was not tried). A rollout's cost is mostly the greedy playout's `legal_actions` and `advance` calls: per node of budget, the rollout cheat costs about 7 times the linear one (700 nodes take as long as 5,000).

Checked on 10 games (seeds 4000-4009) back to back, `tournament --bots cheat heuristic --games 10 --base-seed 4000 [--eval rollout] [--budget N]`:

| Cheat | ms/decision (CPython) |
|---|---|
| linear, 5,000 nodes (default) | 67.7 |
| rollout 1 × 2, epsilon 0, **700 nodes** (equal compute) | 72.5 |
| linear, 20,000 nodes | 218 |
| rollout 1 × 2, epsilon 0, **2,800 nodes** (the larger setting) | 252 |

So 700 nodes matches the default linear cheat (+7%), and 2,800 matches the linear cheat at 20,000 nodes (+16%), both slightly in the rollouts' favour.

### Choosing the settings (seeds 4000-4199)

200 games each, cheat vs heuristic, paired with the linear cheat on the same seeds (the bots are deterministic, so only discordant seeds carry information; exact two-sided sign test):

| Cheat | Wins | Against linear (paired) |
|---|---|---|
| linear, 5,000 nodes | 82/200 (41.0%) | |
| rollout 1 × 2, epsilon 0, 700 nodes | 84/200 (42.0%) | +2 on 78 discordant, p = 0.91 |
| rollout 1 × 2, epsilon 0.1, 650 nodes | 80/200 (40.0%) | -2 on 72, p = 0.91 |
| rollout 1 × 1, epsilon 0.1, 1,100 nodes | 65/200 (32.5%) | -17 on 57, p = 0.03 |
| rollout 1 × 4, epsilon 0.1, 380 nodes | 55/200 (27.5%) | -27 on 79, p = 0.003 |

Two turns per playout is the best length at equal compute: one turn plays only the mover's next turn, so the leaf is scored with the opponent's reply missing, and four leave too few nodes for the search. Randomness made no difference in the cheat (the cheat searches the true state, so a greedy playout already sees the real deck order), so epsilon 0 was kept: cheaper, and with it `rollouts` above 1 would only repeat the same playout. Hence the defaults `Rollout()` = 1 × 2 turns, epsilon 0, and `rollout.ROLLOUT_BUDGET` = 700 nodes, which the tournament command uses with `--eval rollout` unless `--budget` is given.

### cheat-with-rollouts vs heuristic, against the linear cheat (seeds 2000-2499)

500 games each on the same seeds, PyPy, `--jobs 3`, as foreground chunks of consecutive seeds (an even number of games each, so seats alternate as in one run) combined with `tournament.combine`: the linear cheats and the 700-node rollouts in chunks of 200, 200 and 100, the 2,800-node rollouts and 20,000-node linear in chunks of 60-100.

| Cheat | Wins | Win% | 95% interval | As 1st | As 2nd | ms/move (PyPy) |
|---|---|---|---|---|---|---|
| linear, 5,000 nodes (default) | 216/500 | 43.2% | 38.9-47.6% | 133/250 | 83/250 | 63.4 |
| rollout, 700 nodes (equal compute) | 195/500 | 39.0% | 34.8-43.3% | 109/250 | 86/250 | 74.9 |
| linear, 20,000 nodes | 257/500 | 51.4% | 47.0-55.8% | 153/250 | 104/250 | 189 |
| rollout, 2,800 nodes (equal to linear at 20,000) | 241/500 | 48.2% | 43.9-52.6% | 133/250 | 108/250 | 228 |

0 unfinished in every run, mean 24.3-24.9 turns, 70-72 cheat decisions per game. Wall time (PyPy, `--jobs 3`, chunks added up): 13.0, 16.0, 41.0 and 50.3 minutes.

**Paired**, on the same 500 seeds:

| Comparison | Result |
|---|---|
| rollout 700 vs linear 5,000 (equal compute) | **-21** on 209 discordant (94 won, 115 lost), p = 0.17 |
| rollout 2,800 vs linear 20,000 (equal compute, larger) | **-16** on 190 (87, 103), p = 0.28 |
| rollout 2,800 vs linear 5,000 (4x compute) | +25 on 197 (111, 86), p = 0.09 |
| rollout 2,800 vs rollout 700 | +46 on 146 (96, 50), p < 0.001 |
| linear 20,000 vs linear 5,000 | **+41** on 81 (61, 20), p < 0.001 |

The linear cheat scores 43.2% on these seeds, between its 52.0% on 0-499 and 43.8% on 1000-1499: as Milestone 5: tuning found, only comparisons on the same seeds mean anything.

### Verdict

**The linear evaluation stays the default.** At equal compute the rollout evaluation is no better than the tuned linear one, and a little worse: -21 games of 500 at the default compute and -16 at four times it, neither significant on its own, but both the same way, as the selection seeds were level (+2). Each node scored by a playout costs about 7 times a linear one, and the better leaf is not worth the search it costs: with a seventh of the nodes, far fewer of the 2-turn searches finish.

**The finding that matters more is about budget.** Four times the nodes lifts the linear cheat from 43.2% to 51.4% against the heuristic (+41 on 81 discordant seeds, p < 0.001; the rollout cheat gains as much, +46). Milestone 3 found more nodes didn't help on the 30-card deck, but on the full deck with card values (#114) they do: half the turn starts of mid and late game didn't finish even the 1-turn search at 5,000 nodes (Milestone 4: search speed on the full deck). So the next lift is likely from search efficiency (a canonical order for buys and attacks, as #105 found) or a larger default budget, not the leaf evaluation. Raising the default (and the sampler's, which shares it) is a separate decision with its cost in tournament time: about 3x per decision.

For ISMCTS (#117): `Rollout.value(state, perspective, rng)` and `rollout.playout` / `rollout.win_probability` are usable outside alpha-beta; the 2-turn greedy playout with `sigmoid(evaluate / 16)` is the better leaf of those tried here, though epsilon was not shown to matter in the cheat (where the true state is searched) and may in sampled worlds.

Limits: one seed set for choosing settings (200 games, so a small difference between epsilon 0 and 0.1 is not excluded), playout lengths of 1, 2 and 4 turns only, no playouts to the end of the game (3.5-7.5 ms each, #117's measurements: under 20 per decision at equal compute), and the sampler with rollouts not measured (its evaluator is the cheat's, per world).

### Search speed

The linear path now calls its evaluator through `Linear.__call__`. `bench` (CPython, contended), before (`93589cf`, main) and after: the same trees (632,643 nodes at horizon 1, 941,416 at horizon 2), 47,500 against 47,500 and 46,400 nodes/s at horizon 1, 38,100 against 37,900 and 37,700 at horizon 2: within noise.

Raw outputs (not committed): `m5/rollout/` (`cost-*.txt`, the CPython costs; `cpu-*-4000-10.txt`, the equal-compute timings; `sel-*-4000.{txt,pkl}`, the selection runs; `fin-*-<first seed>.{txt,pkl}` and the combined `fin-*-500.txt`, the final runs; `fin-paired-*.txt`; `bench-{before,after,after-2}.txt`; `scripts/` (`fit_scale.py`, `cost.sh`, `run_part.py`, `paired.py`, `combine_all.py`)).

## Milestone 5: ISMCTS

- Date: 2026-10-21
- Commit: the `issue-117-ismcts` branch (#117): `d84cdcf` adds `search.ismcts` (generic SO-ISMCTS with availability counts) and `ismcts.IsmctsBot` (`--bots ismcts`, `tournament --iterations N`); `c79a672` adds the `horizon` leaf option, `32cdffb` the heuristic playout policy (`policy="heuristic"`), and `7aaf99b` makes epsilon 0.25 the default. Design: [ismcts.md](ismcts.md). The selection runs used `d84cdcf` to `32cdffb` with explicit settings (the defaults' behaviour didn't change between them); the final runs used `7aaf99b`.
- Interpreters: the tournaments ran on **PyPy 3.12.14** (`task pypy`'s `.venv-pypy`, installed from each commit above; a scratch script, `run_part.py`, calling `tournament.play_tournament` with the bots, `--jobs 3`, in foreground-equivalent chunks of consecutive seeds combined with `tournament.combine`), the ms/decision used to match compute on **CPython 3.14.0** (`uv run`), alone. Results are identical on both (Milestone 5: PyPy re-measured).
- Machine: Apple M1 MacBook, 8 GB. Nothing else ran during the CPython timings marked "alone"; the machine slept for part of some tournament chunks, which the reported wall times (a monotonic clock that stops in sleep) don't include.
- Seeds: settings chosen on **4000-4199**, measured once on **0-499**, the ladder's seeds. Alternating seats, max 300 turns, `--jobs 3`.

### The bot

`IsmctsBot(seed, iterations=450, exploration=0.7, epsilon=0.25, extra_turns=0)`: single-observer ISMCTS from its view, a fresh `sample_world` per iteration, nodes keyed by the actions from the root, UCB1 over the actions legal in the iteration's world with availability counts, one random untried action expanded per iteration, and the most visited root action played (a lone legal action without searching). A leaf is valued by finishing the turn in progress with the greedy policy read from the state (`rollout.playout`), each move replaced with probability epsilon by a random one other than `EndTurn`, then `sigmoid(evaluate / 16)` for player 0, or 1 or 0 if the game ended (`rollout.win_probability`).

**`scale` = 16, adopted from #116, not refitted**: it was fitted by maximum likelihood on 7,141 turn starts of 300 heuristic-vs-heuristic games for the current default `Weights` (Milestone 5: rollouts), which #115's tuning left unchanged, and this leaf is the same `sigmoid(evaluate / scale)` at a turn boundary.

### Cost, and matching compute

Every decision of the bot's games against the heuristic at seeds 4000-4003 (bot first in even seeds), CPython 3.14.0, alone, single process (`probe.py`: a wrapper timing `choose` and counting `rules.advance_legal` calls). "All" averages every decision, as the tournament's ms/move does (the sampler searches lone actions too, ISMCTS doesn't); "searched" only those with more than one legal action.

| Bot | Decisions | ms/decision (all, searched) | `advance_legal` per decision (all, searched) |
|---|---|---|---|
| sampler, 5,000 nodes, 4 worlds (default) | 363 | 94.3, 99.9 | 3,748, 3,967 |
| ISMCTS 300, epsilon 0.1 | 249 | 63.1, 69.8 | 1,929, 2,135 |
| ISMCTS 450, epsilon 0.1 | 218 | 93.6, 106 | 2,927, 3,324 |
| ISMCTS 600, epsilon 0.1 | 223 | 128, 146 | 3,974, 4,521 |
| **ISMCTS 450, epsilon 0.25 (default)** | 244 | **97.2** | **3,071**, 3,469 |
| ISMCTS 1,800, epsilon 0.25 (4x) | 205 | 384 | 13,524, 15,068 |

So **450 iterations** matches the sampler's time per decision (97 against 94 ms; rechecked back to back after the tournaments, 91.9 against 90.9 ms, with the 4x setting at 384 ms, 4.2x), about 0.2 ms per iteration, with about 80% of the sampler's `advance_legal` calls (3,071 against 3,748; a playout's moves are `advance_legal` calls, as the sampler's nodes are). The research expected 100-250 iterations: the state-level greedy policy, without `observe`, made playouts cheaper than its estimate. The leaf variants were given the iterations that bring them to about the same time (alone, CPython, seeds 4000-4003): `extra_turns=1` 290 (300 took 95.8 ms), `horizon=2` 390 (300 took 70.5 ms), the heuristic policy 350 (88.8 ms). Epsilon barely changes the cost (0.5: 93.4 ms).

Under PyPy the sampler gains more than ISMCTS: in the final runs ISMCTS took 87-107 ms per decision against the sampler's 71-90 (contended, `--jobs 3`), so at the CPython-matched budget ISMCTS spends about 20% more time on PyPy.

Diagnostics at the default (the same games): the tree holds 445 nodes on average (one per iteration, minus finished playouts), the root has 6.4 legal actions, the most visited line passes 0.68 turn ends on average, and 54% of iterations reach the opponent's turn. At 4x: 1,715 nodes, 0.98 and 63%. So at this budget the search mostly covers the current turn's next few decisions, as the research expected.

### Choosing the settings (seeds 4000-4199)

200 games each against the heuristic, PyPy, `--jobs 3` (`run_part.py <name> ismcts heuristic 4000 200 <iterations> <c> <epsilon> <extra turns> [<horizon>|none [<policy>]]`). Each is compared with the research's defaults (c 0.7, epsilon 0.1, greedy, 450 iterations) on the same seeds, by the seeds one won and the other lost (exact two-sided sign test). ISMCTS is stochastic, so unlike the cheat's, a game a setting doesn't affect isn't replayed identically: every seed can be discordant, and the pairing removes less noise than it did for the cheat.

| ISMCTS | Iterations | Wins | Against c 0.7, epsilon 0.1 (paired) |
|---|---|---|---|
| c 0.7, epsilon 0.1 | 450 | 55/200 (27.5%) | |
| c 0.35 | 450 | 57/200 (28.5%) | +2 on 56 discordant, p = 0.89 |
| c 1.0 | 450 | 51/200 (25.5%) | -4 on 56, p = 0.69 |
| epsilon 0 | 450 | 59/200 (29.5%) | +4 on 54, p = 0.68 |
| **epsilon 0.25** | 450 | **68/200 (34.0%)** | **+13 on 57, p = 0.11** |
| epsilon 0.5 | 450 | 61/200 (30.5%) | +6 on 52, p = 0.49 |
| `extra_turns` 1 | 290 | 31/200 (15.5%) | **-24 on 58, p = 0.002** |
| `horizon` 2 | 390 | 41/200 (20.5%) | -14 on 54, p = 0.08 |
| heuristic policy, epsilon 0.1 | 350 | 61/200 (30.5%) | +6 on 46, p = 0.46 |
| heuristic policy, epsilon 0.25 | 350 | 56/200 (28.0%) | +1 on 49, p = 1.0 |
| *sampler (default), for reference* | | 75/200 (37.5%) | ISMCTS c 0.7 / epsilon 0.1: -20 on 72, p = 0.02; epsilon 0.25: -7 on 77, p = 0.49 |

0 unfinished in every run; 6-9 minutes of wall time each (PyPy, `--jobs 3`).

- **The exploration constant doesn't matter** in 0.35-1.0, as Cowling et al. found; 0.7 stays.
- **Some randomness in the playouts helps a little**: epsilon 0.25 was the best (+13, not significant on its own; 0 and 0.5 between). It is the default: the playout's greedy policy never scraps, never destroys bases and buys only by cost, so a deterministic one scores every leaf by the same weak continuation. In the cheat epsilon made no difference (Milestone 5: rollouts), but there the playout ran on the true state.
- **Longer playouts lose at equal compute.** One more whole turn (`extra_turns` 1) cost 35% of the iterations and lost clearly. So did the fixed horizon (`horizon` 2: every leaf played out to the end of the opponent's reply, so siblings reached by ending the turn and by playing on are scored at the same boundary, a bias of "finish the turn" that this was to remove): -14, nearly significant. A longer greedy playout adds the greedy policy's misjudgements of a whole turn to the leaf's value, and here they cost more than the bias.
- **The heuristic as the playout policy didn't help** at equal compute (+6 and +1): its better moves cost a view per move (`observe`), about 1.3x per iteration, so 350 iterations.
- Not tried: playouts to the end of the game (3.5-7.5 ms each, the research's §2.8: about 15-30 per decision) and the enhancements the research ranks as follow-ups.

Chosen: c 0.7, epsilon 0.25, the greedy policy, 0 extra turns, no horizon, 450 iterations.

### ISMCTS against the heuristic and the sampler (seeds 0-499)

At the chosen defaults, PyPy, `--jobs 3`, each pairing as five chunks of 100 consecutive seeds (0-99, ..., 400-499) combined with `tournament.combine`; each chunk equals `task pypy -- tournament --bots <A> <B> --games 100 --base-seed <first seed> --jobs 3` (`run_part.py fin-<A>-<B> <A> <B> <first seed> 100`). The sampler was rerun against the heuristic on the same seeds so the two compare paired: its last 500-game result (27.2%, Milestone 4: the ladder) predates the card values (#114).

| Pairing (first bot's results) | Wins | Win% | 95% interval | As 1st | As 2nd | ms/move (PyPy, contended) | Wall time |
|---|---|---|---|---|---|---|---|
| ismcts vs heuristic | 173/500 | 34.6% | 30.6-38.9% | 99/250 | 74/250 | 107 | 17.8 min |
| sampler vs heuristic | 167/500 | 33.4% | 29.4-37.6% | 89/250 | 78/250 | 89.6 | 19.4 min |
| **ismcts vs sampler** | 243/500 | **48.6%** | 44.2-53.0% | 125/250 | 118/250 | 86.8 (sampler 71.1) | 28.6 min |
| ismcts at 4x (1,800 iterations) vs heuristic, **seeds 0-199 only** | 76/200 | 38.0% | 31.6-44.9% | 46/100 | 30/100 | 362 | 23.4 min |

0 unfinished in every run; mean 24.1-24.5 turns; 55-73 search-bot decisions per game. The first player won 275/500, 261/500 and 257/500 of the 500-game pairings.

**Paired**, on the same seeds (exact two-sided sign test):

| Comparison | Result |
|---|---|
| ismcts vs sampler, each against the heuristic (seeds 0-499) | +6 on 184 discordant (95 won, 89 lost), p = 0.71 |
| ismcts 4x vs ismcts at the default, against the heuristic (seeds 0-199) | +1 on 61 (31, 30), p = 1.0 |
| ismcts 4x vs the sampler, against the heuristic (seeds 0-199) | +2 on 64 (33, 31), p = 0.90 |

The 4x setting ran only 200 games: its chunks ran at a fraction of full speed on the shared machine (memory pressure: about an hour of clock time each where 12 minutes were expected), so it was stopped after seeds 0-199, enough to see it gains nothing measurable over the default on those seeds (the default won 75/200 there).

### Verdict

**ISMCTS plays level with the sampler at equal compute, and no better.** Against the heuristic it wins 34.6% to the sampler's 33.4% on the same seeds (+6 on 184 discordant, p = 0.71), and head to head 48.6% (interval 44.2-53.0%). On the selection seeds the sampler was ahead (75/200 against 68/200 at the chosen settings, -7 on 77, p = 0.49), so over 700 seeds the two are within a few games of each other. Both remain far below the heuristic, the bar for a legitimate bot.

This is what the research predicted (docs/ismcts.md, §1): in Star Realms the hidden information is almost all chance and the opponent's fresh hand, so removing the sampler's clairvoyance gains little. And at about 450 iterations ISMCTS is close to a bandit over the next few micro-actions with an epsilon-greedy completion of the turn: its tree passes 0.68 turn ends on average along the most visited line, against the sampler's 2-turn alpha-beta searches.

**More iterations didn't help** (4x: +1 on 61 seeds), unlike the cheat, which gained clearly from 4x the nodes (Milestone 5: rollouts). The tree goes a little deeper at 4x (0.98 turn ends on the most visited line), but the leaf (greedy completion of the turn, then `evaluate`) limits it, and the settings sweep agrees: changes to the leaf moved the result far more than the exploration constant did (longer playouts -14 and -24, epsilon +13, `c` ±4). So the next lever for ISMCTS is the leaf or priors on the micro-actions (progressive bias, implicit minimax backups with `evaluate`: the research's §5, items 4 and 5) and collapsing orderings of a turn's buys and attacks (§5, item 3), not budget.

`IsmctsBot`'s defaults stay as chosen: 450 iterations, `c` 0.7, epsilon 0.25, the greedy playout, the leaf at the end of its turn.

Raw outputs (not committed): `m5/ismcts/` (`cost-*.txt`, the CPython costs (`cost-sampler-5000-4000-4.txt`, `cost-ismcts-<N>-4000-4.txt`, `cost-variants-idle.txt`, `cost-policy-idle.txt`, `cost-4x-idle.txt`; `cost-leaf-settings.txt`, `cost-horizon.txt` contended); `diag-ismcts-{450,1800}.txt`, the tree diagnostics; `sel-*-4000.{txt,pkl}`, the selection runs; `fin-*-<first seed>.{txt,pkl}` and the combined `fin-*-all.txt`, the final runs; `fin-paired-*.txt`; `scripts/` (`probe.py`, `probe_extra.py`, `run_part.py`, `paired.py`, `combine_all.py`)).

## Milestone 5: the ladder

- Date: 2026-10-21
- Commit: `1531a34` (main after #117), no code changes. The defaults the milestone-5 slices chose: the linear evaluation with the hand-chosen card values (#114, kept by #115's tuning and by #116's rollout comparison), the cheat and the sampler at a 2-turn horizon and **5,000 nodes per decision** (`cheat.BUDGET`; the sampler's 4 worlds 1,250 each), ISMCTS at **450 iterations**, `c` 0.7, epsilon 0.25 (#117).
- **Equal compute**: the cheat and the sampler share the node budget, as in the milestone-3 and -4 ladders. ISMCTS's 450 iterations were matched to the sampler by ms per decision on CPython 3.14.0, alone (97 against 94 ms; Milestone 5: ISMCTS, Cost). Under PyPy ISMCTS spends about 20% more time per decision than the sampler (107 against the sampler's 89.6 ms/move, each against the heuristic, contended), so if anything the matching favours ISMCTS a little.
- Interpreters: the new runs on **PyPy 3.12.14** (`task pypy`'s `.venv-pypy`, installed from `1531a34`); the reused cheat run on CPython 3.14.0. Results are identical on both (Milestone 5: PyPy re-measured); only ms/move differs.
- Machine: Apple M1 MacBook, 8 GB, run under `caffeinate -i`; shared at times (load average 3-6 near the end of the sampler-vs-cheat run), so ms/move and wall times are contended.

500 games per pairing, seeds 0-499, alternating seats, max 300 turns, `--jobs 3`, each pairing as five runs of 100 consecutive seeds (0-99, ..., 400-499) one after another in the foreground, combined with `tournament.combine`. Each run equals `task pypy -- tournament --bots <A> <B> --games 100 --base-seed <first seed> --jobs 3` (a scratch script, `run_part.py <name> <A> <B> <first seed> 100`, calling `tournament.play_tournament` with `cli.configured`'s bots so the per-seed results could be pickled and paired). All six pairings ran the full 500 games (no departure). No game was unfinished.

### Reused and new runs

Four of the six pairings had already been played at seeds 0-499 by the slices that chose the defaults, with code whose moves are unchanged since, so they were reused rather than replayed:

| Pairing | Source | Commit run | Why still valid |
|---|---|---|---|
| cheat vs heuristic | Milestone 5: card values (#114), CPython | `586d7b4` (= `93c9efb`, its squash merge) | Since then #115 and #116 changed no default (weights kept, `Linear(WEIGHTS)` the default evaluator), #116 routed the evaluation through `Linear.__call__` and #117 moved `GreedyBot.choose`'s body into `bots.greedy_move` unchanged: refactors. **Checked by replaying** seeds 400-499 at `1531a34`'s defaults on PyPy: every game the same first player, winner and turns, and the same decision counts (7,508 cheat, 6,562 heuristic) as #114's run (43/100 both times). |
| sampler vs heuristic | Milestone 5: ISMCTS (#117), PyPy | `7aaf99b` | `src/` is identical in `7aaf99b` and `1531a34` (the squash merge). |
| ismcts vs heuristic | the same | `7aaf99b` | the same |
| ismcts vs sampler | the same | `7aaf99b` | the same |
| ismcts vs greedy | **new** | `1531a34` | |
| sampler vs cheat | **new** | `1531a34` | |

### The ladder

| Pairing (first bot's results) | Wins | Win% | 95% interval | As 1st | As 2nd | ms/move | Wall time |
|---|---|---|---|---|---|---|---|
| cheat vs heuristic | 260/500 | **52.0%** | 47.6-56.3% | 152/250 | 108/250 | 119 (CPython) | 25.7 min (CPython) |
| sampler vs heuristic | 167/500 | **33.4%** | 29.4-37.6% | 89/250 | 78/250 | 89.6 | 19.4 min |
| ismcts vs heuristic | 173/500 | **34.6%** | 30.6-38.9% | 99/250 | 74/250 | 107 | 17.8 min |
| ismcts vs sampler | 243/500 | 48.6% | 44.2-53.0% | 125/250 | 118/250 | 86.8 (sampler 71.1) | 28.6 min |
| ismcts vs greedy | 373/500 | 74.6% | 70.6-78.2% | 194/250 | 179/250 | 95.7 | 19.6 min |
| sampler vs cheat | 181/500 | 36.2% | 32.1-40.5% | 111/250 | 70/250 | 75.2 (cheat 58.1) | 25.7 min |

ms/move is PyPy unless marked, contended by the three processes. The two new pairings: ismcts vs greedy, mean 26.5 turns, 34,044 ISMCTS decisions (68 per game), 0.91 hours of ISMCTS thinking time, the first player won 265/500; the five runs 73, 72, 77, 77 and 74 wins of 100. sampler vs cheat, mean 23.5 turns, 33,464 sampler and 32,029 cheat decisions (67 and 64 per game), 0.70 and 0.52 hours, the first player won 291/500; the five runs 38, 36, 43, 28 and 36 wins of 100. The reused pairings' details are in their sections.

**Paired** on the same seeds against the heuristic (exact two-sided sign test on the seeds one won and the other lost):

| Comparison | Result |
|---|---|
| cheat vs sampler | +93 on 171 discordant (132, 39), p < 0.001 |
| cheat vs ismcts | +87 on 199 (143, 56), p < 0.001 |
| ismcts vs sampler (Milestone 5: ISMCTS) | +6 on 184 (95, 89), p = 0.71 |

### Against milestone 4

The same pairings, seeds and seats in the milestone-4 ladder (the full deck, `evaluate` scoring only primary trade and combat). The p-value compares two independent 500-game rates. ISMCTS didn't exist in milestone 4; for scale, the heuristic beat greedy 90.6% and the sampler 62.4% there (the heuristic's and greedy's moves haven't changed since).

| Pairing (first bot's win rate) | Milestone 4 | Milestone 5 | Change | p |
|---|---|---|---|---|
| cheat vs heuristic | 32.8% | **52.0%** | +19.2 | < 0.0001 |
| sampler vs heuristic | 27.2% | **33.4%** | +6.2 | 0.03 |
| sampler vs cheat | 40.0% | 36.2% | -3.8 | 0.22 |
| ismcts vs heuristic | | 34.6% | | |
| ismcts vs sampler | | 48.6% | | |
| ismcts vs greedy (sampler vs greedy in milestone 4) | (62.4%) | 74.6% | | |

### Reading the ladder

**cheat ≈ heuristic >> ismcts ≈ sampler > greedy.**

- **No legitimate bot beats the heuristic.** The strongest are **ISMCTS, 34.6% (30.6-38.9%), and the sampler, 33.4% (29.4-37.6%)**, level with each other: +6 on 184 discordant seeds (p = 0.71) against the heuristic, and 48.6% (44.2-53.0%) head to head. ISMCTS is nominally ahead by 1.2 points; the measurements can't separate them. Both are far below 50% against the heuristic (p < 0.0001), and ISMCTS beats greedy 74.6% where the heuristic does 90.6%. The heuristic stays the bar for a legitimate bot.
- **The card values lifted the cheat much more than the sampler.** The cheat gained 19.2 points against the heuristic (32.8% to 52.0%) and is now level with it; the sampler, with the same evaluation and total budget, gained 6.2 (27.2% to 33.4%, p = 0.03). On the same seeds the cheat wins 93 more games against the heuristic than the sampler, and head to head the sampler's share fell a little, 40.0% to 36.2% (within noise, p = 0.22). So the cost of not seeing the hidden cards at equal compute is now about 19 points against the heuristic, where it was about 6. A likely reason, not isolated by an ablation: the sampler splits the 5,000 nodes over 4 worlds, so each world searches 1,250, and on the full deck budget now matters (#116: the cheat at 4x the nodes gained 8 points). With a better evaluation, each world's shallower search may waste more of it.
- **The seed range matters about as much as some of these differences.** The default cheat scored 52.0% against the heuristic on seeds 0-499, but 43.8% on 1000-1499 (Milestone 5: tuning) and 43.2% on 2000-2499 (Milestone 5: rollouts), the same code: so "level with the heuristic" holds on the ladder's seeds, and on the 1,500 seeds measured the cheat is a little below it. Comparisons on the same seeds (all of the above) are unaffected; a single rate is uncertain by more than its interval suggests.

**What should come next.**

- **Search efficiency, first: a canonical order for buys and attacks** (closed #105). It is the measured cause of unfinished searches on the full deck (Milestone 4: search speed), budget now pays (#116: the cheat at 20,000 nodes rose from 43.2% to 51.4% on seeds 2000-2499, +41 on 81 discordant seeds, p < 0.001), and collapsing orderings would make every node count for the cheat, the sampler's smaller per-world searches and ISMCTS (whose tree spends its iterations on micro-action orderings; docs/ismcts.md §5, item 3) alike.
- **Then the budget and K sweeps for the sampler**, unmeasured since the card values: if the sampler's lag behind the cheat is its per-world budget, fewer worlds or more nodes will show it. A larger default budget is a separate decision with a 3-4x cost in tournament time.
- **For ISMCTS, the leaf, not the budget** (#117): 4x the iterations gained nothing (+1 on 61 seeds), while leaf changes moved it by up to 24 games of 200. Priors on the micro-actions (progressive bias) or implicit minimax backups with `evaluate` (docs/ismcts.md §5, items 4 and 5) are the levers there.
- **A learned evaluation** stays deferred: tuning the linear weights (#115) and greedy rollouts (#116) didn't beat the hand-chosen card values, and the cheat, the upper bound for the legitimate bots, is already level with the heuristic; the gap to close is in the legitimate bots' search, not the cheat's evaluation.

Raw outputs (not committed): `m5/ladder/` (`fin-ismcts-greedy-<first seed>.{txt,pkl}`, `fin-sampler-cheat-<first seed>.{txt,pkl}` and the combined `fin-*-all.txt`; `check-cheat-heuristic-400.{txt,pkl}`, the replay checking the reused cheat run; `paired-cheat-vs-{sampler,ismcts}-heuristic.txt`; `scripts/` (`run_part.py`, `all.sh`, `combine_all.py`, `paired.py`)); the reused runs in `m5/cardvalue/` and `m5/ismcts/`.

## After milestone 5: a leaf-bias check for ISMCTS

2026-10-21, `main` at 71a6f6c, PyPy. "What should come next" above is superseded by the reviews of #127 and #128: the plan is now the design doc's Milestones 6-8 (a whole-turn planner, a Rust engine, a learned value).

ISMCTS scores its leaves at different turn boundaries: a leaf inside the bot's own turn is scored at that turn's end, and one past `EndTurn` after the opponent's reply. If being the side to move is worth a lot of `evaluate` points, that mixes two scales in one tree, and could explain why 4x the iterations gained nothing.

**The size of the bias.** Over the turn starts of 300 heuristic-vs-heuristic games (seeds 5000-5299, 7,441 states), a grid-search maximum-likelihood fit of `P(player 0 wins) = sigmoid((evaluate(s, 0) + t·m) / s)`, with `m` = +1 when player 0 is to move and -1 otherwise, gives **s = 14 and t = 6**: being the side to move is worth about 6 `evaluate` points each way.

**The effect of correcting it.** ISMCTS (450 iterations, defaults otherwise) with the term added to its leaf value (`(evaluate + 6·m) / 14`), against the heuristic on seeds 4000-4199, `--jobs 3`:

| ISMCTS leaf | Wins of 200 |
|---|---|
| baseline (reproduces the recorded run exactly) | 68 |
| with the side-to-move term | 72 |

Paired by seed: +4 on 70 discordant seeds, p = 0.72. **No measurable effect at the default budget.** Whether the correction makes ISMCTS gain from more iterations wasn't tested (4x, about 1-1.5 hours). The flat budget curve is better explained by the leaf evaluation itself than by this bias. A whole-turn planner scores every candidate at the same boundary, so the bias can't affect its choices (design, Search).

Raw outputs and scripts (not committed): `tempo/` (`fit_tempo.py <first seed> <games>`; `run_tempo.py <base|tempo> <first seed> <games> <iterations>`, which subclasses `ismcts.StarRealms` to override its leaf value; `{base,tempo}-450-4000-200.pkl`).

## Milestone 6: full-deck turn sizes

- Date: 2026-10-21
- Commit: the merge of #132 (measured on its branch, from `main` at e884671).
- Command: `python -m minimax_games.games.star_realms turn-sizes --games 20 --jobs 2` (or `task turn-sizes -- --games 20 --jobs 2`): every turn of the heuristic-vs-heuristic games of seeds 0-19, each enumerated up to 100,000 states. 2.4 minutes wall time.
- Machine: Apple M1 MacBook, 8 GB, macOS (Darwin 25.2), CPython 3.14.0, shared with other jobs.

The whole-turn planner (design, Search) lists the distinct states reachable at the end of the mover's turn and scores each. `turn_sizes.turn_ends` does that listing, from each turn's first decision on the true state (one world), each state expanded once, as `bench.turn_tree` merges orderings: **end states** are the states after `EndTurn` or a winning play, **within-turn states** the states expanded to reach them, and **states visited** both, which is what a planner's state budget pays for. **Unordered ends** count the ends up to the order of every pile but the decks (hand, discard pile, bases, Trade Row, scrap heap): the positions a value that counts cards by zone can tell apart. A turn past 100,000 states visited is stopped and counted as capped, ranking above every complete turn. The counts are the same on every run; only the ms vary.

483 turns, 33 (7%) capped:

| Count | q1 | median | q3 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| End states | 26 | 110 | 1,182 | 12,214 | capped | capped |
| Unordered ends | 20 | 79 | 430 | 3,357 | capped | capped |
| Within-turn states | 26 | 115 | 1,412 | 15,470 | capped | capped |
| States visited | 52 | 224 | 2,492 | 28,876 | capped | capped |
| ms (CPython) | 2 | 10 | 135 | 1,446 | 6,767 | 9,717 |

Share of turns visiting more states than a budget (capped turns count as over):

| Turns | Count | median visited | p90 visited | > 1k | > 5k | > 20k | > 100k |
|---|---|---|---|---|---|---|---|
| all | 483 | 224 | 28,876 | 34% | 20% | 12% | 7% |
| 1-5 | 100 | 32 | 116 | 1% | 0% | 0% | 0% |
| 6-10 | 100 | 168 | 3,156 | 23% | 8% | 5% | 0% |
| 11-15 | 100 | 284 | 15,835 | 33% | 15% | 7% | 2% |
| 16-20 | 93 | 1,890 | 48,673 | 53% | 34% | 16% | 9% |
| 21-25 | 64 | 2,541 | capped | 62% | 42% | 30% | 25% |
| 26-30 | 24 | 16,820 | capped | 75% | 54% | 50% | 25% |
| 31-35 | 2 | 65,406 | capped | 100% | 100% | 100% | 50% |

Over the 450 complete turns: 739,789 ends, 258,292 up to pile order, and 984,080 within-turn states, enumerated at 19,100 states/s.

### Reading it

- **The distribution is heavy-tailed and grows with the game.** The median turn visits 224 states and three quarters stay under 2,500, but the 90th percentile is 29,000 and 7% pass 100,000. The opening is trivial (turns 1-5, under 120 states at the 90th percentile); from turn 21 on, a quarter of turns pass 100,000, and those late turns, with big decks, many bases and lethal in reach, are where within-turn tactics matter most.
- **Ends are most of the work, and two thirds of them are pile order.** Within-turn states are about as many as ends (1.3x over complete turns), so a canonical ordering of commuting actions can't shrink a turn by more than its within-turn share plus the ends it merges. Most ends that differ do so only in the order of a pile: up to pile order there are 2.9x fewer (3.6x at the 90th percentile), because playing the same cards in another order leaves the discard pile in another order. A discard pile's order is a real difference in the true world (it decides the deck it is shuffled into), but not to a value that counts cards by zone, nor across sampled worlds, which reshuffle anyway. So a canonical form saves a factor of about 3, not the factors of thousands the transposition table removed (Milestone 2b).
- **The cost on CPython** is about 50 µs per state visited: a 20,000-state plan takes about 1 s.

### Recommended default budget

**20,000 states per plan, shared by the worlds**, enumerated depth first with the greedy policy's preferred moves first (design, Search). It lists every end of 88% of turns, and of 96% before turn 16, for about 1 s per plan on CPython, so 1-3 plans per turn cost at most a few seconds a turn (the median turn's plan takes 10 ms): playable for one tournament stopped by a sequential test, and about the per-decision cost the Rust engine has to cut for self-play. A smaller budget leaves too many mid-game turns unfinished (5,000 misses 20% of all turns and 34-54% from turn 16), and a larger one buys little: 100,000 still misses 7%, at five times the cost. With N worlds, each world after the shared part before the first chance event gets its share (5,000 at N = 4, which finishes 80% of turns).

**The budget binds often enough in the late game (30-50% of turns from turn 21 at 20,000) that the exact fix comes first**, as the design says: canonical orderings of commuting actions, together with merging ends up to pile order, measured against this table. Expect about a factor of 3, which turns the 20,000 budget into one that finishes roughly the turns under 60,000 states today (between the 88% and 93% finished at 20,000 and 100,000). It will not close the tail: the capped turns are at least 100,000 states and some far more, so the planner must still play its best end found so far when the budget runs out, and approximate search within the turn is for those turns.

## Milestone 6: the planner's cost

- Date: 2026-10-21
- Commit: the merge of #133's PR.
- Command: `python -m minimax_games.games.star_realms tournament --bots planner greedy --games 10 --jobs 2`: a smoke test of the whole-turn planner at its defaults (4 worlds, 20,000 states per plan shared by the worlds, linear evaluation), not the milestone's match (#134).
- Machine: Apple M1 MacBook, 8 GB, macOS (Darwin 25.2), CPython 3.14.0, shared with other jobs.

| Bot | Wins | As 1st | As 2nd | ms/move |
|---|---|---|---|---|
| planner | 9/10 | 4/5 | 5/5 | 69 |
| greedy | 1/10 | 0/5 | 1/5 | 0.007 |

**2.58 plans per turn, 157 ms per plan** (all the planner's time per decision over its plans), 22.0 turns per game, 29 s wall time. Plans per turn include the plans made again when automatic play went otherwise than in the sampled worlds (design, Search: the whole-turn planner). A plan costs far less than the 1 s the turn-size measurement put on a 20,000-state plan: most plans are of small turns, or of the part of a turn after a draw, and merging states up to pile order visits a third fewer (on the turns of 4 seeded heuristic games at the default budget, 406,100 states instead of 611,227, and 87 of 100 turn-start plans finish instead of 78, at the same time). Not measured on PyPy: numpy, a core dependency since #135, publishes no PyPy wheels, and `task pypy` is gone (#148); the Rust engine is the route to speed.

## Milestone 6: planner vs heuristic

- Date: 2026-10-21
- Commit: `071deb4` (main after #155), no code changes. The planner at its defaults: 4 worlds, 20,000 states per plan shared by the worlds, the linear `evaluate`, the Python enumeration (the Rust one needs a learned value).
- Machine: Apple M1 MacBook, 8 GB, macOS (Darwin 25.2), CPython 3.14.0, **shared** with another agent's builds and tests (load average 4-4.5 by the end), so the timings are contended.
- Seeds: from 0, the seeds of ISMCTS's recorded run against the heuristic (Milestone 5: the ladder, seeds 0-499). Alternating seats, max 300 turns, `--jobs 2`.

Two sequential tests on the same seeds, each a question about the planner's win rate against the heuristic. Games are deterministic, so the second test's first 112 games are the first test's.

1. **Is it above ISMCTS?** H0 35% (ISMCTS's 34.6%) against H1 45%, a gain of 10 points, which would say the micro-action tree, not the leaf, held ISMCTS back.
2. **Is it above the heuristic?** H0 45% against H1 55%, around level.

```sh
uv run python -m minimax_games.games.star_realms tournament --bots planner heuristic \
    --games 500 --base-seed 0 --jobs 2 --sprt --h0 0.35 --h1 0.45
uv run python -m minimax_games.games.star_realms tournament --bots planner heuristic \
    --games 500 --base-seed 0 --jobs 2 --sprt --h0 0.45 --h1 0.55
```

alpha and beta 0.05 both (bounds ±2.94). No game was unfinished.

| Test | Decision | Games | Planner wins | Win% | 95% interval | As 1st | As 2nd | ms/move | Plans per turn, ms per plan | Wall time |
|---|---|---|---|---|---|---|---|---|---|---|
| H0 35% vs H1 45% | **H1 accepted** (LLR 3.05) | 112 (seeds 0-111) | 52/112 | 46.4% | 37.5-55.6% | 33/56 | 19/56 | 68.1 | 2.52, 154 | 4.3 min |
| H0 45% vs H1 55% | **H0 accepted** (LLR -3.01) | 445 (seeds 0-444) | 215/445 | **48.3%** | **43.7-53.0%** | 121/223 | 94/222 | 80.1 | 2.52, 178 | 20.0 min |

Mean 21.9 and 22.5 turns. The first player won 70/112 and 249/445. The heuristic's moves take 0.01 ms.

**Against ISMCTS**: its recorded per-seed results weren't committed (the pickles of Milestone 5: ISMCTS stayed in a scratch directory that no longer exists), so the comparison is of rates, not paired by seed: the planner's 215/445 (48.3%) on seeds 0-444 against ISMCTS's 173/500 (34.6%) on seeds 0-499, a gain of 13.7 points (two-proportion z = 4.3, p < 0.0001). The seeds overlap almost entirely, so the seed range isn't the difference (it moved the cheat by 8 points between ranges, Milestone 5: the ladder). For scale on the same seeds, the cheat, which searches the true state, won 52.0% and the sampler 33.4%.

**Time per decision**: 68-80 ms per decision over all decisions, 154-178 ms per plan, 2.52 plans per turn (contended, Python enumeration): about ISMCTS's and the sampler's 90-110 ms per decision (on PyPy, contended), so the planner isn't buying its gain with more compute. No cut in worlds or budget was needed: the first test took 4 minutes, the second 20.

### Reading it

- **The tree wasted the budget; the leaf was not the main limit.** With the same hand-valued `evaluate` that ISMCTS's leaves and the sampler's searches used, scoring whole ends of turn lifts a legitimate bot from 34.6% to 48.3% against the heuristic, at about the same time per decision. So most of the gap was the micro-action search spending its budget on orderings within a turn, not the evaluation being too weak to judge a position.
- **The planner plays level with the heuristic, within a few points of the cheat.** H0 at 45% vs 55% says it is nearer 45% than 55%; its interval, 43.7-53.0%, includes 50%. It is the strongest legitimate bot on the ladder, and the cheat's 52.0% on the same seeds, with the true state and a 2-turn horizon, is now only about 4 points ahead.
- **For milestone 8**: planner plus `evaluate` at about 48% is the bar a learned value has to beat, and since the leaf is now what's left to improve, the learned values' better held-out log-loss (0.550 and 0.562 against `evaluate`'s 0.607, Milestone 8: offline value fit) is the right lever to try next: the planner with each value against the planner with `evaluate` (#136), with the Rust enumeration, which the learned values can use, about 14 times faster.

## Milestone 8: offline value fit

- Date: 2026-10-21
- Commit: `9057198` (#135, the first commit of PR #146 on the `value-fit-135` branch): the feature encoder (`features`), the models and their fitting (`value`) and the data and gate (`fit`, the `fit` command). The committed models were trained by that commit's code with the command below.
- Interpreter: CPython 3.14.0 (`uv run`), numpy 2.5.4.
- Machine: Apple M1 MacBook, 8 GB, **shared** with three other jobs; at most 2 processes here.
- Seeds: fit settings chosen on **10000-19999** (scratch runs, below), the result measured once on **20000-29999**, clear of every tournament's seeds.

### What was fitted

V(s), the probability that the player who has just ended their turn goes on to win, at the states `EndTurn` leaves (before the next player's automatic play), from the turn-boundary states of seeded heuristic-vs-heuristic games, each labelled with that player's result (Monte Carlo targets; no randomness beyond the deal, `--epsilon 0`).

- **Features** (`features.encode`, 190 per state): per player, the player who just ended first, a count of each of the 49 cards in their draw pool (deck, hand, discard pile and cards in play: the hand was just drawn from the deck, so which cards it holds is deck order), a count of each of the 18 bases in play, Authority, base defense and the discards they owe; then the turn and a count of each card in the Trade Row. Standardised by the training data's mean and spread, stored with the model.
- **Logistic regression** (191 parameters) and a **network with one hidden layer of 32 tanh units** (6,145 parameters), hand-written in numpy (`value.Logistic`, `value.Mlp`), loss the mean log-loss plus `l2 / 2` times the squared weights, gradients checked against central finite differences. Both fitted by Adam (learning rate 0.001, minibatches of 256, 10 epochs, `l2` 0.003, seed 0), about 1.5 and 2.5 s each.
- **The gate**: mean log-loss on the held-out games' states, against `sigmoid(evaluate / 16)` for the same player on the same states, and against a constant (the training games' share of wins by the player who just ended).

### Result (seeds 20000-29999)

```sh
uv run python -m minimax_games.games.star_realms fit --jobs 2 \
    --save src/minimax_games/games/star_realms/models
```

(the defaults: `--games 8000 --test-games 2000 --base-seed 20000 --seed 0 --epochs 10 --batch 256 --learning-rate 0.001 --l2 0.003 --hidden 32 --epsilon 0`). 8,000 training games (seeds 20000-27999), **189,901 states**; 2,000 held-out games (seeds 28000-29999), **47,311 states**; every game finished. 40 s wall time (76 s CPU, nearly all of it generating the games), about 1 GB peak memory.

| Predictor | Held-out log-loss |
|---|---|
| constant | 0.6930 |
| `sigmoid(evaluate / 16)` | 0.6068 |
| logistic regression | **0.5621** |
| one hidden layer, 32 units | **0.5497** |

**Both models beat `evaluate`**, by 0.045 and 0.057 nats per state, so both are committed: `src/minimax_games/games/star_realms/models/logistic.npz` and `mlp.npz` (`value.load("logistic")`, `value.load("mlp")`, wrapped as an evaluator by `value.ValueEvaluator`). The run is deterministic: rerunning it with `--jobs 1` writes byte-identical model files. `evaluate`'s 0.607 agrees with the 0.606 its scale was fitted to (Milestone 5: rollouts), on other games' turn starts.

What the logistic regression learned, per raw unit: a point of Authority is worth 0.080 log-odds to the player who just ended and -0.075 to the opponent, and each Scout still owned costs 0.31, about 4 points of Authority (`evaluate` charges 1, plus what it dilutes the deck's average). The cards worth most are the big ships, Command Ship (1.2 per copy), Dreadnaught, Battle Blob and Flagship (0.75-1.0); a Cutter is worth 0.35.

So one weight per card, fitted to outcomes, already does much better than `evaluate`'s hand-set weights, and a small network does better again: the linear form is not the whole limit. Whether that carries over to play is the next item of Milestone 8, a tournament of the planner with each value.

### Choosing the fit settings (seeds 10000-19999)

Scratch runs on 8,000 training games (seeds 10000-17999, 189,914 states) and 2,000 validation games (18000-19999, 47,695 states; `evaluate` 0.6048), all seed 0, Adam at 0.001, minibatches of 256:

| Model | Epochs | `l2` | Train | Validation |
|---|---|---|---|---|
| logistic | 10 | 0 | 0.5601 | 0.5557 |
| logistic | 10 | 0.003 | 0.5607 | 0.5567 |
| 32 units | 20 | 0.001 | 0.4852 | 0.5642 |
| 32 units | 10 | 0.003 | 0.5292 | **0.5420** |
| 32 units | 20 | 0.003 | 0.5189 | 0.5416 |
| 32 units | 40 | 0.003 | 0.5129 | 0.5439 |
| 32 units | 20 | 0.01 | 0.5539 | 0.5538 |
| 32 units | 20 | 0.03 | 0.5795 | 0.5770 |
| 16 units | 20 | 0.003 | 0.5201 | 0.5424 |
| 64 units | 20 | 0.003 | 0.5191 | 0.5430 |

The network overfits easily: with `l2` 0.0001 and 30 epochs it scored 0.684 on 1,000 held-out games, worse than `evaluate`, and with half the training data (95,000 states) its best was 0.561, behind the logistic regression's 0.559. `l2` 0.003 is the best penalty tried, and the width matters little; 10 epochs were adopted, as good as 20 at half the time, for both models (the logistic regression barely changes with `l2`). More games would likely help the network most, and cost only seconds per thousand.

## Milestone 8: the planner with a learned value

- Date: 2026-10-21
- Commit: `071deb4` (main after #155) with this PR's CLI change (#136): `planner:<model>` names the planner scoring with a committed learned value (`value.load(<model>)`), on the Rust enumeration. The planner's moves are unchanged by it; the merge of #157 (Rust self-play) since then doesn't touch the planner.
- Machine: Apple M1 MacBook, 8 GB, macOS (Darwin 25.2), CPython 3.14.0, **shared** with another agent, so timings are contended.
- Seeds: from 0, as in Milestone 6: planner vs heuristic. Alternating seats, max 300 turns, `--jobs 2`, alpha and beta 0.05 (bounds ±2.94).

The learned value is the **mlp** (one hidden layer of 32 units, the better of the two committed models offline: held-out log-loss 0.550 against the logistic regression's 0.562 and `evaluate`'s 0.607; Milestone 8: offline value fit). Both planners have the defaults: 4 worlds, 20,000 states per plan. The one on `evaluate` uses the Python enumeration (the Rust one needs a learned value); the backends list the same ends in the same order (Milestone 7: Rust planner enumeration), so only the value differs.

```sh
uv run python -m minimax_games.games.star_realms tournament --bots planner:mlp planner \
    --games 500 --base-seed 0 --jobs 2 --sprt --h0 0.5 --h1 0.6
uv run python -m minimax_games.games.star_realms tournament --bots planner:mlp heuristic \
    --games 1000 --base-seed 0 --jobs 2 --sprt --h0 0.5 --h1 0.6
uv run python -m minimax_games.games.star_realms tournament --bots planner:mlp heuristic \
    --games 500 --base-seed 0 --jobs 2
```

The hypotheses: H0 50% (no better) against H1 60%, a gain of 10 points, for both questions: does the learned value beat `evaluate` in the planner, and does the learned planner beat the heuristic. The last match, against the heuristic over all of seeds 0-499, isn't a test: at 2 minutes it was cheap enough to give a rate comparable with the ladder's 500-game runs.

| Match (first bot's results) | Decision | Games | Wins | Win% | 95% interval | As 1st | As 2nd | ms/move | Plans per turn, ms per plan | Wall time |
|---|---|---|---|---|---|---|---|---|---|---|
| planner:mlp vs planner (`evaluate`) | **H1 accepted** (LLR 2.95) | 94 | 59/94 | **62.8%** | 52.7-71.9% | 34/47 | 25/47 | 6.19 (planner 69.1) | 2.40, 15.8 (planner 2.53, 146) | 3.8 min |
| planner:mlp vs heuristic | **H1 accepted** (LLR 2.97) | 83 | 53/83 | 63.9% | 53.1-73.4% | 32/42 | 21/41 | 7.59 | 2.47, 19.0 | 28 s |
| planner:mlp vs heuristic, seeds 0-499 | (fixed) | 500 | 345/500 | **69.0%** | 64.8-72.9% | 189/250 | 156/250 | 6.56 | 2.49, 16.1 | 2.1 min |

No game was unfinished. Mean 22.8, 24.2 and 24.4 turns.

### Reading it

- **The learned value beats `evaluate` in the planner**, 62.8% head to head (52.7-71.9%) after 94 games, at the same budget and worlds. The offline gate's ranking (log-loss 0.550 against 0.607) carried over to play.
- **The planner with the mlp is the first legitimate bot to beat the heuristic**, and clearly: 69.0% (64.8-72.9%) on seeds 0-499, against 48.3% for the planner on `evaluate` (seeds 0-444, Milestone 6: planner vs heuristic; z = 6.5, unpaired) and 34.6% for ISMCTS. It also beats the cheat's 52.0% on the same seeds, which sees every hidden card but scores with `evaluate`. Milestone 6 said the micro-action tree was the main limit and the leaf the next; with the tree replaced, a better leaf is worth about 20 more points against the heuristic.
- **It is about ten times faster too**, 6-8 ms per decision (Rust enumeration) against 68-80 ms for the Python planner on `evaluate`.
- **The mlp is the ladder's current best**: `planner:mlp`, the committed `src/minimax_games/games/star_realms/models/mlp.npz` (trained by the Milestone 8 offline value fit's command). It is the bar the rounds of policy iteration on the Rust engine (design, Milestone 8) have to beat, with the heuristic kept among the opponents. The logistic regression wasn't played; with its worse log-loss it isn't needed to name the best.

## Milestone 8: policy iteration

- Date: 2026-10-21
- Commit: `f5ccc00` (#162, the loop: `iterate`, `python -m minimax_games.games.star_realms iterate`; the run was made from that PR's branch head `9e48021`, the same tree).
- Machine: Apple M1 MacBook, 8 GB (4 performance and 4 efficiency cores), macOS (Darwin 25.2), CPython 3.14.0, numpy 2.5.4; no other agent running. 8 self-play threads.
- Settings: the agreed design's (design, Learned values: Policy iteration), all the command's defaults. Each round 5,000 games (4,000 training, 1,000 held out), mixed half the current best against itself, a quarter against the heuristic, a quarter against a champion from the pool; both forms refitted from scratch on the last 4 rounds' training games (`value.fit` defaults: seed 0, 10 epochs, batch 256, Adam at 0.001, `l2` 0.003, 32 hidden units); the offline gate on the round's held-out games; then the candidate's planner against the current best's (4 worlds, 20,000 states per plan, both in Rust), SPRT H0 50% against H1 55%, α = β = 0.05 (bounds ±2.94), at most 1,000 games; a promoted value then plays 500 games against the heuristic on seeds 0-499. Start: `mlp`, pool {`mlp`}.
- Seeds: round `r`'s training games 1,000,000 + 10,000 (r − 1) on, held-out 2,000,000 + 10,000 (r − 1) on, gating 3,000,000 + 10,000 (r − 1) on; the ladder 0-499; the confirmation below 4,000,000-4,000,499.

```sh
uv run python -m minimax_games.games.star_realms iterate --out runs/iterate --threads 8
```

Records: [policy-iteration/run-1/](policy-iteration/run-1/): `state.json` (every round's record), `run.log`, and the per-seed results (seed, first, second, winner, turns) of every gating match (`gate-rNN.csv`), the ladder match (`ladder-r03.csv`), `mlp`'s ladder match replayed in Rust (`ladder-mlp.csv`) and the confirmation (`confirm-pi03-mlp.csv`). No game in the run reached the 300-turn cap.

### Rounds

Log-loss is on the round's held-out states; the candidate is the form with the lower one (the network every round). The SPRT column is the candidate's record against the current best.

| Round | Champion (pool) | Training games, states | Held-out states | Fit on (rounds, states) | Log-loss: logistic / mlp / champion | Offline gate | SPRT | Promoted | vs heuristic, seeds 0-499 | Play / fit / gate / ladder / total (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | mlp (mlp) | 4,000, 92,936 | 23,217 | 1, 92,936 | 0.5549 / 0.5455 / 0.5498 | pass | **H0** after 87, 31-56 (35.6%) | - | | 7.8 / 0.0 / 0.2 / - / 8.0 |
| 2 | mlp (mlp) | 4,000, 93,121 | 23,338 | 1-2, 186,057 | 0.5350 / 0.5146 / 0.5276 | pass | **H0** after 179, 79-100 (44.1%) | - | | 8.6 / 0.1 / 0.6 / - / 9.2 |
| 3 | mlp (mlp) | 4,000, 92,872 | 23,453 | 1-3, 278,929 | 0.5422 / 0.5149 / 0.5415 | pass | **H1** after 185, 112-73 (60.5%) | **pi03** | 351/500 (70.2%) | 8.7 / 0.1 / 0.5 / 0.7 / 9.9 |
| 4 | pi03 (mlp, pi03) | 4,000, 92,268 | 23,132 | 1-4, 371,197 | 0.5317 / 0.5045 / 0.5056 | pass | **H0** after 487, 241-246 (49.5%) | - | | 9.4 / 0.1 / 1.1 / - / 10.6 |
| 5 | pi03 (mlp, pi03) | 4,000, 92,430 | 23,199 | 2-5, 370,691 | 0.5675 / 0.5386 / 0.5435 | pass | **H0** after 87, 31-56 (35.6%) | - | | 9.7 / 0.1 / 0.3 / - / 10.0 |
| 6 | pi03 (mlp, pi03) | 4,000, 93,007 | 23,298 | 3-6, 370,577 | 0.5596 / 0.5346 / 0.5362 | pass | **H0** after 177, 78-99 (44.1%) | - | | 9.8 / 0.1 / 0.6 / - / 10.5 |

**Stopped after round 6: 3 rounds in a row without a promotion.** 58 minutes in all (59 wall), within the 2-hour cap. A round took 8-11 minutes, nearly all of it self-play (10.7 games/s in round 1, slowing round by round to 8.5 in round 6 with the same mix of games, probably the laptop throttling over an hour at full load); the fits took 2-6 s and the matches 10-70 s.

### The promoted value, pi03

`src/minimax_games/games/star_realms/models/pi03.npz` (`value.load("pi03")`, `planner:pi03`): the network fitted in round 3 to rounds 1-3's 12,000 training games (278,929 states), all played by `mlp`'s planner. Two checks after the run, same planner settings, in Rust on 8 threads:

```python
from pathlib import Path
from minimax_games.games.star_realms import iterate, selfplay
from minimax_games.games.star_realms.value import load
players = {"mlp": selfplay.Planner(load("mlp")), "pi03": selfplay.Planner(load("pi03")), "heuristic": "heuristic"}
for a, b, seeds, name in [("mlp", "heuristic", range(500), "ladder-mlp"),
                          ("pi03", "mlp", range(4_000_000, 4_000_500), "confirm-pi03-mlp")]:
    named = iterate._alternating(a, b, seeds)
    played = selfplay.play([(s, players[x], players[y]) for s, x, y in named], threads=8)
    iterate._write_games(Path(f"{name}.csv"), named, played.winners, played.turns)
```


| Match (first bot's results) | Seeds | Wins | Win% | 95% interval | As 1st | Time |
|---|---|---|---|---|---|---|
| planner:pi03 vs planner:mlp | 4,000,000-4,000,499 | 290/500 | **58.0%** | 53.6-62.2% | 164/250 | 59 s |
| planner:mlp vs heuristic (replayed in Rust) | 0-499 | 345/500 | 69.0% | 64.8-72.9% | 189/250 | 27 s |
| planner:pi03 vs heuristic (the ladder match) | 0-499 | 351/500 | 70.2% | 66.0-74.0% | 193/250 | 39 s |

- `mlp`'s ladder match in Rust reproduces Milestone 8 (the planner with a learned value) exactly, 345/500 and 189 as first player, so the Rust self-play and the Python planner play the same games here too.
- **pi03 beats mlp head to head**, 60.5% in its gating match and 58.0% on 500 fresh seeds: the promotion holds up, it isn't the SPRT's 5% false positive.
- **Against the heuristic it is level with mlp**: 70.2% against 69.0% on the same seeds, and paired by seed, pi03 alone won 80 games and mlp alone 74 (271 both): no difference.

### Reading it

- **One step of improvement, then a plateau.** The loop found one value that beats `mlp` (by about 8-10 points head to head), and three more rounds, fitted on up to 16,000 games of `pi03`'s and `mlp`'s play, failed to beat `pi03`: 49.5%, 35.6% and 44.1%.
- **The offline gate passed every round, and predicted the match only once in six.** Every candidate had a lower held-out log-loss than the current best on its own round's games, yet four of the five matches went to H0, two of them clearly worse (35.6%). Log-loss on the current best's own states measures how well a value predicts outcomes where the current best goes; the planner instead takes the maximum over thousands of ends per turn, most of them states the current best would never reach, and a fit's errors there are what it exploits. The pre-gate costs nothing, so it does no harm, but as built it barely filters. Scoring a candidate on the planner's alternative ends (the states it rejects, labelled by playing them out) would be a gate that tracks play.
- **The candidates vary a lot between rounds.** Rounds 1 and 5 lost 35.6%, round 3 won 60.5%, with similar data. Fits from scratch on about 100,000-370,000 states of near-identical play give values whose games differ by up to 25 points, so a round's promotion is partly the luck of the fit. Fitting with several seeds, or averaging them, would steady it, as would larger windows.
- **No gain against the heuristic.** The heuristic is a quarter of every round's games, yet beating `mlp` didn't carry over to beating the heuristic more often. The improvement is in the mirror match, which is what self-play trains for.
- **What it suggests next**, as the design anticipates for a plateau (the form, not the data): the network's capacity and targets. Temporal-difference targets (or Monte Carlo targets averaged over several playouts per state) cut the label noise of single game results, which at about 90,000 states per round may be what makes the fits vary; a wider network or richer features; and a gate on the planner's off-policy ends. Self-play at about 9-11 games/s keeps each round near 10 minutes, so these can be tried as more runs of the same command.

## Milestone 8: ensemble candidates and policy-iteration run 2

- Date: 2026-10-21
- Commit: `42c7c08` (#166, value models of several members: `value.fit_ensemble`, `iterate --members`), issue #164.
- Machine: Apple M1 MacBook, 8 GB, macOS (Darwin 25.2), CPython 3.14.0, numpy 2.5.4; no other agent running. 8 self-play threads.

In run 1, candidates fitted on similar data won anywhere from 35.6% to 60.5% against the champion, so promotion looked partly like the luck of one fit. Here a candidate is an ensemble of k fits from seeds 0 to k − 1: its value is the mean of their win probabilities (design, Learned values: Ensembles).

### Spread of single fits against ensembles

Run 1's round data (`data/round-NN.npz`) were never committed and were gone with its working directory, so the measurement used run 2's round-1 data instead. That data comes from the same mix with `pi03` as champion (4,000 training games, 92,144 states; 1,000 held out, 22,998). Those games were played first and then reused by the run itself, so nothing was played twice. All fits were the mlp at the `value.fit` defaults. The 8 single fits used seeds 0-7. The ensembles drew on fits from seeds 8-23: four 4-member ensembles (8-11, 12-15, 16-19, 20-23) and two 8-member ones (8-15, 16-23). Each candidate's planner then played `pi03`'s (4 worlds, 20,000 states per plan, in Rust) over the same 300 fixed seeds, 5,000,000-5,000,299, with alternating seats. The script and per-seed results are in [policy-iteration/run-2/spread/](policy-iteration/run-2/spread/). The run took 21 s for the 24 fits and about 9 minutes for the matches.

| Candidates | Held-out log-loss (pi03: 0.5328) | Win% against pi03, each (300 games) | Mean | SD |
|---|---|---|---|---|
| 8 single fits | 0.5523-0.5589 | 32.0, 35.7, 28.3, 25.3, 26.0, 34.7, 31.7, 30.3 | 30.5% | 3.8 |
| 4 ensembles of 4 | 0.5470-0.5498 | 38.0, 32.0, 34.7, 36.0 | 35.2% | 2.5 |
| 2 ensembles of 8 | 0.5472-0.5473 | 33.3, 38.7 | 36.0% | 3.8 |

- At 300 games, a win rate near 33% has a binomial SD of 2.7 points. The single fits' 3.8 therefore puts the seed's own share at about 2.7 points, sqrt(3.8² − 2.7²). The ensembles of 4 vary no more than the binomial noise alone. **Averaging removes the seed's variance, but that variance was small**: a few points, not the 25-point range run 1 showed. Run 1's range mostly came from elsewhere: SPRT matches that stop early on extreme runs (87 games), and different data each round.
- **Ensembles are also stronger**: 35.2% against 30.5% (422/1,200 against 732/2,400, z ≈ 2.8), and their log-loss is about 0.007 lower. Eight members gave no more than four.
- **k = 4 was chosen.** The cost per planner move, measured on one thread with the planner against the heuristic over seeds 600-639 (k copies of `pi03`, so the games are the same):

| Members | Forward pass, µs per row | ms per plan | Relative |
|---|---|---|---|
| 1 | 1.76 | 7.2 | 1.0 |
| 2 | 3.18 | 8.8 | 1.2 |
| 4 | 6.67 | 12.6 | 1.75 |
| 8 | 12.55 | 19.4 | 2.7 |

The forward pass is linear in k; it takes about a fifth of a one-member plan's time. An ensemble champion would slow self-play by up to 1.75×. In this run the champion stayed `pi03`, a single fit, so only the matches paid: a 4-member candidate against `pi03` played about 1.4× slower than a single fit. Each fit of the run's four members took 5-23 s.
- Every candidate fitted to one round of data lost clearly to `pi03`, which was fitted to three (279,000 states). One round's data isn't enough to beat it, whatever the seed.

### Run 2

```sh
uv run python -m minimax_games.games.star_realms iterate --out runs/run-2 --start pi03 \
    --prefix pi2- --members 4 --threads 8
# after the stop at round 3, resumed (a departure from the design, below):
uv run python -m minimax_games.games.star_realms iterate --out runs/run-2 --start pi03 \
    --prefix pi2- --members 4 --threads 8 --patience 10
```

All other settings were the agreed design's and run 1's: rounds of 5,000 games (4,000 training, 1,000 held out), the same mix, a window of the last 4 rounds, both forms refitted from scratch, the offline gate, and SPRT H0 50% against H1 55% with α = β = 0.05 and at most 1,000 games. Seeds came from the same ranges as run 1's, so run 2's round r used the same seeds as run 1's round r, but with `pi03` playing. Start: `pi03`, pool {`pi03`}; candidates were named `pi2-NN` so they couldn't collide with run 1's names.

Records: [policy-iteration/run-2/](policy-iteration/run-2/), holding `state.json` (rounds 1-7, with the resumed run's `patience` of 10; `state-after-r3.json` is the record as the agreed rule left it), `run.log`, and the per-seed results of every gating match (`gate-r03.csv` to `gate-r07.csv`). No model was promoted, so there is no ladder match and no model to commit. No game reached the 300-turn cap.

| Round | Fit on (rounds, states) | Log-loss: logistic / mlp (4 members) / champion | Offline gate | SPRT against pi03 | Promoted | vs heuristic | Play / fit / gate / total (min) |
|---|---|---|---|---|---|---|---|
| 1 | 1, 92,144 | 0.5610 / 0.5482 / 0.5328 | **fail** | - | - | | 8.9 / 0.1 / - / 9.0 |
| 2 | 1-2, 184,799 | 0.5569 / 0.5282 / 0.5248 | **fail** | - | - | | 10.0 / 0.2 / - / 10.2 |
| 3 | 1-3, 277,191 | 0.5325 / 0.5041 / 0.5146 | pass | **undecided** at 1,000, 531-469 (53.1%, LLR 1.20) | - | | 10.0 / 0.3 / 3.2 / 13.4 |
| 4 | 1-4, 369,419 | 0.5349 / 0.5028 / 0.5094 | pass | **H0** after 299, 142-157 (47.5%) | - | | 9.6 / 0.4 / 1.1 / 11.1 |
| 5 | 2-5, 369,797 | 0.5650 / 0.5313 / 0.5447 | pass | **H0** after 364, 176-188 (48.4%) | - | | 9.6 / 0.4 / 1.1 / 11.1 |
| 6 | 3-6, 370,207 | 0.5478 / 0.5207 / 0.5267 | pass | **H0** after 116, 46-70 (39.7%) | - | | 9.7 / 0.4 / 0.4 / 10.5 |
| 7 | 4-7, 370,315 | 0.5382 / 0.5099 / 0.5208 | pass | **H0** after 651, 327-324 (50.2%) | - | | 9.9 / 0.4 / 2.5 / 12.8 |

**The agreed stopping rule stopped the run after round 3**: three rounds in a row without a promotion, where an undecided match counts as a failure. That took 33 minutes. **Rounds 4-7 come from resuming with `--patience 10`, a departure from the design**: the stopping rule was not the agreed one, and no one was asked before about 45 more minutes were spent. The run was then stopped by hand during round 8's match. Round 8 is discarded (the end of `run.log` shows its fit), and rounds 1-7 stand. The 7 rounds took 78 minutes. The design's stopping rule is unchanged.

### Reading it

- **Run 2 found nothing: every candidate failed.** The first two failed the offline gate. The third was undecided at the 1,000-game limit (53.1%, above 50% but short of the 55% that H1 asks). The four from the resume each went to H0. `pi03` stays the current best.
- **Ensembles didn't make the candidates steady.** With four members each, and windows of nearly identical size (about 370,000 states in rounds 4-7), the candidates' records against `pi03` ranged from 39.7% to 53.1%, about as wide as run 1's spread among rounds that played the same champion. Per the measurement above, the seed's share of that spread was only a few points. The rest comes from which games are in the window, plus the SPRT's early stops at extreme records. So the variance between rounds is about the data and the label noise of single game results, not the fit's seed. Temporal-difference targets, or several playouts per state, attack the cause directly (#165).
- **The offline gate filtered the two rounds whose windows were too small**, and agreed with the match there, which run 1's never got the chance to show. From round 3 on, it passed every candidate, and none promoted. As in run 1, a lower log-loss on the champion's own states doesn't predict winning against it.
- Ensembles gave a measurable but small gain in strength (about 5 points in the measurement), at 1.75× the cost per move. They are kept as an option (`--members`); `iterate` still defaults to one member. The plateau at `pi03` holds with them.

## Milestone 8: temporal-difference targets, tested offline

- Date: 2026-10-21
- Commit: `c6989db` (#165, this PR's first commit: self-play records each state's `position` in its game, `fit.Dataset` too, and `td.targets` builds TD(λ) targets).
- Machine: Apple M1 MacBook, 8 GB, macOS (Darwin 25.2), CPython 3.14.0, numpy 2.5.4; no other agent running. 8 self-play threads.
- Seeds: training games 6,000,000-6,003,999, held-out games 6,100,000-6,100,999, the match 6,200,000-6,200,299. They are clear of every range in `iterate.Settings` (below 4,000,000 for up to 100 rounds), of pi03's confirmation (4,000,000-4,000,499) and of the ensemble measurement (5,000,000-5,000,299).

The idea (design, Learned values: Targets) tested the way the design now asks of a new idea: one fixed dataset of the champion's games, candidates fitted to it, and a short fixed-seed match of each against the champion, with no policy-iteration run. A TD(λ) target is the λ-return of a turn-boundary state: `G_t = (1 − λ)(1 − V(s_t+1)) + λ(1 − G_t+1)`, back from the game's end, where `s_t+1` is the next turn-boundary state of the same game (left by the opponent, hence `1 − ...`), the last state of a game takes the game's result, and V is the current value, pi03. λ = 1 gives the Monte Carlo targets.

```sh
uv run python docs/td-targets/td.py runs/td-165
```

- **The dataset**: 5,000 games of pi03's planner under policy iteration's mix (half pi03 against itself, a quarter against the heuristic, a quarter against a champion drawn from {mlp, pi03}; `iterate.mix`), 4,000 training games (**92,596 states**) and 1,000 held out (**22,858 states**). Every game finished. 8.6 minutes of play.
- **The candidates**: for each λ in 0, 0.5, 0.8 and 0.95, and Monte Carlo (λ = 1), a 4-member mlp ensemble (`value.fit_ensemble`, seeds 0-3, `value.fit` defaults), fitted to the training states with that target. About 4 s each.
- **The match**: each candidate's planner against pi03's (4 worlds, 20,000 states per plan, both in Rust), on the same 300 seeds, seats alternating. About 55 s each.

Script, log, the dataset's record and the per-seed results (seed, first, second, winner, turns) of every match: [td-targets/](td-targets/). The dataset itself (`dataset.npz`, 91 MB: rows, labels, game, position, training split) isn't committed; it is regenerated by the script from its seeds.

| Target | Held-out log-loss (Monte Carlo labels; pi03 0.5161) | Wins against pi03 (300) | Win% | 95% interval | As 1st | Paired with Monte Carlo: won only by it / only by MC |
|---|---|---|---|---|---|---|
| TD(0) | 0.5252 | 137 | 45.7% | 40.1-51.3% | 85/150 | 55 / 36 |
| TD(0.5) | 0.5244 | 134 | 44.7% | 39.1-50.3% | 86/150 | 51 / 35 |
| TD(0.8) | **0.5232** | 144 | **48.0%** | 42.4-53.6% | 85/150 | 66 / 40 |
| TD(0.95) | 0.5245 | 141 | 47.0% | 41.4-52.7% | 86/150 | 54 / 31 |
| Monte Carlo (λ = 1) | 0.5294 | 118 | 39.3% | 34.0-45.0% | 73/150 | - |

(Wilson intervals. The paired column counts the seeds the candidate won and the Monte Carlo candidate lost, and the reverse.)

13.4 minutes of compute in all (803 s wall): 8.6 for the dataset, 0.3 for the fits, 4.5 for the matches. **No candidate beat pi03**, so there was no confirmation and nothing is promoted; `pi03` stays the current best.

### Reading it

- **TD targets beat Monte Carlo targets on the same data.** Every TD candidate did better against pi03 than the Monte Carlo one: 44.7-48.0% against 39.3%, together 556/1,200 (46.3%). Paired by seed, TD(0.8) and TD(0.95) won 26 and 23 more games than Monte Carlo (sign tests z ≈ 2.5), TD(0) and TD(0.5) 19 and 16 (z ≈ 2.0 and 1.7). Their held-out log-loss on game results is lower too, by 0.004-0.006, the first time in this milestone that the two measures agreed. The Monte Carlo candidate's 39.3% is close to the 4-member ensembles fitted to one round in the ensemble measurement (35.2%), so the data and fit are comparable with that one.
- **The λ hardly matters between 0 and 0.95**: 44.7-48.0% is within the binomial noise of 300 games (SD about 2.9 points). 0.8 scored best on both measures.
- **But none beats pi03**: the best, 48.0%, has an interval of 42.4-53.6%, and pi03 was fitted to three times as much data (279,000 states). The design's bar for a run, a candidate that beats the champion in the short match, is not met.
- **Part of the gain may be pi03 itself, not less noise.** The bootstrap values are pi03's, so a TD target partly copies pi03's opinion of the next state, and a candidate that copied pi03 exactly would score 50% against it. The lower λ, the more it copies, yet TD(0) did no better than TD(0.95), whose targets are 95% game results, so most of the gain probably is the noise reduction. This experiment can't fully separate the two.
- **What it suggests**: TD targets, with λ about 0.8, are a better way to label the same games, worth about 7 points against the champion at one round's data. Adopting them in `iterate` (bootstrapping from the current best, on the window of 4 rounds) is the follow-up the issue reserves: on that much data the candidates' starting point is close to pi03's, and 7 points could be enough to get past it. That would be a run, and is not part of this issue.

## Milestone 8: policy-iteration run 3, TD targets

- Date: 2026-10-21
- Commit: `6709f21` (#171, `iterate --targets td --lam L`), issue #170.
- Machine: Apple M1 MacBook, 8 GB, macOS (Darwin 25.2), CPython 3.14.0, numpy 2.5.4; no other agent running. 8 self-play threads.

The follow-up to the offline test above: TD(0.8) targets in `iterate`, at the data size pi03 was fitted to. Each round, every window round's training targets were rebuilt as TD(0.8) returns from the current best's values of that round's states (design, Learned values: Temporal-difference targets). Held-out log-loss, and with it the offline gate, stayed on game results.

```sh
uv run python -m minimax_games.games.star_realms iterate --out runs/run-3 --start pi03 \
    --prefix pi3- --members 4 --targets td --lam 0.8 --threads 8
```

All other settings were `iterate`'s defaults, the agreed design's, including the short stopping rules: 3 rounds, or 3 in a row without a promotion, or 30 minutes. The candidates were 4-member ensembles as in run 2, and the start was `pi03`, pool {`pi03`}. The run was not resumed or extended.

Records: [policy-iteration/run-3/](policy-iteration/run-3/), holding `state.json`, `run.log` and the per-seed results of the one gating match (`gate-r03.csv`). Nothing was promoted, so there is no ladder match, no confirmation and no model to commit. No game reached the 300-turn cap.

**Same games as run 2.** The seeds are run 2's, and so are the champion (`pi03`, not promoted), the mix and the planner settings. So run 3 played exactly run 2's rounds 1-3: the same state counts (92,144, 184,799 and 277,191 in the windows) and the same champion log-losses. The only difference is the targets, which makes the two runs a paired comparison of Monte Carlo and TD(0.8) targets at one, two and three rounds of data. The gating seeds are shared too.

| Round | Fit on (rounds, states) | Log-loss: logistic / mlp (4 members) / champion | Run 2's mlp (Monte Carlo) | Offline gate | SPRT against pi03 | Promoted | Play / fit / gate / total (min) |
|---|---|---|---|---|---|---|---|
| 1 | 1, 92,144 | 0.5582 / 0.5375 / 0.5328 | 0.5482 | **fail** | - | - | 9.0 / 0.1 / - / 9.1 |
| 2 | 1-2, 184,799 | 0.5574 / 0.5262 / 0.5248 | 0.5282 | **fail** | - | - | 10.0 / 0.2 / - / 10.2 |
| 3 | 1-3, 277,191 | 0.5369 / 0.5115 / 0.5146 | 0.5041 | pass | **H0** after 255, 119-136 (46.7%, LLR −2.99) | - | 11.0 / 0.3 / 1.0 / 12.3 |

**Stopped after round 3: 3 rounds played**, and none promoted. It took 31.6 minutes, as estimated beforehand from run 2's rounds (about 10-13 minutes each). Play slowed from 9.3 to 7.5 games/s across the rounds, the same slowdown run 1 showed. Building the TD targets (pi03's forward pass on up to 277,000 states) added a few seconds to each fit.

**Paired with run 2's round 3** (seeds 3,020,000-3,020,254, the first 255 games of both gating matches, against the same pi03): TD's pi3-03 won 119, Monte Carlo's pi2-03 won 130 (51.0%). 28 seeds were won only by TD and 39 only by Monte Carlo, a sign test of z ≈ −1.3. Run 2's match went on to 1,000 games and ended undecided at 53.1%.

### Reading it

- **TD targets didn't beat pi03, and didn't beat Monte Carlo targets on the same data.** The one candidate that reached a match lost it (46.7%, H0 after 255 games). On the same 255 seeds the Monte Carlo candidate fitted to the same games did slightly better, 51.0%. The difference isn't significant, but it points the wrong way for the offline gain to have carried over.
- **The offline result didn't scale with the data.** At one round of data, the offline test had TD(0.8) about 9 points ahead of Monte Carlo against pi03 (48.0% against 39.3%), with a lower log-loss on game results. Here TD's held-out log-loss was lower at one round (0.5375 against 0.5482), about level at two (0.5262 against 0.5282), and higher at three (0.5115 against 0.5041). A likely reading: with little data, TD's bootstrapped targets cut noise that a Monte Carlo fit can't average away, but with three rounds of data the averaging happens anyway, and the bootstrap's bias toward pi03's own values (its errors included) is what remains. The design's warning that TD partly copies the champion fits that.
- **The offline gate filtered rounds 1 and 2 again**, as in run 2: the candidates fitted to one or two rounds had a higher log-loss than pi03 on its own states.
- **`pi03` stays the current best**, after three runs and 16 candidates. TD targets stay an option (`--targets td`), and `iterate` still defaults to Monte Carlo. Per the design's stopping rule, the run is not resumed. The plateau now points away from the targets and the fit's seed, and toward the form (a wider network, richer features) or toward a gate that tracks play (the planner's off-policy ends), as run 1's reading suggested.

## Milestone 8: value capacity and hand-made features, tested offline

- Date: 2026-10-21
- Commit: this PR (#173; on `2bf5c61`): `features.extra` and `features.EXTRA`, `value.Model.extra` and `fit(..., extra=)`, and the Rust forward pass reading them (`value::Extra`).
- Machine: Apple M1 MacBook, 8 GB, macOS (Darwin 25.2), CPython 3.14.0, numpy 2.5.4; no other agent running. 8 self-play threads.
- Seeds: the match 7,000,000-7,000,299, clear of every earlier range (`iterate.Settings`, below 4,000,000 for up to 100 rounds; 4,000,000-4,000,499; 5,000,000-5,000,299; 6,000,000-6,200,299). Planner cost: seeds 600-639 against the heuristic, as in the ensemble measurement.

Sixteen candidates over three runs reached the same plateau below `pi03`, and the design says a plateau points at the value's form or its features. This tests both offline, as the design asks of a new idea: #165's fixed dataset of pi03's games (`runs/td-165/dataset.npz`, reused, not regenerated: 4,000 training games, **92,596 states**, from seed 6,000,000; 1,000 held out, **22,858 states**, from 6,100,000), one candidate per form, and a short fixed-seed match of each against pi03.

```sh
uv run python docs/capacity/capacity.py runs/capacity-173 runs/td-165/dataset.npz
```

- **Every candidate** is a 4-member mlp ensemble (`value.fit_ensemble`, seeds 0-3, the `value.fit` defaults but for the width: 10 epochs, batch 256, Adam at 0.001, `l2` 0.003), fitted to the training states' game results (Monte Carlo targets).
- **The baseline**, `w32`, is today's form (32 hidden units), refitted and replayed on the new seeds so every comparison is paired. It is the same fit as #165's Monte Carlo candidate (held-out log-loss 0.5294 again); on #165's seeds it won 118/300, here 122/300.
- **The hand-made features** (`features.EXTRA`, 28 of them, computed by `features.extra` from the 190 counts alone, so the dataset needn't be regenerated): per player, *combat per hand* (5 × the pool's primary combat / the pool's size, plus the bases in play's combat), the *kill threat* (that less the other player's Authority plus base defense), and per faction the pool's *density* of it, the *expected same-faction pairs* in a hand of 5 (10 · n(n − 1) / (N(N − 1)); ally abilities fire on pairs), and the faction's *bases in play*. A model reading them (`value.Model.extra`) appends them to each row before standardising; both forward passes compute them identically (the Rust one from card tables Python hands over, `features.extra_tables`), checked on real rows and on whole planner games against the Python planner.
- **The match**: each candidate's planner against pi03's (4 worlds, 20,000 states per plan, both in Rust), on the same 300 seeds, seats alternating. **Planner cost**: ms per plan of the candidate's planner, on one thread, against the heuristic over seeds 600-639.

Script, log, summary and per-seed results (seed, first, second, winner, turns) of every match: [capacity/](capacity/).

### Step 1: is today's form underfitting?

| | Training log-loss | Held-out log-loss | Gap |
|---|---|---|---|
| `w32` (4 members, today's form) | 0.5056 | 0.5294 | 0.024 |
| pi03 (fitted to other games, for reference) | 0.5280 | 0.5161 | - |

**No.** The training loss is well below the held-out loss: the fit already finds more in its training states than carries over to new games, so more capacity has little to work with. The gap, 0.024, is four times the spread between the widths below.

### Steps 2 and 3: width and features

| Candidate | Width | Extra features | Parameters (4 members) | Training log-loss | Held-out log-loss (pi03 0.5161) | Wins against pi03 (300) | Win% | 95% interval | As 1st | Paired with `w32`: won only by it / only by `w32` (sign z) | ms per plan | Fit / match (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `w32` (baseline) | 32 | - | 24,580 | 0.5056 | 0.5294 | 122 | 40.7% | 35.3-46.3% | 75/150 | - | 10.3 | 3.8 / 41 |
| `w64` | 64 | - | 49,156 | 0.5053 | 0.5278 | 127 | 42.3% | 36.9-48.0% | 75/150 | 39 / 34 (0.59) | 17.8 | 5.3 / 59 |
| `w128` | 128 | - | 98,308 | 0.5064 | 0.5263 | 117 | 39.0% | 33.7-44.6% | 67/150 | 43 / 48 (−0.52) | 27.7 | 7.5 / 96 |
| `f32` | 32 | 28 | 28,164 | **0.4897** | **0.5157** | **151** | **50.3%** | 44.7-56.0% | 86/150 | **64 / 35 (2.91)** | 14.9 | 4.4 / 59 |

(Wilson intervals. The sign z is (only it − only `w32`) / √(their sum).)

**Step 4 (two hidden layers) did not run.** Its condition was that width helped or step 1 showed underfitting, and neither holds. The note on #173 about tinygrad for training it therefore doesn't apply.

**Nothing is promoted.** `f32` drew level with pi03 (50.3%, interval 44.7-56.0%) but did not clearly beat it, so there was no confirmation on fresh seeds or against the heuristic. `pi03` stays the current best.

6.1 minutes of compute in all: 4.7 for the widths (281 s wall; fits 17 s, matches 3.3 minutes, timing 1.1 minutes) and 1.4 for the features (82 s), plus about a minute of tests and builds. No dataset was played.

### Reading it

- **Width doesn't help.** 64 and 128 hidden units won 42.3% and 39.0% against the baseline's 40.7%, both within the noise of the paired comparison (z 0.6 and −0.5), at 1.7× and 2.7× the planner's cost. Their held-out log-loss fell a little (0.5294, 0.5278, 0.5263), but their training loss didn't move (0.5056, 0.5053, 0.5064): at the fit's settings (10 epochs, `l2` 0.003) the extra units add no fit at all. With the training loss already well below the held-out one, the limit isn't the network's size.
- **The hand-made features help, clearly.** At width 32 they lowered both losses by about 0.015, the largest change in either measure this milestone has seen (TD targets gave 0.006), took the held-out log-loss below pi03's own on its states for the first time with one round of data, and won 29 more games than the baseline on the same seeds (64 against 35, sign test z ≈ 2.9). That is the strongest candidate fitted to one round's data so far: 50.3% against pi03, where the same data with today's features gives 40.7% and with TD targets gave at best 48.0%. They cost 1.45× the planner's time per move (the extra features and 28 more inputs per row).
- **Why features and not width**: the extra features are ratios and differences of counts (combat over the pool's size, pairs over the pool's size squared, combat against Authority). A tanh network over standardised counts can only approximate a division, and with this much data and regularisation it doesn't learn one; given the ratios directly, it uses them.
- **Not enough to promote**, by the design's bar: level with pi03 is not beating it. But pi03 was fitted to three times as much data (279,000 states), and in run 2 one round's data gave candidates 35-40% where three rounds gave 53%. Whether the features at the window's size beat pi03 is what a policy-iteration run would test; per the design, that run is a separate decision, not part of this issue.

## Milestone 8: hand-made features at pi03's data size, offline

- Date: 2026-10-22
- Commit: `49e01ea` (#174's merge; nothing in the package changed for the test), issue #175.
- Machine: Apple M1 MacBook, 8 GB, macOS (Darwin 25.2), CPython 3.14.0, numpy 2.5.4. 8 self-play threads.
- Seeds: the match 8,000,000-8,000,299, the gate 8,100,000 on (it used 8,100,000-8,100,548), the confirmation 8,200,000-8,200,499, the ladder 0-499. All are clear of every earlier range (`iterate.Settings`, below 4,000,000 for up to 100 rounds; 4,000,000-4,000,499; 5,000,000-5,000,299; 6,000,000-6,200,299; 7,000,000-7,000,299), and of each other.

#173's offline test, at the data size pi03 was fitted to. In #173, `features.EXTRA` drew a candidate fitted to one round of data level with pi03 (50.3%), and pi03 had seen about 3 times as much. No games were played for training and no policy-iteration run was made.

```sh
uv run python docs/features-window/window.py runs/features-175 runs/run-2/data
```

- **The data**: run 2's saved rounds 1-4 (`runs/run-2/data/round-01.npz` to `round-04.npz`, read in place): pi03's games under policy iteration's mix, **369,419 training states** (the window pi03's successors were fitted to in run 2's round 4) and **92,354 held-out states**, the four rounds' held-out sets together. Round 8, incomplete, wasn't used. pi03 was fitted to other games (run 1's), so its log-loss here is out of sample on both sets: 0.5210 on the training states, 0.5204 held out.
- **The candidates**, each a 4-member mlp ensemble (`value.fit_ensemble`, seeds 0-3, the `value.fit` defaults: width 32, 10 epochs, batch 256, Adam at 0.001, `l2` 0.003), fitted to game results (Monte Carlo targets): `b32`, today's form, and `f32`, the same with `features.EXTRA`. `b32` is bit-identical to run 2's round-4 candidate `pi2-04` (same rows, same settings), which lost its gating match to pi03 (H0 after 299 games, 47.5%).
- **The match**: each candidate's planner against pi03's (4 worlds, 20,000 states per plan, both in Rust) on the same 300 seeds, seats alternating, compared seed by seed. The bar for gating: `f32`'s Wilson interval above 50%, or over 55% won.
- **The gate**, since `f32` cleared that bar: `iterate`'s SPRT, H0 50% against H1 55%, α = β = 0.05 (bounds ±2.94), at most 1,000 games, undecided failing, played in batches of 128 with the decision taken game by game in seed order. **The checks**, since it passed: 500 fresh seeds against pi03, and seeds 0-499 against the heuristic (the ladder).

Script, log, summaries and the per-seed results (seed, first, second, winner, turns) of every match: [features-window/](features-window/).

| Candidate | Extra features | Training log-loss (pi03 0.5210) | Held-out log-loss (pi03 0.5204) | Wins against pi03 (300) | Win% | 95% interval | As 1st | Fit / match (s) |
|---|---|---|---|---|---|---|---|---|
| `b32` (baseline) | - | 0.5022 | 0.5137 | 134 | 44.7% | 39.1-50.3% | 76/150 | 15 / 42 |
| `f32` | 28 | **0.4963** | **0.5088** | **169** | **56.3%** | **50.7-61.8%** | 98/150 | 19 / 50 |

(Wilson intervals.) **Paired by seed**, `f32` won 71 games `b32` lost and `b32` 36 that `f32` lost: a sign test of z = (71 − 36) / √107 ≈ **3.4**.

`f32` clearly beat pi03 by the bar, so the gate ran:

| Match (`f32`'s results) | Seeds | Result | Win% | 95% interval | As 1st | Time |
|---|---|---|---|---|---|---|
| The gate, SPRT against pi03 | 8,100,000-8,100,548 | **H1 after 549**, 303-246 (LLR 2.96) | 55.2% | 51.0-59.3% | | 119 s |
| Confirmation against pi03 | 8,200,000-8,200,499 | 270/500 | 54.0% | 49.6-58.3% | 147/250 | 93 s |
| The ladder, against the heuristic | 0-499 | 370/500 | **74.0%** | 70.0-77.7% | 196/250 | 71 s |

- Against pi03 over all three matches, on 1,349 different seeds: 742 won, **55.0%** (52.3-57.6%).
- On the ladder's seeds, paired with pi03's ladder match (Milestone 8: policy iteration, `ladder-r03.csv`, 351/500): `f32` alone won 80 and pi03 alone 61, z ≈ 1.6.
- No game reached the 300-turn cap.

**Promoted: `f32` is committed as `pf01`** (`src/minimax_games/games/star_realms/models/pf01.npz`, `value.load("pf01")`, `planner:pf01`), the new current best.

6.9 minutes of compute in all (413 s wall): 34 s of fits, 92 s for the two 300-game matches, 2.0 minutes for the gate, 1.6 for the confirmation and 1.2 for the ladder, plus a few seconds loading the data, and under a minute of tests.

### Reading it

- **At the window's size the features beat pi03.** The same data and fit without them gave 44.7% (and, as run 2's `pi2-04`, lost its gate); with them, 56.3%, and the paired gain, 35 games in 300 (z ≈ 3.4), is about what #173 measured on one round's data (29 games, z ≈ 2.9). The gain from the features and the gain from more data add up: one round with features was level with pi03, four rounds without them about 45%, four rounds with them about 55%.
- **The SPRT passed, and the fresh seeds agree on the size, if not the certainty.** 55.2% in the gate, then 54.0% on 500 fresh seeds, whose interval just reaches 50% (49.6%). An SPRT stopping on H1 tends to stop on a lucky run, so the confirmation, not the gate, is the unbiased estimate. Over all 1,349 games against pi03 it is 55.0% (52.3-57.6%), but the 300-game match was what chose to run the gate, so that pool is biased upward a little too. The fair reading: `pf01` is about 4-5 points better than pi03 head to head; smaller than pi03's step over `mlp` (58%), and real.
- **Against the heuristic it gains a little**: 74.0% against pi03's 70.2% on the same seeds (z ≈ 1.6, not significant alone). It's the first value since `mlp` with any sign of improving against the scripted bot.
- **Both losses fell with the features again**, by 0.006 training and 0.005 held out, less than the 0.015 on one round: with four times the data the plain form gets closer by itself, yet the match gain held. As before, log-loss understates what the planner gets from the value.
- **Costs**: by #173's measurements on seeds 600-639, a 4-member model with the features plans in about 14.9 ms, against 7.2 for a one-member model like pi03: about 2× per move (1.75× for the members, 1.45× for the features, measured separately on different baselines, so roughly), so self-play with `pf01` as champion will be slower. Not measured again here. The issue's last step, done in a second PR: the features as an option on `iterate` (`--extra`, off by default), without a run.

## Milestone 8: more hand-made features, and a cheaper pf01, offline

- Date: 2026-10-22
- Commit: this PR (#178; on `ee683c8`): `features.MORE` and `features.extra(x, more=True)`, `value.EXTRAS`, and the Rust forward pass reading them (`value::More`).
- Machine: Apple M1 MacBook, 8 GB, macOS (Darwin 25.2), CPython 3.14.0, numpy 2.5.4. 8 self-play threads; planner cost on 1.
- Seeds: the match 9,000,000-9,000,299 (the gate would have taken 9,100,000 on and the confirmation 9,200,000-9,200,499; neither ran). All are clear of every earlier range (`iterate.Settings`, below 4,000,000 for up to 100 rounds; 4,000,000-4,000,499; 5,000,000-5,000,299; 6,000,000-6,200,299; 7,000,000-7,000,299; 8,000,000-8,200,499), which the script asserts. Planner cost: seeds 600-639 against the heuristic, as in #173.

Two offline experiments on #175's data, with `pf01` as the opponent. No games were played for training and no policy-iteration run was made.

```sh
uv run python docs/more-features/more.py runs/features-178 runs/run-2/data
```

- **The data**: #175's, read in place: run 2's rounds 1-4 (`runs/run-2/data/round-01.npz` to `round-04.npz`), **369,419 training states** and **92,354 held out**. pf01 was fitted to these training states, so its training log-loss (0.4963) is in sample; its held-out one is 0.5088.
- **The more features** (`features.MORE`, 17, computed by `features.extra(x, more=True)` from the 190 counts alone and appended after `features.EXTRA`'s 28, so no data was regenerated). Per player: *trade per hand* and *Authority per hand* (5 × the pool's primary trade or Authority / its size, plus the bases in play's), the pool's densities of cards that *draw*, that *scrap* and that make the opponent *discard* (in any of a card's abilities, through choices), and of *bases*, and the *defense of their Outposts* in play. Then three leads of the player who just ended over the other: *Authority*, *combat per hand* and *trade per hand*. A model reads none, `EXTRA`, or `EXTRA` then `MORE` (`value.EXTRAS`); `pf01`'s saved form is unchanged, and a test pins its probabilities on fixed rows (to 1e-13, as matrix products may differ in the last bits across platforms). Both forward passes compute them identically (the Rust one from the tables `features.extra_tables(more=True)` adds under `more`), checked on real rows and on whole planner games against the Python planner.
- **The candidates**, mlps at width 32 (the `value.fit` defaults: 10 epochs, batch 256, Adam at 0.001, `l2` 0.003) fitted to game results, from fit seeds 4 on, since pf01 is the fit from seeds 0-3 on these rows and a refit from them would be pf01 itself: `p32`, pf01's form (`EXTRA`, 4 members, seeds 4-7), the pf01-equivalent refit and the baseline of every paired comparison; `m32`, the same with `MORE` too; `p32x1` and `p32x2`, pf01's form with 1 and 2 members (seeds 4, and 4-5). `fit_ensemble` fits each seed on its own under one standardisation, so these two are `p32`'s first members, taken rather than refitted.
- **The match**: each candidate's planner against pf01's (4 worlds, 20,000 states per plan, both in Rust) on the same 300 seeds, seats alternating, compared seed by seed with `p32`. The bar for gating `m32`: its Wilson interval above 50%, or over 55% won. **Planner cost**: ms per plan of each planner on one thread against the heuristic over seeds 600-639, one run each.

Script, log, summaries and the per-seed results (seed, first, second, winner, turns) of every match: [more-features/](more-features/).

| Candidate | Extra features | Members | Training log-loss (pf01 0.4963) | Held-out log-loss (pf01 0.5088) | Wins against pf01 (300) | Win% | 95% interval | As 1st | Paired with `p32`: won only by it / only by `p32` (sign z) | ms per plan (pf01 14.52) | Fit / match (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `p32` (pf01-equivalent) | 28 | 4 | 0.4962 | 0.5090 | 145 | 48.3% | 42.7-54.0% | 82/150 | - | 15.79 | 17 / 55 |
| `m32` (more features) | 45 | 4 | **0.4949** | **0.5084** | 148 | 49.3% | 43.7-55.0% | 88/150 | 32 / 29 (0.38) | 16.74 | 19 / 67 |
| `p32x2` | 28 | 2 | 0.4970 | 0.5093 | 140 | 46.7% | 41.1-52.3% | 81/150 | 25 / 30 (−0.67) | 11.96 | - / 60 |
| `p32x1` | 28 | 1 | 0.4986 | 0.5119 | 149 | 49.7% | 44.0-55.3% | 81/150 | 48 / 44 (0.42) | **8.25** | - / 50 |

(Wilson intervals. The sign z is (only it − only `p32`) / √(their sum). No game reached the 300-turn cap.)

**`m32` did not clearly beat pf01** (49.3%, interval 43.7-55.0%), so the gate and the checks did not run, and **nothing is promoted**: `pf01` stays the current best. The cheaper models are recorded only, as the issue asks.

6.1 minutes of compute in all (365 s wall): 37 s of fits, 3.9 minutes for the four 300-game matches, and the remaining 1.6 minutes timing the five planners, computing log-losses and loading the data; plus about two minutes of tests and builds.

### Reading it

- **The more features add nothing measurable.** They lowered both losses a little (by 0.0013 training and 0.0006 held out), a tenth of what `EXTRA` gave, and the match didn't move: 148 wins against `p32`'s 145 on the same seeds, 32 seeds won only with them and 29 only without (z ≈ 0.4). What they encode is mostly there already: trade and Authority per hand are ratios like the combat per hand `EXTRA` gave, and the leads are differences of features the network already reads. The density of draw, scrap and discard cards is new, but its weight in the loss is small. Cost: 1.06× `p32`'s time per plan for the 17 more inputs.
- **The pf01-equivalent refit is level with pf01** (48.3%, 42.7-54.0%), as it should be: a refit from other seeds is a sample of the same value, and this is the noise of 300 games against an equal.
- **One member is as strong as four here, at about 0.55× the cost.** `p32x1` won 49.7% against pf01 and drew level with its own 4-member ensemble seed by seed (48 against 44, z ≈ 0.4), in 8.25 ms per plan against pf01's 14.52 (0.57×) and `p32`'s 15.79 (0.52×). `p32x2` won 46.7% (25 against 30 with `p32`, z ≈ −0.7), at 12.0 ms. Neither difference is significant: with 300 games the match can only rule out gaps of about 6 points, so "level" here means within about 5 points of pf01, not equal to it. The averaging still shows in the losses (0.5119 held out for one member, 0.5093 for two, 0.5090 for four), but, as in run 2 and #173, held-out log-loss gains of this size haven't shown up as wins.
- **Timing noise**: `p32` has pf01's form and size yet timed at 15.79 against 14.52 ms per plan, so one 40-game timing run is good to about 10%. The ratios above are rough to that degree. By #173's numbers (7.2 ms for pi03's one member without the features), the features themselves cost about 1.15× at one member.
- **For the owner**: replacing pf01 with a one-member model would make policy iteration's self-play and gating with the champion about 1.8× cheaper at no measured loss of strength. Whether "level within 5 points on 300 games" is enough, or wants a longer match (an SPRT for H0 50% against H1 45%, say, to bound the loss), is the owner's call; nothing was replaced here.

## Milestone 9

The yardstick, and today's champion, `pf01`, on it ([the design](star-realms.md), Milestone 9). No player is trained; each subsection is one end-state test, and names the belief it tests.

### Milestone 9: more sampled worlds

- Date: 2026-10-22
- Commit: this PR (#186; on `b25066c`): `sampled_worlds` (the comparison, `Probe` and the report), `tournament.play_paired`, and the `worlds` command (`task worlds`).
- Machine: Apple M1 MacBook, 8 GB, macOS (Darwin 25.2), CPython 3.14.0. 2 processes. **Three other agents were working on the machine throughout**, so absolute times are inflated and noisy; the counts were timed interleaved (each plan at 4, 8 and 16 worlds one after another) and the match's two players in the same games, so their ratios are fair.
- Seeds: the probed games 18,600,000-18,600,059, the match 18,610,000-18,610,149 (each in both seats), clear of every earlier range.

**The belief** (How it chooses): *beyond the turn, hidden cards are chance, not strategy, so judging a few sampled worlds is enough.* Its test: how much the choice changes with more sampled worlds. Predicted: little, and no measurable strength from more.

```sh
uv run python -m minimax_games.games.star_realms worlds --games 60 --base-seed 18600000 --match-seeds 150 --match-base-seed 18610000 --jobs 2
```

- **The plans**: 60 games of `planner:pf01` against itself (Rust enumeration), both players probing. At every plan the champion makes, 16 worlds are sampled from its view once, and the position is planned from the first 4 (today's plan, the one it plays), the first 8 and all 16, at the same **5,000 states per world** (budgets 20,000, 40,000 and 80,000), so the larger counts add worlds to the 4. A line *changes* when it differs from the 4 worlds' line, compared up to the order of its plays (the position before its last move, up to pile order, and that move). As the yardstick for that, the 4-world plan is also made from worlds 5-8, a fresh sample disjoint from its own: how often that changes the line is the noise of sampling alone.
- **The match**: `planner:pf01` at 16 worlds (80,000 states) against itself at 4 (20,000), each of 150 seeds played twice, once with each first (`tournament.play_paired`), compared seed by seed.

| Turns | Plans | Changed: 4 fresh worlds | 8 worlds (incl. the 4) | 16 worlds (incl. the 4) |
|---|---|---|---|---|
| All | 3,558 | 11.3% | 6.7% | 8.7% |
| 1-5 | 660 | 9.4% | 6.2% | 8.3% |
| 6-10 | 697 | 11.2% | 5.9% | 7.7% |
| 11-15 | 765 | 12.3% | 7.2% | 9.8% |
| 16-20 | 722 | 12.7% | 8.2% | 10.0% |
| 21-25 | 461 | 11.9% | 7.2% | 8.5% |
| 26-30 | 225 | 8.4% | 4.0% | 6.2% |
| 31-35 | 28 | 10.7% | 7.1% | 7.1% |

(Turns as the game numbers them, both players'. ms per plan, interleaved: 17.8 at 4 worlds, 34.3 at 8, 69.7 at 16.)

| Bot | Wins | Win% | 95% interval | As 1st | As 2nd | ms per plan | Plans per turn |
|---|---|---|---|---|---|---|---|
| 16 worlds | 163/300 | 54.3% | 48.7-59.9% | 97/150 | 66/150 | 72 | 2.49 |
| 4 worlds | 137/300 | 45.7% | 40.1-51.3% | 84/150 | 53/150 | 17.9 | 2.47 |

Seeds won both / split / lost both by 16 worlds: 30 / 103 / 17 (sign z = (30 − 17) / √47 = 1.90). No game reached the 300-turn cap; mean 23.3 turns.

About 11 minutes wall in all, 6.7 of them (403 s) the match.

#### Reading it

- **The choice changes on about one plan in eleven with four times the worlds**, evenly through the game (8-10% in every band of turns with many plans), and **a fresh sample of 4 changes it more often than 16 do** (11.3%). So most of what changes is the noise of a 4-world sample: the plans where two lines are judged within sampling noise of each other. The 8-world line changes less than the 16-world one (6.7% against 8.7%) because half its worlds are the 4 it is compared with, against a quarter of the 16's; neither is a different kind of choice.
- **The match leans towards 16 worlds but doesn't settle it**: 54.3%, interval 48.7-59.9%, and 30 seeds won both against 17 lost both (sign z 1.90, about p = 0.06 two-sided). With 300 games it can only rule out gaps of about 6 points, so the effect of four times the worlds is somewhere between nothing and about 10 points, at **4.0× the time per plan** (72 against 17.9 ms).
- **On the belief**: the choice changes little (under a tenth of plans, mostly near-ties), as predicted; whether those near-ties are worth a few points of strength is not decided by this match. If they are, it is the cheapest kind of strength there is (no training), and the same budget spent on 4 worlds at 4× the states per world would be the fair comparison before changing the default, which is out of this slice's scope (a decision on this result, in its own PR). The obvious next measurement is an SPRT of 16 against 4 worlds, H0 50% against H1 55%.

### Milestone 9: the reference population and pf01's baseline

- Date: 2026-10-22
- Commit: this PR (#183; on `b25066c`): the `population` command (`population.py`).
- Machine: Apple M1 MacBook, 8 GB, macOS (Darwin 25.2), CPython 3.14.0. 2 threads, **shared** with three other agents, so the time is contended.
- Seeds: 0-499 against every member, `pf01` first on even seeds (the ladder's seeds and seats). Planners at 4 worlds and 20,000 states per plan, all in Rust; max 300 turns.

**The belief it tests**: the reason the design measures against a population rather than the last champion, that a chain of head-to-head wins can each be narrow and still not add up to playing better against everything. If it holds, `pf01`'s margins over the older members fall short of what the chain of promotions predicts, or it loses to one of them.

```sh
uv run python -m minimax_games.games.star_realms population pf01 --threads 2 --out docs/population/pf01
```

| Opponent | Wins | Win% | 95% interval | As 1st | Predicted by the chain |
|---|---|---|---|---|---|
| heuristic | 370/500 | **74.0%** | 70.0-77.7% | 196/250 | - |
| `mlp` | 308/500 | **61.6%** | 57.3-65.8% | 171/250 | 64.3% |
| `pi03` | 283/500 | **56.6%** | 52.2-60.9% | 150/250 | (its confirmation at promotion, fresh seeds: 54.0%) |
| **Mean** | | **64.1%** | 61.7-66.5% | | |

(Wilson intervals; the mean's is the normal one from the three rates' variances. No game was unfinished; mean 23.5-23.8 turns.) "Predicted by the chain" adds the Elo gaps of the promotions: `pi03` over `mlp` 58.0% (56 Elo) and `pf01` over `pi03` 56.6% here (46 Elo) give 64.3% for `pf01` over `mlp`.

Per-seed results (seed, first, second, winner, turns, the form of gating and ladder matches): [population/pf01/](population/pf01/), one CSV per opponent. A later candidate's run pairs with them seed by seed: `population <candidate> --paired docs/population/pf01`.

- **The command reproduces the recorded results**: its 500 games against the heuristic are `f32`'s ladder match (Milestone 8: hand-made features at pi03's data size, `features-window/ladder-f32.csv`) game for game, winners and turns, with `f32` named `pf01`. Pairing them with `pi03`'s ladder (`policy-iteration/run-1/ladder-r03.csv`) gives that section's 80 seeds won only by `pf01` and 61 only by `pi03` (z 1.60).
- **Time**: 14.7 minutes (879 s) on 2 threads, contended. On 8 free threads the same three matches took 71 s (the ladder) and 93 s (500 games against `pi03`, the confirmation) in Milestone 8, so the whole run is about 4 minutes there; a candidate cheaper than `pf01` takes less. With `--games 300` the run is 40% shorter and still pairs, on the seeds both played.

#### Reading it

- **`pf01` loses to none of the population**, and every interval is above 50%. The bar the design sets for promotion is met by today's champion, as it should be: the population is the chain of champions it beat.
- **Among the learned players the chain adds up**: 61.6% against `mlp` where the two promotions predict 64.3%, well inside the interval (57.3-65.8%). The champions so far are transitive, so far as 500 games can tell; no narrow win hid a weakness against an older champion.
- **Against the heuristic it doesn't**: `mlp` won 69.0% on these seeds, `pi03` 70.2% and `pf01` 74.0%. Two promotions worth about 100 Elo head to head moved the rate against the scripted bot by 5 points (paired with `pi03`, z 1.6). So the belief holds where the population differs in kind from the champions: progress measured on champions alone overstates progress against a different opponent. It is the case for keeping the heuristic, and hand-crafted opponents, in the population.
- **The yardstick for Milestone 10**: a candidate must beat `pf01` head to head and lose to none of the heuristic, `mlp` and `pi03`, with its rates and its paired seeds against `pf01`'s here. The mean, 64.1% (61.7-66.5%), summarises it; with 500 games each it moves by about ±2.4 points on noise alone, so pairing (seed by seed against these CSVs) is the sharper comparison.

### Milestone 9: the exploitation gap

- Date: 2026-10-22
- Commit: this PR (#184; on `b25066c`): self-play's `judge` (`selfplay::Record`), `gap.py` and the `gap` command.
- Machine: Apple M1 MacBook, 8 GB, macOS (Darwin 25.2), CPython 3.14.0. 2 self-play threads, with three other agents' builds and runs on the machine at the same time (load average 8-10).
- Seeds: 18,400,000-18,400,999, clear of every range a champion was fitted to and of every earlier experiment's (all below 10,000,000).

**Belief tested** (End state, What it learns from): *choosing the best of many noisy judgements favours the overestimated ones, so the errors that cost games are in the positions the player chooses between.* If so, the champion's judgement of the ends it chose exceeds how often it went on to win from them. A gap near zero says learning from rejected positions isn't needed.

```sh
uv run python -m minimax_games.games.star_realms gap --champion pf01 --pool mlp pi03 pf01 --games 1000 --base-seed 18400000 --threads 2
```

(`--pool`, `--base-seed` and `--games` are the defaults.) 1,000 games of `pf01`, mixed as a policy-iteration round mixes them: 500 against itself, 250 against the heuristic, 250 against a champion drawn per game from `mlp`, `pi03` and `pf01`; every player at the planner's full settings (4 worlds, 20,000 states per plan). **Chosen** are the ends `pf01` played (17,900 positions), each with the judgement it recorded as it played it: its value on the position its turn ended in. **Held out** are the ends its opponents chose (4,865 positions, from the 426 games not against itself), judged by `pf01`. The gap is the mean judged chance less the share of those positions whose player went on to win; ± is one standard error, counting a game's positions together.

| Positions | n | Games | Judged | Won | Gap |
|---|---|---|---|---|---|
| Chosen | 17,900 | 1,000 | 0.533 | 0.530 | **+0.002 ± 0.005** |
| Held out | 4,865 | 426 | 0.363 | 0.343 | +0.020 ± 0.018 |

Calibration, by judged chance:

| Judged | Chosen n | Judged | Won | Gap | Held-out n | Judged | Won | Gap |
|---|---|---|---|---|---|---|---|---|
| 0.0-0.1 | 1,366 | 0.047 | 0.040 | +0.007 ± 0.008 | 931 | 0.042 | 0.034 | +0.007 ± 0.010 |
| 0.1-0.2 | 1,188 | 0.152 | 0.160 | −0.008 ± 0.015 | 654 | 0.151 | 0.142 | +0.009 ± 0.025 |
| 0.2-0.3 | 1,320 | 0.252 | 0.238 | +0.014 ± 0.017 | 635 | 0.248 | 0.202 | +0.047 ± 0.026 |
| 0.3-0.4 | 1,685 | 0.352 | 0.334 | +0.018 ± 0.016 | 629 | 0.348 | 0.297 | +0.051 ± 0.031 |
| 0.4-0.5 | 2,293 | 0.453 | 0.450 | +0.003 ± 0.013 | 606 | 0.451 | 0.419 | +0.032 ± 0.034 |
| 0.5-0.6 | 2,514 | 0.550 | 0.563 | −0.013 ± 0.012 | 495 | 0.550 | 0.491 | +0.059 ± 0.038 |
| 0.6-0.7 | 2,321 | 0.647 | 0.653 | −0.006 ± 0.015 | 319 | 0.646 | 0.699 | −0.053 ± 0.041 |
| 0.7-0.8 | 1,725 | 0.748 | 0.734 | +0.013 ± 0.016 | 226 | 0.748 | 0.761 | −0.013 ± 0.042 |
| 0.8-0.9 | 1,695 | 0.850 | 0.842 | +0.008 ± 0.015 | 191 | 0.851 | 0.869 | −0.019 ± 0.043 |
| 0.9-1.0 | 1,793 | 0.953 | 0.955 | −0.002 ± 0.008 | 179 | 0.948 | 0.966 | −0.018 ± 0.017 |

By game phase (the game's turn, both players' turns counted):

| Turns | Chosen n | Judged | Won | Gap | Held-out n | Judged | Won | Gap |
|---|---|---|---|---|---|---|---|---|
| 1-6 | 4,722 | 0.533 | 0.540 | −0.007 ± 0.006 | 1,278 | 0.423 | 0.352 | +0.070 ± 0.021 |
| 7-12 | 4,722 | 0.544 | 0.540 | +0.004 ± 0.005 | 1,278 | 0.371 | 0.352 | +0.019 ± 0.019 |
| 13-18 | 4,600 | 0.538 | 0.536 | +0.003 ± 0.005 | 1,256 | 0.364 | 0.352 | +0.012 ± 0.017 |
| 19-24 | 2,895 | 0.511 | 0.506 | +0.005 ± 0.006 | 781 | 0.307 | 0.320 | −0.013 ± 0.019 |
| 25+ | 961 | 0.514 | 0.481 | **+0.033 ± 0.012** | 272 | 0.205 | 0.290 | −0.085 ± 0.043 |

12.6 minutes wall (23 CPU-minutes) on 2 threads with the machine shared, so about 3-4 minutes on 8 quiet threads; `--games 500` halves it, with an error of about ±0.007 on the chosen gap.

#### Reading it

- **pf01 has no exploitation gap to speak of.** Its judgement of the ends it chose is 0.533 against 0.530 won, a gap of +0.002 ± 0.005, and every bin of the calibration table is within 0.02 of its outcome (each within about 1.2 standard errors). Choosing the best of up to 20,000 judged ends per turn hasn't made it overrate what it chose. **The belief fails for today's player**, so learning from rejected positions isn't needed now (End state, What it learns from, updated in this PR). It is a diagnostic to re-run on each new champion, since a stronger or differently fitted judgement could change it.
- **Why the selection effect is small here**, a likely reading, not measured: the planner chooses among ends whose judgements mostly differ by what the turn bought and dealt, and most ends of a turn are plainly worse than the best few (How it explores). An error large enough to be selected has to beat that spread, and a value fitted to about 370,000 positions of games like these seldom makes one. The ends that are nearly tied are the ones a selection effect could bias, but a near tie also costs little when it goes wrong.
- **The one place a gap shows is long games**: from turn 25 on, the chosen gap is +0.033 ± 0.012 (961 positions, about 3 standard errors). Long games are rarer in the training data, which is the likeliest reason (not tested here). It is small, 5% of the positions, and worth watching on the next champion rather than acting on.
- **Held out is not a clean control.** The opponents' ends are calibrated about as well overall (+0.020 ± 0.018), but in a different shape: overrated early (+0.070 ± 0.021 in turns 1-6) and in the middle bins of judged chance, underrated late. The judgement scores a position as if it were played on by the kind of player its training games had, and these positions are played on by the heuristic and older champions against pf01, who are weaker than that: an opponent's position early in the game is worth less to them than to pf01. So the held-out gap mixes plain miscalibration with who is playing. That the chosen positions show no gap at all, while the same judgement misreads others' positions by up to 0.07, says the selection effect isn't hiding beneath a miscalibration it cancels.
- **For #185**: the chosen judgements, with their outcomes and turns, are what the other diagnostics break down further (`gap.positions`, from `selfplay.play(..., judge=True)`).

### Milestone 9: probe, same-function synergy

- Date: 2026-10-22
- Commit: this PR (#189; on `b25066c`): `probe` (probe positions, the synergy probe and its report) and the `probe synergy` command.
- Machine: Apple M1 MacBook, 8 GB, macOS (Darwin 25.2), CPython 3.14.0, numpy 2.5.4. Games on 2 threads, with three other agents' builds and runs on the machine at the same time.
- Seeds: the champion's games 12,000,000-12,000,199 (`probe.BASE_SEED`), clear of every earlier range (all below 10,000,000); the cards' draws seed 0.

**The belief**: *cards of the same function are worth more together* (star-realms/strategy.md, Synergy, [lore]): a second and third card that draws, scraps or makes the opponent discard add more than the first. The End state leaves such ideas for the judgement to learn (What it sees), and this probe asks whether `pf01`'s has.

```sh
uv run python -m minimax_games.games.star_realms probe synergy --model pf01 --games 200 --threads 2
```

- **Positions**: every turn's end in 200 games of `pf01`'s planner against itself (4 worlds, 20,000 states per plan, in Rust): 4,548 positions.
- **The probe**: in each position, 3 cards of a function are added one at a time to the deck of the player who just ended, each drawn at random from the function's Trade Deck cards with a copy out of sight (neither player holds it nor the Trade Row shows it, so the changed position is still a valid one), and `pf01` judges each step. Four draws per position, averaged. The functions are as `features.MORE` counts them (any of a card's abilities; a card can have more than one): of the base set's distinct Trade Deck cards, 21 draw, 8 scrap and 5 make the opponent discard. The **control** is the 18 that do none of these: any card's worth is diluted by the cards beside it, so each further card is expected to add less, and only a function's gains against the control's say anything about that function.
- **Scales**: gains in judged chance, and in logit (log(p / (1 − p))). A judgement whose score is linear before its sigmoid gains less in chance from each card as the chance nears 1, which would read as a lack of synergy where it has none; the logit scale takes that out. Both are tested on constructed judgements with known answers (a linear one shows no synergy, a quadratic one exactly its synergy, a sigmoid of a linear one none on the logit scale).

What each card added, over all 4,548 positions (mean, and standard deviation across positions), and what the second and third added beyond the card before (mean and standard error, paired by position):

| Function | 1st | 2nd | 3rd | 2nd − 1st | 3rd − 2nd | 2nd − 1st, logit | 3rd − 2nd, logit | Positions where 2nd > 1st |
|---|---|---|---|---|---|---|---|---|
| draw | +0.126 (0.081) | +0.096 (0.064) | +0.069 (0.052) | −0.0301 (0.0011) | −0.0273 (0.0008) | −0.061 (0.005) | −0.059 (0.005) | 32% |
| scrap | +0.091 (0.080) | +0.071 (0.063) | +0.048 (0.046) | −0.0199 (0.0008) | −0.0228 (0.0007) | −0.041 (0.004) | −0.079 (0.004) | 37% |
| discard | +0.065 (0.050) | +0.065 (0.049) | +0.058 (0.043) | −0.0004 (0.0007) | −0.0069 (0.0007) | **+0.017 (0.004)** | −0.005 (0.004) | 48% |
| none (control) | +0.069 (0.052) | +0.059 (0.046) | +0.049 (0.040) | −0.0102 (0.0007) | −0.0103 (0.0006) | −0.028 (0.004) | −0.029 (0.003) | 42% |

The second card against the first on the logit scale, by phase (turns 1-8: 1,400 positions; 9-16: 1,600; 17 on: 1,548):

| Function | Early | Middle | Late |
|---|---|---|---|
| draw | −0.057 (0.011) | −0.070 (0.009) | −0.054 (0.007) |
| scrap | −0.082 (0.008) | −0.038 (0.007) | −0.008 (0.005) |
| discard | **+0.055 (0.007)** | +0.013 (0.006) | −0.014 (0.006) |
| none (control) | −0.028 (0.006) | −0.027 (0.006) | −0.028 (0.005) |

2.7 minutes in all (163 s wall): 162 s for the games, 1.1 s for the probe itself. An earlier run on seeds 6,000,000-6,000,199 (before the range was moved clear of #175's) gave the same picture to within the standard errors.

#### Reading it

- **`pf01` hasn't learned same-function synergy for cards that draw or scrap.** Each further draw or scrap card adds less than the one before, and falls off faster than the control: on the logit scale the second draw card adds 0.061 less than the first against the control's 0.028, the second scrap card 0.041 less, and the third scrap card 0.079 less than the second. Only about a third of positions judge a second draw or scrap card worth more than the first. In chances the shortfall is larger still (3.0 and 2.0 points against the control's 1.0), part of it the saturation the logit scale removes. The scrap shortfall is steepest early (−0.082) and gone late, as a deck with little left to scrap would suggest, but nowhere does it turn positive.
- **Discard is the one function whose second card adds more than its first**, by 0.017 (0.004) in logit over all positions and 0.055 (0.007) early, against the control's −0.028: a small synergy, early, fading to below zero late. In chances it is flat (−0.0004).
- **What the probe can't separate**: the added cards are drawn at random across factions, and `pf01` reads each faction's density and expected pairs in a hand (`features.EXTRA`), so some of what a second card adds is faction synergy; random draws average it over, and the control carries the same mix. The row can't tell the Trade Deck from the scrap heap, so a copy that was scrapped can count as one left to add. And the probe measures the judgement, not the game: it says what `pf01` believes, not whether the lore is true, so the strategy doc's label stays [lore].
- **What it points to** (How its understanding is judged): same-function synergy for draw and scrap is an idea the judgement hasn't built for itself, so describing it directly is a candidate experiment for Milestone 10. #178's per-player densities of draw, scrap and discard cards added nothing measurable, but a density is linear in the count; the probe's question is about pairs, as `features.EXTRA`'s expected same-faction pairs in a hand was for factions.

### Milestone 9: probe, faction pairs

- Date: 2026-10-22
- Commit: this PR (#190; on `72534cc`): `probe.faction_pairs`, `mostly`, `pairs` and their report, and `probe faction-pair`.
- Machine: Apple M1 MacBook, 8 GB, macOS (Darwin 25.2), CPython 3.14.0. Games on 2 threads, with three other agents' builds and runs on the machine at the same time.
- Seeds: the same 200 games as the synergy probe, 12,000,000-12,000,199 (`probe.BASE_SEED`), so the same 4,548 positions; the cards' draws seed 0.

**The belief**: *some factions pair particularly well* (star-realms/strategy.md, Synergy, [lore]): beyond a faction's own ally abilities, a card of the right second faction adds more to a deck of the first than other factions' cards do. The probe asks whether `pf01`'s judgement singles out any pair.

```sh
uv run python -m minimax_games.games.star_realms probe faction-pair --model pf01 --games 200 --threads 2
```

- **The probe**: in each position, one card of each faction is added to the deck of the player who just ended, drawn at random from that faction's Trade Deck cards with a copy out of sight (as the synergy probe adds them), and `pf01` judges the position before and after. Four draws per position, averaged.
- **Mostly one faction**: the deck (draw pool and bases in play) holds **at least 4 faction cards and one faction has a strict majority of them** (Scouts, Vipers and Explorers belong to none). Below 4, one purchase decides the majority, and a tie isn't "mostly" anything. Of the 4,548 positions, 1,112 qualify, close to evenly: **Blob 276, Trade Federation 295, Star Empire 271, Machine Cult 270**; the other 3,436 (early decks and two-faction decks) are left out. A threshold of 6 cards keeps 167-186 per faction and gives the same picture.
- **Reading a pair**: a cell's gain mixes what any card adds to such a deck (its row: a weaker deck gains more from anything), what such a card adds to any deck (its column) and the pair's own part. The **interaction** takes the first two out: the cell less its row's and column's means, plus the table's. Its standard error adds the rows' variances (disjoint positions); it counts positions of one game as independent, so it understates the error somewhat. Tested on constructed judgements: one linear in each faction's cards has no interaction, one with a Blob × Trade Federation product singles out exactly those two pairs.

Judged gain on the logit scale of one card of each faction (columns) in decks mostly of each (rows): mean (standard deviation across positions), standard error about 0.012-0.020:

| Deck mostly of | Positions | + Blob | + Trade Federation | + Star Empire | + Machine Cult |
|---|---|---|---|---|---|
| Blob | 276 | **+0.678** (0.323) | +0.325 (0.299) | +0.365 (0.227) | +0.277 (0.250) |
| Trade Federation | 295 | +0.383 (0.273) | **+0.583** (0.328) | +0.375 (0.205) | +0.401 (0.226) |
| Star Empire | 271 | +0.489 (0.280) | +0.449 (0.309) | **+0.638** (0.236) | +0.373 (0.231) |
| Machine Cult | 270 | +0.402 (0.255) | +0.497 (0.331) | +0.412 (0.199) | **+0.455** (0.264) |

Each pair's interaction (logit, ± standard error):

| Deck mostly of | + Blob | + Trade Federation | + Star Empire | + Machine Cult |
|---|---|---|---|---|
| Blob | **+0.223 ± 0.011** | −0.106 ± 0.011 | −0.050 ± 0.010 | −0.067 ± 0.009 |
| Trade Federation | −0.097 ± 0.011 | **+0.128 ± 0.012** | −0.064 ± 0.009 | **+0.033 ± 0.009** |
| Star Empire | −0.042 ± 0.012 | −0.058 ± 0.012 | **+0.147 ± 0.009** | −0.047 ± 0.009 |
| Machine Cult | −0.084 ± 0.011 | +0.036 ± 0.012 | −0.034 ± 0.009 | **+0.081 ± 0.009** |

In chances the tables have the same shape (a card adds 0.046-0.112; diagonal interactions +0.015 to +0.035, Machine Cult into Trade Federation +0.005 ± 0.002). About 2 minutes in all (115 s for the games, 0.4 s for the probe; a first run took 120 s).

#### Reading it

- **The judgement's strongest pairing is a faction with itself**: in every faction a card adds most to a deck mostly of its own faction, by 0.08 (Machine Cult) to 0.22 (Blob) in logit beyond its row and column, 9-20 standard errors. That is ally abilities, which `pf01` sees directly through `features.EXTRA`'s expected same-faction pairs in a hand, so it is the expected result and a check that the probe reads what the features say.
- **Of the twelve pairs of different factions, it singles out one, slightly**: a Machine Cult card in a deck mostly of Trade Federation, +0.033 ± 0.009, which holds with a second seed of draws (+0.039) and with the 6-card threshold (+0.030). The reverse, a Trade Federation card in a Machine Cult deck, reads +0.036 here but +0.010 with the second draw seed, so it isn't settled. Every other cross-faction pair is below zero, which is partly arithmetic: each row and column of interactions sums to zero, so a large diagonal pushes the rest down, and among them only the Trade Federation and Machine Cult pair stands out from the rest.
- **On the belief**: `pf01`'s judgement values a faction's own cards far above any pairing, and beyond that values one cross-faction pair a little (Machine Cult cards with Trade Federation decks, under half the smallest same-faction interaction and a seventh of the largest). Whether that pair is in fact a good one, or other pairs are better than the judgement thinks, the probe can't say: it reports what the judgement learned, not the game, so the strategy doc's label stays [lore].
- **Row effects** are visible too, though they aren't the question: Star Empire decks gain the most from any card (mean +0.49) and Machine Cult decks' own cards add least to them (+0.455, against 0.58-0.68 for the other factions); why is not tested here.

### Milestone 9: probe, denial

- Date: 2026-10-22
- Commit: this PR (#192; on `72534cc`): the denial probe (`probe.denial`), Trade Row choices from recorded games and their judged worth (`probe.choices`, `probe.worth`, `probe.denial_shares`), and the `probe denial` command.
- Machine: Apple M1 MacBook, 8 GB, macOS (Darwin 25.2), CPython 3.14.0. Games on 2 threads, with three other agents' builds and runs on the machine at the same time.
- Seeds: the champion's games, and the heuristic's against itself, on 19,200,000-19,200,199, clear of every earlier range. The probe draws nothing at random.

**The belief**: *keeping a card from the opponent has value, even when you don't want it* (star-realms/strategy.md, Denial, [lore]). The probe asks whether `pf01`'s judgement holds it: whether a position is judged worse for the player who just ended when the Trade Row shows a card the next player would value, and whether the champion buys or scraps cards for denial.

```sh
uv run python -m minimax_games.games.star_realms probe denial --model pf01 --games 200 --base-seed 19200000 --threads 2
```

- **Positions**: every turn's end in 200 games of `pf01`'s planner against itself (4 worlds, 20,000 states per plan, in Rust, every decision recorded): 4,415 positions.
- **A card the next player would value** is the judgement's own answer, since the probe asks what the judgement believes: of the Trade Deck cards with a copy out of sight, the one whose copy added to the next player's draw pool lowers the judged chance of the player who just ended most (the *most valued*), and as the contrast the one that lowers it least (the *least valued*). The extremes give the largest contrast to read a small effect against, and each position's own pair (rather than one fixed list of good cards) keeps what the opponent values in that position, with their deck and the turn.
- **The effect**: each of the Trade Row's distinct cards in turn is swapped for the most valued card and, separately, for the least valued one (the swapped-out copy goes out of sight, the new one comes from out of sight, so the row still shows five cards and every card is in one place; tested), and the effect is the judged chance with the most valued card in the row less that with the least valued, averaged over the row's cards. *Valued* is how much more the next player owning the most valued card costs than owning the least valued; **seen** = −effect / valued is the share of that the judgement sees in the card being in the Trade Row for them: 0 if the row means nothing for the next buyer, 1 if a card in the row is as good as theirs. A card in the row can also be bought later by the player who just ended, so the effect is the net of both; denial shows as an effect below 0.
- **Denial in games**: for each Trade Row card a player acquired (bought, or taken free by Blob Carrier) or scrapped from the Trade Row, judged in the position its turn ended in, without the card (for a card acquired, its copy taken from the chooser's draw pool and put out of sight; for a card scrapped, the position itself): **own** is what the chooser owning it adds to their judged chance, **kept** what the opponent owning it would take away. A choice is denial when own < kept. Kept is an upper bound on what denial saves, since the opponent might not have bought the card; Explorers aren't Trade Row cards and are left out. The heuristic's buys and scraps are from its own games against itself on the same seeds, judged by `pf01`; its Trade Row scraps follow a scripted denial rule (the card worth most to the opponent, if worth more than the average refill), its buys none.

The effect of the most valued card in the Trade Row against the least valued (mean, standard deviation across positions, standard error), by phase:

| Phase | Positions | Effect (chance) | sd | se | Valued | Seen | Below 0 | Effect (logit) | se | Seen (logit) |
|---|---|---|---|---|---|---|---|---|---|---|
| All | 4,415 | −0.0074 | 0.0186 | 0.0003 | 0.334 | **2.2%** | 67% | −0.038 | 0.002 | 1.9% |
| Early (turns 1-8) | 1,400 | −0.0076 | 0.0140 | 0.0004 | 0.420 | 1.8% | 75% | −0.034 | 0.002 | 1.5% |
| Middle (9-16) | 1,599 | −0.0105 | 0.0233 | 0.0006 | 0.350 | 3.0% | 69% | −0.055 | 0.003 | 2.6% |
| Late (17 on) | 1,416 | −0.0038 | 0.0158 | 0.0004 | 0.230 | 1.7% | 59% | −0.023 | 0.003 | 1.3% |

The share of Trade Row choices that are denial, judged by `pf01` (with their mean own and kept):

| Player | Kind | All | Early | Middle | Late | Own | Kept |
|---|---|---|---|---|---|---|---|
| `pf01` | buys (4,494) | **38.6%** | 36.8% (1,555) | 38.0% (1,667) | 41.5% (1,272) | +0.087 | +0.076 |
| heuristic | buys (5,059) | 44.1% | 41.2% (1,328) | 41.5% (1,902) | 49.0% (1,829) | +0.069 | +0.063 |
| `pf01` | Trade Row scraps (302) | **75.8%** | 87.0% (23) | 81.5% (124) | 69.7% (155) | +0.060 | +0.107 |
| heuristic | Trade Row scraps (296) | 56.1% | 33.3% (3) | 57.4% (115) | 55.6% (178) | +0.134 | +0.155 |

2-2.5 minutes in all: 114 s and 144 s for both sets of games in two identical runs (the heuristic's take a few seconds), about 1 s for the probe itself.

#### Reading it

- **`pf01` sees a little denial in the Trade Row, about 2% of the card's worth.** The card the next player values most in the row lowers the judged chance of the player who just ended by 0.0074 against the least valued (se 0.0003, two positions in three below 0), where the next player owning it would cost 0.33 more: the judgement reads the row as bearing on the next buyer, but by a fiftieth of what the card is worth to them. The effect is largest in the middle game (3.0%) and smallest late, the same on the logit scale. Whether 2% is too little the probe can't say (a card in the row is bought by the next player only sometimes, and may go to the player who ended instead), but nothing in the row tells the judgement what the next player can afford, so it can only learn an average.
- **Its Trade Row scraps are mostly denial, more than the heuristic's scripted ones**: 75.8% of its 302 scraps took a card worth less in its own deck than kept from the opponent, against 56.1% of the heuristic's. A scrap moves the judged chance only through the Trade Row, and the 2% the row shows is enough to point the planner's scrap at the card the opponent values. The comparison is by `pf01`'s own judgement, which its choices maximise, so it favours `pf01`: it says the champion's scraps agree with its judgement of the opponent, not that they deny better in play.
- **Its buys don't deny**: 38.6% meet the definition against the heuristic's 44.1%, and the heuristic doesn't consider denial when it buys. Most cards are worth about as much to either player, so about two buys in five clear the bar by chance; a player that bought to deny would show more, not fewer. The champion buys for its own deck.
- **On the belief**: `pf01` holds a weak form of it: the row matters a little for the next buyer, enough to steer its scraps, not its buys. Describing the Trade Row to the judgement as it bears on the next buyer (what they can afford, how much each card is worth to them) is Milestone 10's experiment, and this probe is its diagnostic: the share seen and the buys' denial share should rise if the idea was missing. The strategy doc's label stays [lore]: the probe measures what the judgement believes, not whether denial wins games.

### Milestone 9: the judgement's errors by reshuffle distance and rare cards, and turns not examined

- Date: 2026-10-22
- Commit: this PR (#185; on `72534cc`): self-play's `diagnose` (`selfplay::Record::diagnostics`: each turn's end's cards left before each player's reshuffle, each plan's turn and completeness), `gap.measure(..., every=)`, `diagnostics.py` and the `diagnose` command.
- Machine: Apple M1 MacBook, 8 GB, macOS (Darwin 25.2), CPython 3.14.0, numpy 2.5.4. 2 self-play threads, **with three other agents working on the machine at the same time**, so the time is contended.
- Seeds: the exploitation gap's games (18,400,000-18,400,999, mixed as `iterate.mix` mixes them), every third of them: no new seeds.

**Beliefs tested** (End state, What it sees, How it chooses):

- (a) *When a card will be drawn matters as much as owning it* (the deck's cycle). The judgement sees each player's draw pool as one set, so if the belief holds, its error is larger near a reshuffle than far from one.
- (b) *What is learned about common cards should carry to rare ones* (cards by what they do). The judgement knows cards only by name, so if it doesn't carry, its error is larger on positions holding a card the Trade Deck has one copy of.
- (c) *Almost every turn's outcomes can be examined in full* (within the turn, tactics are calculated). Its test: the share of the champion's turns that weren't.

```sh
uv run python -m minimax_games.games.star_realms diagnose --champion pf01 --pool mlp pi03 pf01 --games 1000 --every 3 --base-seed 18400000 --threads 2
```

(Every option but `--threads` is the default.) 334 of the exploitation gap's 1,000 games, the 1st, 4th, 7th, ... of its mix (the same games, in the same proportions: 167 against itself, 83 against the heuristic, 84 against the pool), every player at 4 worlds and 20,000 states per plan. **Chosen** are the 6,077 ends `pf01` played, with its judgement as it played them; **held out** the 1,616 ends its opponents played (142 games), judged by `pf01`. Each bin gives the gap (mean judged chance less share won) and the squared miss (mean of (judged − won)², the Brier score), each ± one standard error counted by game. Near a reshuffle is 0-4 cards left in the deck: the player's next hand draw reshuffles their discard pile.

(a) By the cards left in the deck of the player who just ended:

| Left | Chosen n | Games | Gap | Squared | Held-out n | Gap | Squared |
|---|---|---|---|---|---|---|---|
| 0-4 (near) | 2,657 | 334 | −0.001 ± 0.010 | 0.188 ± 0.007 | 675 | +0.063 ± 0.031 | 0.163 ± 0.012 |
| 5-9 | 2,214 | 334 | −0.012 ± 0.009 | 0.181 ± 0.008 | 569 | +0.048 ± 0.031 | 0.155 ± 0.013 |
| 10-14 | 1,073 | 303 | +0.002 ± 0.012 | 0.170 ± 0.011 | 298 | +0.026 ± 0.033 | 0.133 ± 0.016 |
| 15+ | 133 | 88 | +0.007 ± 0.039 | 0.122 ± 0.022 | 74 | −0.008 ± 0.049 | 0.102 ± 0.034 |
| 5+ (far) | 3,420 | 334 | −0.007 ± 0.008 | 0.175 ± 0.009 | 941 | +0.037 ± 0.030 | 0.144 ± 0.014 |

By the next player's: near 2,529 chosen, gap −0.008 ± 0.009, squared 0.185 ± 0.007; far 3,548, −0.002 ± 0.009, 0.177 ± 0.008 (held out: near 690, +0.043 ± 0.035, 0.166 ± 0.015; far 926, +0.052 ± 0.029, 0.141 ± 0.012).

(b) By a rare card held by either player (23 cards: one copy each in the Trade Deck):

| | Chosen n | Games | Gap | Squared | Held-out n | Gap | Squared |
|---|---|---|---|---|---|---|---|
| With | 4,718 | 334 | +0.001 ± 0.008 | 0.167 ± 0.009 | 1,239 | +0.028 ± 0.030 | 0.137 ± 0.014 |
| Without | 1,359 | 331 | −0.022 ± 0.012 | 0.228 ± 0.008 | 377 | +0.114 ± 0.039 | 0.201 ± 0.013 |

(a) and (b) within each game phase, chosen positions (n, squared miss):

| Turns | Near | Far | With a rare card | Without |
|---|---|---|---|---|
| 1-6 | 1,045, 0.227 | 533, 0.226 | 380, 0.211 | 1,198, 0.232 |
| 7-12 | 507, 0.193 | 1,071, 0.190 | 1,427, 0.190 | 151, 0.203 |
| 13-18 | 608, 0.143 | 945, 0.163 | 1,543, 0.155 | 10, 0.117 |
| 19-24 | 343, 0.141 | 652, 0.128 | 995, 0.133 | 0, - |
| 25+ | 154, 0.190 | 219, 0.166 | 373, 0.176 | 0, - |

(c) The champion's turns not examined in full (some plan in the turn ran out of its 20,000 states before listing every end of the turn in every world):

| Turns | The champion's turns | Not in full | Share |
|---|---|---|---|
| 1-6 | 1,578 | 0 | 0.0% |
| 7-12 | 1,578 | 62 | 3.9% |
| 13-18 | 1,567 | 338 | 21.6% |
| 19-24 | 1,151 | 463 | 40.2% |
| 25+ | 492 | 280 | 56.9% |
| All | 6,366 | 1,143 | **18.0%** |

3.0 minutes wall (177 s, 5.9 CPU-minutes) on 2 contended threads; the full 1,000 games (`--every 1`) take three times that, about the gap's 12.6 minutes. Two runs gave identical output.

#### Reading it

- **(a) No sign that the deck's cycle limits the judgement.** The gap is near zero near a reshuffle and far from one, for either player (−0.001 against −0.007 for the player who ended). The squared miss is a little higher near (0.188 against 0.175), but within each phase there is no consistent direction: equal in turns 1-12, lower near in 13-18, higher near in 19 on, each difference about 0.01-0.02 on a few hundred positions, about the size of one standard error. Overall, near positions are a little more common early, where every position is hard to call. So for `pf01` **the belief isn't supported by this diagnostic**, and splitting the judgement's description of a deck by zone (Milestone 10) isn't called for by it. The test of the split itself, with and without, remains the design's; this says only that the error it would remove isn't visible.
- **(b) Positions with a rare card are judged no worse.** Without the phase they look better judged (0.167 against 0.228), but that is the phase: 88% of the positions without a rare card are in turns 1-6, before anyone has bought one. Within turns 1-6 and 7-12, the only phases with both, positions holding a rare card are judged a little *better* (0.211 against 0.232; 0.190 against 0.203). So **what is learned about common cards isn't visibly failing to carry to rare ones**, and describing cards by what they do isn't called for by this diagnostic. Its limit: "holds any of 23 cards" is broad, and from turn 7 on almost every position does, so a few badly judged rare cards could hide in it.
- **(c) Turns not examined in full are common, and grow through the game**: 18% of the champion's turns overall, none in turns 1-6, 4% in 7-12, then 22%, 40% and 57%. The design's premise that *almost every* turn can be examined in full holds early but not late, when decks are bigger and have more draw and more choices. That is a fact about today's budget (5,000 states per world), not a cost in strength: whether examining those turns in full changes the choice, or wins more, is the examination test (How it chooses), which this makes worth running, on those turns. It may also bear on the one exploitation gap found (+0.033 in turns 25+), where most turns are cut short, though nothing here connects the two.
- **Held out** shows the same pattern as in the gap's measurement (opponents' positions overrated, more so without a rare card, which is the early game again), so it adds nothing to (a) or (b) beyond who is playing.

## Milestone 7: Rust tracer bullet (toolchain and first-slice engine)

2026-10-21, #137 (tracked by #127), CPython 3.14, Apple M1 (8 GB) shared with three other agents' builds and tests. The first end-to-end path through Rust: maturin builds `minimax_games._native` from `rust/`, and its Star Realms engine plays the starter decks and the 30-card first slice (design, Compiled engine).

**Speed**, `task native-bench -- --games 5000` (seeds 0-4999, greedy vs greedy, release build, three runs):

| Measure | Result |
|---|---|
| Greedy games | 5,210-5,270 games/s (0.19 ms per game) |
| Decisions in those games | 582,000-588,000/s (112 per game) |
| `advance` alone, over the 558,237 recorded decisions | 819,000-825,000 calls/s (1.2 µs per call) |

The games figure includes legal actions and the greedy bot's choice; the `advance` figure replays each recorded (state, action) pair through `advance_legal` (clone the state, apply, auto-play), as a search would. The engine is unoptimised: piles are `Vec`s cloned whole on every `advance`, as the Python engine copies tuples. These are absolute numbers on the first slice only, with no CPython comparison (design, Compiled engine).

**Differential tests**, `tests/star_realms/test_native.py`: 200 greedy-vs-greedy and 100 seeded random games (up to 300 turns, 8 of them reach it), comparing the whole state and the list of legal actions with the Python engine after every action, plus the Rust greedy bot's choice at every decision and its whole-game result. Random play covers every first-slice action (scrapping ships and bases, attacking bases, and the Trade Deck running out). They take about 3.5 s.

**Bugs differential testing caught: none.** The engine matched the Python one on the first run of the differential tests. Two deliberately injected bugs (refilling the Trade Row at its end rather than in place; not freeing a scrapped base's used primary) each failed the tests on the first affected seed, so the harness does catch divergence. The likely reasons the port went cleanly: the slice is small, the Python rules are already written as an explicit, documented reference, and the RNG port was checked on its own first (`cargo test` against values from CPython).

**Effort.** About an hour of agent time end to end, roughly a third on the engine (RNG, cards, rules, greedy bot) and the rest on the toolchain (maturin, pre-commit, CI, release), the differential tests and these docs. The extension builds in about 15 s from clean on the M1 (two build jobs) and in about 1 s incrementally; `uv run pre-commit run --all-files` takes about 20 s.

**Surprises.**

- `rust-toolchain.toml` pinning `1.95.0` made rustup download that toolchain although the installed `stable` was the same 1.95.0: rustup treats the names as different toolchains. A one-time download.
- `task pypy` no longer installs: maturin builds a PyPy-specific wheel (`pp312`), which uv refuses under the task's `--python-version 3.13` workaround for PyPy having no 3.13. The compiled engine is the path to speed now, so the task is left broken pending a decision (drop it, or install PyPy's copy another way).

## Milestone 7: Rust engine, full deck

2026-10-21, #138 (tracked by #127), CPython 3.14, Apple M1 (8 GB) shared with another agent's builds and tests. The Rust engine now plays the whole 80-card base set: every kind of effect and action, Stealth Needle's copies included, plus `observe` and `sample_world` (design, Compiled engine, "The full deck").

**Speed**, `task native-bench -- --games 5000` (seeds 0-4999, greedy vs greedy, full deck, release build, three runs):

| Measure | Result |
|---|---|
| Greedy games | 3,250-3,290 games/s (0.31 ms per game) |
| Decisions in those games | 493,000-500,000/s (152 per game) |
| `advance` alone, over the 759,480 recorded decisions | 627,000-735,000 calls/s (1.4-1.6 µs per call) |

Against the first slice (Milestone 7, tracer bullet), a game is 1.6 times slower, mostly because full-deck games are longer (152 decisions against 112), and `advance` 10-25% slower per call, from the checks automatic play now makes (scrappers from hand, Fleet HQ, Embassy Yacht) and the larger state cloned on every call. Still unoptimised: piles are `Vec`s cloned whole, and automatic play recomputes the playable actions after every step.

**Differential tests**, `tests/star_realms/test_native.py`: 100 greedy-vs-greedy and 60 seeded random games, comparing after every action the whole state, the legal actions, the winner, both views and a world sampled from the current player's view (bit-identical to Python's from the same seed), plus the greedy bot's choices and results. The random games take every kind of action (the rarest, `AcquireFreeShip`, 6 times), and the test fails if one goes missing. About 6 s. Once, outside the suite: 1,000 more random games (seeds 1000-1999, up to 400 turns) and 500 greedy ones, about 367,000 actions, all identical, in 70 s.

**Bugs differential testing caught: none in the rules.** As with the first slice, the engine matched the Python one on the first run of the differential tests. The only failure on the way was in a new Rust unit test, which set a pending discard after automatic play had already emptied the hand. Two deliberately injected bugs each failed the tests at a named seed and action: not capping pending discards at the hand's size (seed 58, action 659, an `EndTurn` on turn 171: a long random game, so the wider random games matter), and forgetting that Stealth Needle in hand keeps ships waiting (seed 11, action 68). The reasons are the tracer bullet's: the Python rules are an explicit, documented reference, small functions with one job each, so the port is function for function.

**`sample_world` is bit-identical** to Python's given the same seed: it makes the same draws from the same CPython RNG port, in the same order, so the tests compare sampled worlds exactly rather than by distribution.

**Effort.** About 20 minutes of agent time, end to end: reading the Python engine and the crate, writing the cards (the longest part, by volume), the rules, views, greedy bot and boundary, the differential tests and these docs. The extension rebuilds in about 10 s; `uv run pre-commit run --all-files` takes about 30 s.

**Surprises.**

- Stealth Needle's copies were the one design question: a Python copy is a new `Card` made at run time. In Rust it is a card variant naming its target, `NeedleAs(i)`, whose data is built once on first use, which keeps cards two bytes and `Copy`, so piles and actions needed no change.
- The boundary's actions had to grow: `Choose`, `Recycle` and `ScrapAndDraw` carry more than one card name, so actions now cross as `(kind, *fields)` tuples, not `(kind, card)` pairs.

## Milestone 7: Rust heuristic bot

2026-10-21, #139 (tracked by #127), CPython 3.14, Apple M1 (8 GB) shared with other agents' builds and tests. The Rust engine gains the heuristic bot, `bots.HeuristicBot` ported to `heuristic.rs`, choosing from views (design, Compiled engine, "The heuristic bot").

**Speed**, `task native-bench -- --games 5000` (seeds 0-4999, heuristic vs heuristic, full deck, release build, three runs):

| Measure | Result |
|---|---|
| Heuristic games | 1,270-1,300 games/s (0.78 ms per game) |
| Decisions in those games | 163,000-167,000/s (128 per game) |
| Greedy games, same runs | 3,120-3,300 games/s (152 decisions per game) |

A heuristic decision costs about 6 µs against greedy's 2 µs, mostly from building the view before every decision (`observe` clones every pile and sorts the hidden ones), which greedy doesn't need. Unoptimised, as the engine is; at this speed the heuristic is no bottleneck as a self-play opponent beside the planner's milliseconds per decision.

**Differential tests**, `tests/star_realms/test_native.py`: 60 heuristic-vs-heuristic, 60 heuristic-vs-greedy and 60 heuristic-vs-random games (seeds 0-59, seats alternating), comparing the Rust heuristic's choice with the Python one's at every decision of every player (19,613 decisions), plus `play_heuristic`'s winner and turns. The heuristic chooses every kind of action across them (the rarest, `Recycle`, twice), and the test fails if one goes missing. About 2 s. Once, outside the suite: the same three pairings on seeds 1000-1999 (random games up to 400 turns), 3,000 games and 333,084 decisions, all identical, in 24 s.

**Bugs differential testing caught: none.** The port matched the Python bot on the first run. Injected bugs: drawing with Blob World at 2 Blob cards instead of 3 failed at seed 39, action 107, turn 24. Three others passed, each for a reason worth knowing: breaking ties to the last least key rather than the first (every key ends in a name or an option index, so equal keys are the same action); scrapping from the Trade Row when its card is worth exactly the refill's average (never exactly equal in these games); and an off-by-one in the opponent's turn for that scrap (it differs only on turn 4, when no Trade Row scrap came up). So the tests pin down what changes choices in practice, not every line.

**Floats.** The heuristic's scores are floats, but every term is a multiple of 0.5, so all its sums are exact in any order and the one rounded operation, the Trade Deck's average worth, is the same IEEE division as Python's; the port still adds in Python's order (design, "The heuristic bot").

**Effort.** About 25 minutes of agent time end to end: reading the Python bot and the crate, the test, the port (about 400 lines of Rust with its unit tests), the benchmark and these docs. The port went cleanly for the engine's reasons: the Python bot's docstring and helpers already state each rule as one small function.

## Milestone 7: Rust planner enumeration and feature encoder

2026-10-21, #140 (tracked by #127), CPython 3.14, Apple M1 (8 GB) shared with other agents' builds and tests. The whole-turn planner's enumeration and the feature encoder are ported to Rust (`planner.rs`, `features.rs`), and the Python planner can use them as a backend, scoring the plan's feature matrix in one numpy call (design, Compiled engine, "The planner's enumeration").

**Time per plan**, at every turn start of the heuristic-vs-heuristic games of seeds 0-19 (the turn-size measurement's 483 turns, Milestone 6), 4 worlds sampled from the mover's view, the default 20,000 states, the logistic value. "Plan" is the whole `planner.plan(..., backend="rust")` call (state dicts in, scoring and choice in Python); "enumeration" is `plan_ends` alone, encoding included. 421 of the 483 plans finished.

| | q1 | median | q3 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| States visited | 136 | 566 | 3,159 | 15,546 | 20,000 | 20,000 | |
| Plan, Rust backend (ms) | 0.54 | 1.95 | 15.0 | 71.2 | 127 | 161 | 16.9 |
| Enumeration alone (ms) | 0.39 | 1.62 | 13.9 | 68.3 | 120 | 151 | 16.0 |

By plan size:

| States visited | Plans | Plan, median | Enumeration, per state |
|---|---|---|---|
| under 1,000 | 288 | 0.64 ms | 3.2 µs |
| 1,000-5,000 | 93 | 7.8 ms | 4.2 µs |
| 5,000-20,000 | 102 | 67 ms | 4.7 µs |

Against the Python backend, on the 123 turn starts of seeds 0-4 (same worlds, both backends, every plan checked equal): **14 times faster** in all, 2.2 s against 30.7 s; the median plan 1.8 ms against 21.5 ms, the 90th percentile 77 ms against 1.14 s. The Python enumeration costs 44-71 µs per state (more on bigger turns), the Rust one 3-5 µs. Fixed costs per plan (converting the worlds to dicts, the numpy call, replaying the chosen line) are about 0.2-0.3 ms.

**Where the Rust time goes.** Every state listed advances each of its legal actions (`advance_legal` clones the whole state, about 1.5 µs a call, Milestone 7 full deck), so a state costs a few advances. Two costs on top came out of profiling (`sample`) and were cut: hashing states with the standard SipHash, card by card, was about 40% of the time, so the tables use `rustc-hash` and each lookup hashes once (the hash-map entry API): 7.6 → 6.0 µs per state on the big plans; and seeding the Mersenne Twister for every reshuffle was a fifth, since most ends of a turn draw the next hand and reshuffle the discard pile from the world's seed, so shuffles are cached by seed and pile size: 6.0 → 4.7 µs. What is left is the engine's: piles as `Vec`s cloned whole on every advance, and legal actions and automatic play recomputed at every state.

**Differential tests**, `tests/star_realms/test_native_planner.py`: 50 positions, 1-4 worlds, budgets 40-3,000 (some run out): the same ends as the states the Python planner scores, in its order, the same feature rows bit for bit, the same states visited and completeness; equal plans with both backends and both packaged models; two whole seeded planner games with identical transcripts, plans and replans. About 6 s.

**Bugs differential testing caught: none.** The enumeration matched on its first run. Four injected bugs each failed the tests: not noting the ends below a reused state (Python's plan then asserts that a candidate reached no end: some candidates are worth only what a state found earlier reaches); noting only a reused state's direct ends; not sorting the discard pile in the merge key; and a world's share one state too big.

**Effort.** About 70 minutes of agent time end to end: reading the planner, encoder and crate, the tests, the port (about 500 lines of Rust), the boundary, the profiling and two optimisations, these measurements and docs.

**Surprises.**

- Rust is only 14 times faster than CPython here, not the hundreds a compiled engine might suggest: the planner is bound by `advance`, which clones the state, and by hashing whole states. The design's "roughly a millisecond" of enumeration per plan holds at the median (1.6 ms) but not the tail (68 ms at the 90th percentile).
- The enumeration doesn't need the value at all: which ends a candidate's best is taken over is fixed by the walk, so the scoring can wait until every end is listed and then run as one matrix call, with no callback into Python.
- A batch of rows and a single row give model probabilities that differ in the last bit (numpy's matrix products sum in another order); the scores are rounded to whole points out of a million, so they agree.

## Milestone 7: Rust self-play with the value's forward pass

2026-10-21, #141 (tracked by #127), CPython 3.14, Apple M1 (8 GB: 4 performance and 4 efficiency cores), **shared**: another agent was running matches on 2 cores throughout (load average 3-9 while measuring). The value's forward pass, the planner's choice and its bot, and whole games across threads, in Rust: one call (`selfplay.play`) plays a batch and returns the training rows and results (design, Compiled engine, "Self-play").

**Throughput**, the costliest pairing a round plays: the planner with the network against the planner with the logistic regression, both at the defaults (4 worlds, 20,000 states per plan), seats alternating, 96 games from seed 50000 (24.5 turns and 56 plans per game, 2,253 training rows), release build, one run per thread count:

| Threads | Games/s | Time of a 20,000-game round |
|---|---|---|
| 1 | 1.31 (0.76 s per game) | 4.2 h |
| 4 | 3.70 | 1.5 h |
| 6 | 4.51 | 1.2 h |
| 8 | 4.58 | 1.2 h |

```python
from minimax_games.games.star_realms.selfplay import Planner, alternating, play
from minimax_games.games.star_realms.value import load
games = alternating(Planner(load("mlp")), Planner(load("logistic")), range(50_000, 50_096))
played = play(games, threads=6)
```

So **a 20,000-game self-play round takes about 1.2 hours** on this machine with 6 threads while it is shared, perhaps an hour with all 8 cores free. Four threads scale 2.8 times, and past 6 threads nothing is gained: two cores were busy, and the efficiency cores are slower. A smaller batch (24 games) scaled worse, 5.6 games/s at 6 threads and less at 8, because the last few long games run alone; a round's thousands of games don't have that tail. A game against the heuristic plans half as often, so a round that mixes in heuristic games is quicker. The time is the planner's enumeration (Milestone 7, planner enumeration: about 5 µs per state); the forward pass costs about 2 µs per end of turn for the network and 0.5 µs for the logistic regression (`predict` over those rows, the copy into Rust included).

**The forward pass against numpy**, on the 10,392 ends the planner lists from every turn start of a seeded heuristic game (one world, 5,000 states): the largest difference in probability is 1.1e-15 for the logistic regression and 4.4e-16 for the network (80% and 57% of the probabilities differ in the last bits), and **no score differs** once rounded to whole points out of a million. The tests' tolerance is 1e-12. Only the order of the products' sums differs: numpy's are blocked and vectorised, Rust's left to right. A score could flip only if `SPAN * (2p - 1)` fell within about 2e-9 of a half point; the nearest of these 10,392 was 5e-5 away, so a flip is a few in a billion ends, and it would change a move only if that end were a candidate's best and the flip reordered the candidates. Ties between candidates can't differ: scores are whole numbers, so their sums are exact and the means are the same correctly rounded division, and both take the first of equal means.

**Agreement with the Python planner.** The tests (about 1.5 s) play three seeded games of the planner (3 worlds, 1,500 states, both models, against greedy and the heuristic, in either seat) with both, and the decisions, winner, turns, plans and replans are identical. Once, outside the suite, at the defaults (4 worlds, 20,000 states), seeds 0-23, both models, against greedy and the heuristic, in either seat: **24 of 24 games identical**, decision for decision, plans and replans included, in 26 s (nearly all of it the Python planner's). The training rows of heuristic games are exactly `fit.dataset`'s, labels and winners included, and a mixed batch gives identical results on 1, 2 and 5 threads.

**Bugs caught: none in the port.** Every differential test passed on its first run. The failures on the way were the tests' own: a real-rows set too small for its assertion, and a check that every game planned in a batch whose last game is heuristic against greedy. Clippy asked for the planner bot to be boxed inside the bot enum (2.6 KB against a few bytes for the others).

**Effort.** About 35 minutes of agent time end to end: reading the planner, values and crate, the tests, the port (about 600 lines of Rust with its unit tests, and 150 of Python), the measurements and these docs.

**Surprises.**

- The port needed nothing new for determinism: the CPython rng port already makes each planner's sampled worlds Python's, games are independent, and std scoped threads with an atomic counter, the results put back in game order, keep a batch identical on any number of threads. No rayon.
- Scores being whole numbers is what makes the agreement exact: the floating-point differences sit about nine orders of magnitude below the rounding, and everything after it (sums, means, ties) is exact.

## Milestone 7: fewer state copies in the Rust engine and planner

2026-10-21, #156 (tracked by #127), CPython 3.14, Apple M1 (8 GB: 4 performance and 4 efficiency cores), **shared**: another agent was running matches on 2 cores throughout (load average 2-3 for the "before" runs, 2-5 for the "after" ones). Before: `main` at 51cdaa3; after: this PR. Same scripts, same machine, release builds (`uv sync`), each measured just before the change and just after.

**What changed** (design, Compiled engine, "Fewer copies"): every pile is a fixed-capacity inline list (`state::Pile`), so a state is 2,560 bytes of plain data and a copy allocates nothing; the planner's tables hold indices into lists of the states it keeps, hashed from the state's fields (the order-free piles as sums over their cards) and compared as multisets, with no sorted copy; a play's state is made in place, a copy of its parent's, and dropped again if it merges with one listed before.

**Time per plan and per state**, the planner's enumeration (`plan_ends`, encoding included) at every turn start of the heuristic-vs-heuristic games of seeds 0-19 (483 turns), 4 worlds sampled from the mover's view (`Game.sample_world`, seeds fixed by game and turn), the default 20,000 states:

| | Before | After | Speed-up |
|---|---|---|---|
| All 483 plans | 9.69 s | 3.30 s | 2.9x |
| Mean plan | 20.1 ms | 6.8 ms | |
| Median plan | 2.14 ms | 0.79 ms | |
| Per state, all plans | 5.74 µs | 1.96 µs | 2.9x |
| Per state, plans of 5,000 states or more (101) | 5.78 µs | 1.96 µs | 2.9x |

(Milestone 7, planner enumeration, measured 4.7 µs per state on the big plans with worlds sampled by the Python planner on a less loaded machine; these runs are slower in absolute terms but compare like with like.) Step by step, per state over all plans: 5.74 µs; inline piles alone 4.96; then tables of indices with the order-free hash, no merge-key copies, 2.73; then states made in place, the lists of states reserved up front, decks hashed by their length and top four cards, and small piles compared as multisets without sorting, 1.91-1.96.

**Self-play throughput**, the benchmark of Milestone 7 (self-play): the planner with the network against the planner with the logistic regression, both at the defaults (4 worlds, 20,000 states), seats alternating, 96 games from seed 50000, one run per thread count:

| Threads | Before, games/s | After, games/s | Speed-up | 20,000-game round, before | after |
|---|---|---|---|---|---|
| 1 | 1.33 | 3.40 | 2.6x | 4.2 h | 1.6 h |
| 4 | 3.54 | 11.91 | 3.4x | 1.6 h | 28 min |
| 6 | 4.42 | 12.98 | 2.9x | 1.3 h | 26 min |
| 8 | 4.49 | 14.22 | 3.2x | 1.2 h | 23 min |

So **a 20,000-game self-play round now takes about 25 minutes** on this machine while it is shared (6 or 8 threads), against 1.2-1.3 hours before. Threads scale better than before (4 threads: 3.5 times one thread, against 2.7): every copy of a state had been 16 allocations and frees, and the threads contended for the allocator.

**Where the time goes**, sampled with macOS `sample` (15-25 s while the benchmark runs), each sample's own time put under the innermost engine function of its stack. Shares of the engine thread's samples:

| | Enumeration, before | after | Self-play (1 thread), before | after |
|---|---|---|---|---|
| Hashing states for the tables | 25% | 10% | 22% | 7% |
| Copying states (clone and drop; nearly all malloc, free and memmove) | 29% | 9% | 26% | 7% |
| Sorting merge keys | 5% | 0% | 5% | 1% |
| Comparing states in the tables | 6% | 16% | 5% | 11% |
| Rules: advance, automatic play, legal actions | 20% | 46% | 20% | 43% |
| Planner bookkeeping: graph, tables, noting ends | 13% | 13% | 11% | 9% |
| Feature encoder | 3% | 7% | 2% | 6% |
| Value's forward pass | | | 9% | 16% |
| All malloc, free and memmove, wherever called | 42% | 28% | 38% | 21% |

Half the time is now the rules engine's own (legal actions and automatic play recomputed at every state, and their small vectors: lists of actions, distinct cards). Making those lists inline too was tried and gave nothing (zero-filling 200-byte inline lists cost what the allocations did), so it was left out. Comparing states is the true merges: a state reached by plays in another order matches one listed before, and its order-free piles are compared as multisets. The next levers, if self-play must be faster still, are apply/undo in the planner's search (design, Why not...), rules that allocate nothing, and a one-byte card, which would halve a state.

**Memory.** A state is 2,560 bytes (it was about 500 plus 16 small heap blocks), its piles' capacities fixed by the card counts (a player owns at most 100 cards: 10 starting, the 80-card Trade Deck and 10 Explorers). A 20,000-state plan keeps up to about 50 MB of states; its lists are reserved up front (at most 25,000 states each), which costs nothing for the pages never written. Eight threads planning at once stay under half a gigabyte.

**Agreement.** Every existing differential test passed unchanged at every step (engine, heuristic, planner, self-play: 479 tests in `tests/star_realms`), so the plans and games are the same, decision for decision. New unit tests: the capacities equal the card counts, a pile panics past its capacity, and over thousands of states from greedy games and their children, `merged` holds exactly when the old sorted merge keys are equal, merged states hash alike, and reordering any pile but the decks and the choices still merges.

**Bugs caught: none.** The differential tests passed on the first run of every step.

**Effort.** About an hour of agent time end to end: profiling, the three steps with a measurement after each, one tried and reverted, the tests and these docs.

**Surprises.**

- Inline piles alone gained only 14%: a copy no longer allocated, but it became a 2.5 KB memmove, and the tables keyed by whole merge keys moved them again on every insertion and rehash. Keeping each state once and the tables as indices is what paid, more than the piles themselves.
- Hashing a state from its top cards is enough. Within a world's turn a deck only changes at the top (draws, ships put on top, the Trade Row refilling from the Trade Deck), so hashing whole decks, 75 cards of Trade Deck in every state, bought nothing; the comparison still checks every card.

Commands:

```python
# Planner enumeration (time per plan and per state).
import statistics, time
from minimax_games._native import star_realms as native
positions = []
for seed in range(20):
    g, turn = native.Game(seed), None
    while g.winner() is None:
        s = g.state()
        if s["turn"] != turn:
            turn = s["turn"]
            positions.append([g.sample_world(s["current"], 1000 * seed + turn * 10 + k).state() for k in range(4)])
        g.advance(g.heuristic_move())
times, states = [], []
for worlds in positions:
    t = time.perf_counter()
    states.append(native.plan_ends(worlds, 20_000)["states"])
    times.append(time.perf_counter() - t)
print(sum(times), statistics.median(times), sum(times) / sum(states))

# Self-play throughput.
from minimax_games.games.star_realms.selfplay import Planner, alternating, play
from minimax_games.games.star_realms.value import load
games = alternating(Planner(load("mlp")), Planner(load("logistic")), range(50_000, 50_096))
played = play(games, threads=6)
```

Profiles: run either in a loop and `sample <pid> 20 -file out.txt` while it runs.
