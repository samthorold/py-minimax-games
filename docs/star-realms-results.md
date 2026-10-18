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
