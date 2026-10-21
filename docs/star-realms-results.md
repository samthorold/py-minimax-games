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
