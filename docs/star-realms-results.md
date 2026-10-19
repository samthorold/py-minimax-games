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
