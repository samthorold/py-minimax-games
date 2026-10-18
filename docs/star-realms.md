# Star Realms: design

Decisions for adding Star Realms to this repo, agreed in a design (grill) session on 2026-10-17. Change them deliberately: update this file in the same PR as the code that departs from it.

## Goal

A **strong Star Realms bot**, using whatever search technique works. Minimax is a means, not the point.

"Strong" is defined by **seeded bot-vs-bot win rates**. Each candidate plays a fixed set of seeded games (e.g. 500, alternating seats) against a ladder of baselines and the previous best. It is stronger if it wins clearly more than 50%, reported with the budgets used and the measured time per move.

The ladder of baselines:

- **random**: a random legal action.
- **greedy**: play everything, buy the most expensive affordable card (repeatedly), attack the opponent directly, destroying the weakest Outpost first while Outposts guard them; never scrap. `GreedyBot`'s docstring has the exact priority order.
- **heuristic**: a hand-written buying and attacking policy: faction synergy, economy early and combat late, valuing bases, scrapping Explorers (and anything with a combat scrap ability when that wins), destroying bases before hitting the player and Outposts sensibly. `HeuristicBot`'s docstring has the exact priority order. Scrapping Scouts and Vipers needs scrap-from-hand/discard effects, which arrive with a later card slice (milestone 4), so it isn't possible with the current cards yet.

## Rules and scope

- Base set only: an 80-card Trade Deck across four factions (Blob, Trade Federation, Star Empire, Machine Cult), plus Explorers, Scouts and Vipers. No expansions.
- Built in **slices**. The first slice uses only cards whose effects are trade, combat, authority, draw, and ally abilities limited to those. Later slices each add one kind of effect: opponent discards, destroy target base, scrap from hand/discard pile, scrap from the Trade Row, "or" choices, next ship acquired goes on top of the deck.
- Rules references: [RulesPal rulebook](https://rulespal.com/star-realms/rulebook), [Wise Wizard rulebook](https://files.wisewizardgames.com/wwg/rules/SRROE_Rulebook_Web_23-08-24.pdf).
- **The card list is a data file** (e.g. `cards.toml`) transcribed from online sources, with the sources recorded in the file. It holds game data only (name, cost, faction, type, defense, primary/ally/scrap effects, copies), not flavour text or art. It is cross-checked against `../py-realms` (names, costs, counts only; don't reuse its implementation). It lives at `src/minimax_games/games/star_realms/cards.toml` and is loaded by `catalogue.py`. Tests check the per-faction counts, the total of 80, and that every effect named in the file parses into the `Effect` enum (unknown effects fail loading). A card is in the first slice (`Card.in_slice_1`) when all its primary, ally and scrap effects are trade, combat, authority or draw.

## Where it lives

- `src/minimax_games/games/star_realms/`: engine, cards, bots, match runner and tournament.
- `src/minimax_games/search/` grows beyond minimax (ISMCTS joins `alphabeta`). The repo name may be revisited later.

## Engine

- **Immutable state**: frozen dataclasses with tuples for piles. A `Card` hashes by its name (unique in the catalogue), which is cheap and agrees with equality. `apply(state, action) -> state` returns a new state sharing unchanged parts. States are hashable, so transpositions can be detected.
- **superenums** for the closed sets, with every `match` checked for completeness:
  - `Action`: `PlayCard`, `UseAlly`, `UseBase`, `Scrap`, `Acquire`, `AttackPlayer`, `AttackBase`, `Choose`, `EndTurn`.

    Bases stay in play across turns, in their own per-player pile, and count as in play for ally abilities and scrapping. A base's primary ability is used once in each of its owner's turns: playing the base uses it that turn, and `UseBase` uses it on later turns. It is an explicit action, not applied at the start of the turn, because when to use it can matter (draws, and the bases with choices or targets in later slices); automatic play of pure-gain abilities takes it over like any other action. `AttackBase` spends combat equal to a base's defense and puts the base in its owner's discard pile. `AttackPlayer` spends all the combat. While a player has an Outpost, only Outposts can be attacked.
  - `Effect`: `Trade`, `Combat`, `Authority`, `Draw`, `OpponentDiscards`, `DestroyBase`, `ScrapHandOrDiscard`, `ScrapTradeRow`, `NextShipOnTop`, `Choice`, plus one variant each for the base-set abilities that fit none of those: `AcquireShipFreeOnTop` (Blob Carrier), `DrawIfBases` (Embassy Yacht), `DrawPerFactionCard` (Blob World), `DiscardThenDraw` (Recycling Station), `ScrapThenDraw` (Brain World), `ScrapFromHand` (Machine Base), `CopyShip` (Stealth Needle), `ShipsGainCombat` (Fleet HQ) and `AllyForAllFactions` (Mech World). Each is implemented in the slice that first needs its card.

  Cards are plain frozen dataclasses loaded from the data file.
- **Moves are micro-actions, with conservative automatic play.** Actions that can't make any later option worse are applied automatically and never branch. Everything else is a decision node: buys, scraps, targets, "or" choices, attack allocation, and orderings that interact with draws. Baselines and search bots share the same move generator.

  The rule lives in one place, `rules.automatic(action)`. An action is automatic when it is:
  - `PlayCard` of a **ship** whose primary effects are all pure gain (trade, combat, Authority). Hand and ships in play both go to the discard pile at the end of the turn, and in play the ship only adds options (ally abilities, scrapping);
  - `UseAlly` whose ally effects are all pure gain;
  - `UseBase` whose primary effects are all pure gain.

  Not automatic: anything with `Draw` (a draw can reshuffle the discard pile, and acquired cards go there, so drawing before or after a buy changes the deck); **playing a base**, even a pure-gain one (a base left in hand goes to the discard pile and is reshuffled, played it stays out of the deck, so it changes deck composition); `Scrap`, `Acquire`, attacks and `EndTurn`. Each later slice that adds an effect or action (opponent discards, scrapping from hand, ...) must revisit the rule.

  `rules.auto_play(state)` applies automatic actions, in `legal_actions` order, until none is left; `rules.advance(state, action)` applies a decision then auto-plays. `rules.advance_legal` is the same for an action taken from `legal_actions(state)`: it skips `apply`'s legality check (which recomputes `legal_actions`), so the search uses it, while bots' actions go through `advance` and are checked. The match runner auto-plays after dealing and after every action, so bots are only offered decisions. `rules.apply` stays the bare rules, without automatic play. Tests brute-force every way to play a turn from small hand-built states with and without automatic play: every outcome with it is reachable without, and every outcome without it is matched by one with it that leaves the player at least as much Authority and the opponent at most as much (a pure-gain card left unplayed only loses Authority or damage).
- **Fully specified worlds.** The state always holds a concrete order for every deck, driven by a seeded RNG, so a seed fully determines a game.
- **Observation views.** `observe(state, player)` returns what that player knows: their own hand, everything public, and the contents but not the order of hidden piles. **Bots only ever receive views**, enforced by tests. The one exception is the benchmark-only "cheat" bot: a bot marks itself benchmark-only by implementing `bots.BenchmarkBot` (`choose_from_state(state, actions)`) instead of `bots.Bot` (`choose(view, actions)`), and only then does the match runner give it the true state. The tournament's timing wrapper keeps that marking.
- **Sampling.** A sampler builds a full world consistent with a view, used by the sampler bot and ISMCTS.

## Search

- **Depth counts turns, not plies.** `alphabeta` gains a generic `ply_cost: Callable[[N], int]` hook, defaulting to 1 per node so other games are unchanged. Star Realms passes 1 at the end of a turn and 0 otherwise, so the search only stops at turn boundaries. The default horizon is 2 turns: my turn plus the opponent's reply.
- **Transpositions are merged.** `alphabeta` takes an optional generic `key: Callable[[N], Hashable]` hook (default `None`: no table, so other games are unchanged). With it, results are stored by key and remaining depth, with whether each is exact or a fail-soft lower or upper bound, and reused only where valid for the current window. A reused leaf's `moves` are rebuilt from the current path, so callers can still read the principal line from the returned leaf. A `table` argument lets searches share one (for iterative deepening).
- **Evaluation**: `evaluate(state, perspective) -> int`, where bigger is better for `perspective`. It starts as a hand-written linear function of:
  - the Authority difference;
  - base defense in play, with Outposts weighted higher;
  - deck economy: average primary trade and combat per card over every card the player owns (deck, hand, in play, discard and bases);
  - a penalty per Scout/Viper still owned.

  Each term is scored for both players and differenced, so the score is antisymmetric. A finished game scores `WIN` (or `-WIN`), and other scores are capped strictly inside that. The weights live in one place (`evaluate.Weights`, default `WEIGHTS`) and are tuned by tournament. Rollouts (playing out with the greedy policy) are the next contender. Learned evaluation is deferred.
- **Bots**, in the order they're built:
  1. **cheat**: alpha-beta on the true world. It's a benchmark and upper bound, not a legitimate player. `node.StarRealmsNode` adapts a state to `search.Node` from the searching player's side: children from `legal_actions` + `rules.advance_legal`, `score()` from `evaluate` (a finished game's `WIN` moved one point toward 0 per move, so quicker wins and slower losses rank higher), and `node.ends_turn` as the `ply_cost`. `CheatBot(turns=2)` plays the first move of the best line; it is `cheat` in the CLI. The search uses `alphabeta`'s transposition table (its generic `key` hook, here `node.transposition_key`: the state and the number of moves to it, since a finished game's score depends on that), so the many orderings of a turn's micro-actions that reach the same state are searched once. Each decision has a deterministic node budget (`CheatBot(budget=5_000)` by default, `cheat.BUDGET`, chosen from the benchmark so a 2-turn game takes about 11 s): `cheat.decide` deepens one turn at a time, 1 then 2 up to the horizon, and plays the best move of the deepest search that finished within the budget, or, if not even the 1-turn search finished, the best of the root moves it finished (the first legal action if none). Nodes are charged as they are generated by a `StarRealmsNode` subclass that raises once the budget is spent, so `alphabeta` is unchanged and a decision never passes its budget. The tournament command prints the cheat's horizon and budget.
  2. **sampler (PIMC)**: sample K worlds from the bot's view, run depth-limited alpha-beta on each, and vote on the move.
  3. **ISMCTS**: in `search/`, sampling a world on each iteration, with a rollout policy.
- **Deterministic budgets**: worlds × turn depth for the sampler, iterations for ISMCTS, and a number of rollouts. No wall-clock limits in anything we measure. Bots are compared at equal budgets.
- **Performance**: benchmark nodes per second once the cheat works. `bench` (`python -m minimax_games.games.star_realms bench`, or `task bench`) runs the cheat's search (`cheat.search`) on fixed positions, each the start of a turn in a seeded greedy-vs-greedy game (`bench.position(seed, turn)`), at horizons 1 and 2 turns. Nodes are counted by a `StarRealmsNode` subclass, so `alphabeta` is unchanged, and each search stops past a deterministic node budget (`--max-nodes`), so node counts are reproducible and the command always finishes; only timings vary. `--states` also counts distinct states, in each search and exactly over each position's whole first turn (`bench.turn_tree`, which merges transpositions), to size the transposition problem. `--budget N` times the cheat's budgeted decisions (`cheat.decide`) at the same positions instead: the depth each finished, its nodes and its time. Results are in [star-realms-results.md](star-realms-results.md). The target is a 500-game tournament between two search bots in about 10 minutes on a laptop. Optimise the state representation only if the benchmark says search depth limits strength.

## Harness

- A plain match loop (deal; ask the current bot for an action given its view; apply; repeat until someone's Authority reaches 0) and a plain tournament loop over seeds and pairings. No py-ev: the base set has no timing to model.
- **Game log**: a replayable text transcript per seed. It is **opt-in** (off by default, so tournaments don't pay for I/O) and records only the actions the match runner applies, automatic ones marked "(auto)", never search nodes. Nothing is formatted when it's off. `play_match(..., log=sink)` takes a `Callable[[str], None]` sink (e.g. `print` or `list.append`); `transcript.describe(action, state)` formats one action with an exhaustive `match` over `Action`, so each new action kind needs one new case. `python -m minimax_games.games.star_realms --seed 7 --log` prints a game's transcript. Search code logs only behind `logger.isEnabledFor(logging.DEBUG)`.
- **Tournament**: `python -m minimax_games.games.star_realms tournament --bots greedy random --games 500 [--base-seed 0] [--max-turns 300]` plays game `i` from seed `base_seed + i`, bot A moving first in even games and bot B in odd ones. Each bot is built per game from the game seed (seat 0 gets `seed`, seat 1 `seed + 1`, as in the single-game command, so any game replays on its own). It reports win rates overall and by seat, the mean number of turns and each bot's mean milliseconds per decision (time inside `choose`, measured with `time.perf_counter`). The log is off. `--jobs N` plays the games in N processes, each a run of consecutive seeds starting with bot A first, and combines them (`tournament.combine`): the results are the same as one process's, and each bot's time per decision is measured in the process that made it. A game past `--max-turns` is **unfinished**: listed by seed and left out of every win rate, whose denominator (the finished games) is printed. Results (`tournament.TournamentResult`) are identical for the same seeds and bots; timings (`tournament.Timing`) are kept apart because they vary.
- A human-playable CLI is deferred. When it comes, it's another bot whose decisions come from `input()`.

## Dependencies

- `superenums` from GitHub, pinned to a tag via `[tool.uv.sources]` (it isn't on PyPI). `uv.lock` pins the commit. A consequence is that the released wheel's dependency on `superenums` isn't resolvable from PyPI. That's acceptable while this is run from a checkout; publishing superenums to PyPI is the fix if it ever isn't.

## Milestones

Each milestone ends with tournament results recorded in the repo, in [star-realms-results.md](star-realms-results.md).

1. **Engine core and harness**:
   - the first-slice card data and its validation test;
   - immutable state, seeded RNG, `observe`, legal moves with automatic play;
   - random and greedy baselines;
   - the match runner with an opt-in log;
   - a tournament command reporting win rates by seat.
2. **Cheat**:
   - the `ply_cost` hook in `alphabeta`;
   - linear `evaluate`;
   - the cheat bot at a 2-turn horizon;
   - the heuristic baseline;
   - a nodes/sec benchmark.
3. **Sampler**: sampling worlds from views, plus voting; tests that bots only receive views.
4. **Card slices**: one per remaining kind of effect, each with a tournament check that nothing regressed.
5. **ISMCTS** with a rollout policy, and rollout-based `evaluate` as a contender.

## Why not...

- **Chance nodes / expectiminimax**: they would change `Node` and `alphabeta`. Pruning with chance nodes is weak, and almost every draw in Star Realms is a chance event. Sampling worlds keeps `search/` two-player.
- **Whole-turn moves**: listing a turn's outcomes is itself a search over micro-actions, and mid-turn draws break it.
- **Mutable state with apply/undo**: faster, but undo across hidden information, reshuffles and RNG state is bug-prone, and one silent bug invalidates every win-rate comparison.
