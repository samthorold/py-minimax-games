# Star Realms: design

Decisions for adding Star Realms to this repo, agreed in a design (grill) session on 2026-10-17. Change them deliberately: update this file in the same PR as the code that departs from it.

## Goal

A **strong Star Realms bot**, using whatever search technique works. Minimax is a means, not the point.

"Strong" is defined by **seeded bot-vs-bot win rates**. Each candidate plays a fixed set of seeded games (e.g. 500, alternating seats) against a ladder of baselines and the previous best. It is stronger if it wins clearly more than 50%, reported with the budgets used and the measured time per move.

The ladder of baselines:

- **random**: a random legal action.
- **greedy**: play everything, buy the most expensive affordable card, attack the opponent directly.
- **heuristic**: a hand-written buying and attacking policy (faction synergy, scrapping Scouts/Vipers, respecting Outposts).

## Rules and scope

- Base set only: an 80-card Trade Deck across four factions (Blob, Trade Federation, Star Empire, Machine Cult), plus Explorers, Scouts and Vipers. No expansions.
- Built in **slices**. The first slice uses only cards whose effects are trade, combat, authority, draw, and ally abilities limited to those. Later slices each add one kind of effect: opponent discards, destroy target base, scrap from hand/discard pile, scrap from the Trade Row, "or" choices, next ship acquired goes on top of the deck.
- Rules references: [RulesPal rulebook](https://rulespal.com/star-realms/rulebook), [Wise Wizard rulebook](https://files.wisewizardgames.com/wwg/rules/SRROE_Rulebook_Web_23-08-24.pdf).
- **The card list is a data file** (e.g. `cards.toml`) transcribed from online sources, with the sources recorded in the file. It holds game data only (name, cost, faction, type, defense, primary/ally/scrap effects, copies), not flavour text or art. It is cross-checked against `../py-realms` (names, costs, counts only; don't reuse its implementation). It lives at `src/minimax_games/games/star_realms/cards.toml` and is loaded by `catalogue.py`. Tests check the per-faction counts, the total of 80, and that every effect named in the file parses into the `Effect` enum (unknown effects fail loading). A card is in the first slice (`Card.in_slice_1`) when all its primary, ally and scrap effects are trade, combat, authority or draw.

## Where it lives

- `src/minimax_games/games/star_realms/`: engine, cards, bots, match runner and tournament.
- `src/minimax_games/search/` grows beyond minimax (ISMCTS joins `alphabeta`). The repo name may be revisited later.

## Engine

- **Immutable state**: frozen dataclasses with tuples for piles. `apply(state, action) -> state` returns a new state sharing unchanged parts. States are hashable, so transpositions can be detected.
- **superenums** for the closed sets, with every `match` checked for completeness:
  - `Action`: `PlayCard`, `UseAlly`, `Scrap`, `Acquire`, `AttackPlayer`, `AttackBase`, `Choose`, `EndTurn`.
  - `Effect`: `Trade`, `Combat`, `Authority`, `Draw`, `OpponentDiscards`, `DestroyBase`, `ScrapHandOrDiscard`, `ScrapTradeRow`, `NextShipOnTop`, `Choice`, plus one variant each for the base-set abilities that fit none of those: `AcquireShipFreeOnTop` (Blob Carrier), `DrawIfBases` (Embassy Yacht), `DrawPerFactionCard` (Blob World), `DiscardThenDraw` (Recycling Station), `ScrapThenDraw` (Brain World), `ScrapFromHand` (Machine Base), `CopyShip` (Stealth Needle), `ShipsGainCombat` (Fleet HQ) and `AllyForAllFactions` (Mech World). Each is implemented in the slice that first needs its card.

  Cards are plain frozen dataclasses loaded from the data file.
- **Moves are micro-actions, with conservative automatic play.** Actions that can't make any later option worse are applied automatically and never branch, e.g. playing a ship whose effects are pure gain, or triggering an ally ability that only adds trade/combat/authority. Everything else is a decision node: buys, scraps, targets, "or" choices, attack allocation, and orderings that interact with draws. Baselines and search bots share the same move generator.
- **Fully specified worlds.** The state always holds a concrete order for every deck, driven by a seeded RNG, so a seed fully determines a game.
- **Observation views.** `observe(state, player)` returns what that player knows: their own hand, everything public, and the contents but not the order of hidden piles. **Bots only ever receive views**, enforced by tests. The one exception is the benchmark-only "cheat" bot.
- **Sampling.** A sampler builds a full world consistent with a view, used by the sampler bot and ISMCTS.

## Search

- **Depth counts turns, not plies.** `alphabeta` gains a generic `ply_cost: Callable[[N], int]` hook, defaulting to 1 per node so other games are unchanged. Star Realms passes 1 at the end of a turn and 0 otherwise, so the search only stops at turn boundaries. The default horizon is 2 turns: my turn plus the opponent's reply.
- **Evaluation**: `evaluate(state, perspective) -> int`, where bigger is better for `perspective`. It starts as a hand-written linear function of:
  - the Authority difference;
  - base defense, with Outposts weighted higher;
  - deck economy: average trade and combat per card over the whole deck;
  - a penalty per Scout/Viper still in the deck.

  The weights live in one place and are tuned by tournament. Rollouts (playing out with the greedy policy) are the next contender. Learned evaluation is deferred.
- **Bots**, in the order they're built:
  1. **cheat**: alpha-beta on the true world. It's a benchmark and upper bound, not a legitimate player.
  2. **sampler (PIMC)**: sample K worlds from the bot's view, run depth-limited alpha-beta on each, and vote on the move.
  3. **ISMCTS**: in `search/`, sampling a world on each iteration, with a rollout policy.
- **Deterministic budgets**: worlds × turn depth for the sampler, iterations for ISMCTS, and a number of rollouts. No wall-clock limits in anything we measure. Bots are compared at equal budgets.
- **Performance**: benchmark nodes per second once the cheat works. The target is a 500-game tournament between two search bots in about 10 minutes on a laptop. Optimise the state representation only if the benchmark says search depth limits strength.

## Harness

- A plain match loop (deal; ask the current bot for an action given its view; apply; repeat until someone's Authority reaches 0) and a plain tournament loop over seeds and pairings. No py-ev: the base set has no timing to model.
- **Game log**: a replayable text transcript per seed. It is **opt-in** (off by default, so tournaments don't pay for I/O) and records only the actions the match runner applies, never search nodes. Nothing is formatted when it's off. `play_match(..., log=sink)` takes a `Callable[[str], None]` sink (e.g. `print` or `list.append`); `transcript.describe(action, state)` formats one action with an exhaustive `match` over `Action`, so each new action kind needs one new case. `python -m minimax_games.games.star_realms --seed 7 --log` prints a game's transcript. Search code logs only behind `logger.isEnabledFor(logging.DEBUG)`.
- A human-playable CLI is deferred. When it comes, it's another bot whose decisions come from `input()`.

## Dependencies

- `superenums` from GitHub, pinned to a tag via `[tool.uv.sources]` (it isn't on PyPI). `uv.lock` pins the commit. A consequence is that the released wheel's dependency on `superenums` isn't resolvable from PyPI. That's acceptable while this is run from a checkout; publishing superenums to PyPI is the fix if it ever isn't.

## Milestones

Each milestone ends with tournament results recorded in the repo.

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
