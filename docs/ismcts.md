# ISMCTS: design reference

Research notes for issue #117 (generic ISMCTS in `search/`, and an ISMCTS bot for Star Realms), written 2026-10-20 and committed with the implementation (`src/minimax_games/search/ismcts.py`, `src/minimax_games/games/star_realms/ismcts.py`). Every claim about the literature cites a source in [References](#6-references); anything not checked against the source itself is marked **(unverified)**. Claims about this repo cite its docs or code. Measurements are in [star-realms-results.md](star-realms-results.md), "Milestone 5: ISMCTS".

**Outcome** (results, Milestone 5: ISMCTS): at the sampler's time per decision, ISMCTS plays level with it, as §1 predicted: 34.6% against the heuristic to the sampler's 33.4% on seeds 0-499 (+6 on 184 discordant seeds, p = 0.71), and 48.6% head to head. 4x the iterations gained nothing on seeds 0-199. The leaf settings mattered more than the exploration constant.

**Where the implementation departs from these notes** (each also marked in place below):

- `sample` takes no argument (`Callable[[], S]`): it is a closure over its own rng, so the worlds' random numbers are separate from the expansions' and playouts' (the search's `rng`), and changing a playout setting never changes which worlds are sampled (§2.9).
- `Result` carries the tree's root (`tree`, a `Node` with `children`, `visits`, `reward`, `available`, `mover`) instead of a node count, so tests and diagnostics can read availability counts, tree size and depth. Nodes don't store their action: a node is its parent's `children[action]`.
- The single-legal-action shortcut lives in the bot (`IsmctsBot.choose`), not in the generic search, which needs at least one iteration.
- A legal action that is never expanded is not counted as available: availability is counted for existing children, as in [1, Algorithm 1], so a new child starts at 1.
- The leaf value reuses #116's pieces: `rollout.playout(state, 1 + extra_turns, epsilon, rng, policy)` (the policy's moves read from the state, each replaced with probability ε by a random move other than `EndTurn`; a playout ends after that many turn ends, the rest of the leaf's turn counting as one), then `rollout.win_probability` for player 0. Its `scale` (16) was fitted by #116 for the current default weights (#115's tuning left them unchanged), so it was adopted, not refitted. #116's rollout *evaluator* inside alpha-beta did not beat the linear one, so the default plan of §2.5 was kept.
- **Default ε is 0.25**, not 0.1: the best of 0, 0.1, 0.25 and 0.5 on the tuning seeds. `c` stays 0.7 (0.35 and 1.0 were no different) and `extra_turns` 0 (1 lost clearly at equal compute).
- Two leaf options were added and measured, and lost or tied, so they are off by default: `policy="heuristic"` (`rollout.heuristic`: `HeuristicBot` on the mover's view, about 1.3x the cost per iteration; §2.5 suggested it), and `horizon` (play every leaf out to the end of the same turn from the root, at least to the end of its own). The second answers a bias in "finish the current turn": an `EndTurn` leaf is scored after the opponent's whole reply, its siblings before it.
- The default budget is **450 iterations**, not 100-250: the state-level greedy playout (no `observe`) made iterations cheaper than §2.8's estimate, about 0.2 ms each on CPython.

## 1. Summary and recommendation

**Variant: single-observer ISMCTS (SO-ISMCTS)** with UCB1 modified by availability counts (Cowling, Powley and Whitehouse 2012, [1]). In Star Realms every action is public: both players see each card played, bought, scrapped, discarded and attacked. Cowling et al. note that "all three variants of ISMCTS are equivalent for a game with fully observable moves" [1, §VII-B], so SO-ISMCTS+POM and MO-ISMCTS (one tree per player) add nothing here. They only differ when some moves are partly hidden.

**Why ISMCTS can help here, and why the gain may be small.** In this game the hidden information is almost all *chance*: deck orders, the Trade Deck order, and the opponent's hand. At any of our decisions the opponent's hand is a fresh random draw that they haven't acted on yet. So `view.sample_world`'s uniform deal is the exact posterior, and nothing about opponents needs to be inferred. The sampler's real weakness is *clairvoyance* (a form of strategy fusion [1, §III-B]). On each sampled world, alpha-beta knows exactly what a Draw will bring, what the next hand is, and which card will refill the Trade Row. It plays as if it could choose differently for each outcome. ISMCTS samples a new world on every iteration and keys its tree by action history, so a node's value is averaged over chance outcomes and the clairvoyance goes away. Long et al.'s analysis of PIMC [2], however, predicts that PIMC does well in games where hidden information is revealed quickly (high disambiguation) and leaf values are strongly correlated. Star Realms reveals each hand as it is played, and a lead in economy tends to stay a lead. The repo's own numbers also show sampling costs little: the sampler trails the cheat by about 5 points at equal compute (docs/star-realms-results.md, Milestones 3 and 4). So **don't expect ISMCTS to beat the sampler by much at equal compute.** Its value is a search that handles chance correctly, with the same evaluation, and a base for the enhancements in §5.

**Concrete design (details in §2):**

1. `search/ismcts.py`: a generic, two-player, zero-sum SO-ISMCTS. The game supplies a small protocol (legal actions, apply, player to move, terminal, a leaf value in [0, 1]), plus a `sample(rng)` callable that returns a world consistent with the root information set. Tree nodes are keyed by the action path from the root ("open loop" over chance). No states are stored in the tree: every iteration replays its path on a freshly sampled world.
2. **Selection**: UCB1 over the actions legal *in this iteration's world*, with the parent visit count replaced by the child's availability count [1, §IV-B]. Exploration constant **0.7** with rewards in [0, 1], as in [1, §IV-A]. Expand one random untried legal action per iteration.
3. **Leaf value (Star Realms)**: don't play to the end of the game. A greedy playout to the end costs 3.5-7.5 ms here (§2.8), so only about 15-30 would fit in the sampler's time per decision. Instead, finish the current turn with a fast, slightly randomised greedy policy, then score the position at the turn boundary as `sigmoid(evaluate / scale)`, with `scale` calibrated from game outcomes. Make the number of extra turns a setting (default 0, sweep 0-1). Use #116's rollout evaluator if it is the better one by then. A terminal state scores 1 or 0.
4. **Final move**: the most visited root child, ties broken in `legal_actions` order. If only one action is legal, return it without searching.
5. **Budget**: a number of iterations, deterministic for a seed. Pick the default by measuring ms/decision against the sampler's (about 100 ms alone, Milestone 3). From the microbenchmarks in §2.8, expect only about **150-500 iterations** at that cost on CPython: 1-5% of the 10,000 iterations the literature typically uses [1][4]. Also report `advance_legal` calls per decision, the closest unit to the sampler's node count.
6. **Determinism**: one `random.Random(seed)` per bot. Each decision derives separate rngs for worlds and for playouts. Iterate only over sequences or insertion-ordered dicts, never sets (`str` hashes are salted per process, and cards hash by name).

## 2. Recommended design for #117

### 2.1 The generic protocol (`search/ismcts.py`)

The issue asks for "sample a world, legal actions, apply, terminal/result, player to move, and an information-set key per action history". In SO-ISMCTS with fully observable moves the information-set key *is* the action path from the root, so no separate key hook is needed: a node is identified by its parent and the action leading to it. Actions must be hashable, and equal across worlds when they mean the same move. Star Realms actions name cards, and cards hash and compare by name, so this holds.

```python
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from random import Random
from typing import Protocol


class Game[S, A: Hashable](Protocol):
    """A two-player, zero-sum game with fully observable moves, seen as
    determinized worlds `S`. Players are 0 and 1."""

    def actions(self, state: S) -> Sequence[A]:
        """Legal actions, in a fixed order (ties are broken by it).
        Empty only when terminal."""

    def apply(self, state: S, action: A) -> S:
        """The state after `action`, including any automatic play and chance:
        chance is resolved by the world's own seeded randomness."""

    def to_move(self, state: S) -> int: ...

    def is_terminal(self, state: S) -> bool: ...

    def value(self, state: S, rng: Random) -> float:
        """Player 0's reward in [0, 1]: exact (1, 0 or 0.5) if terminal,
        otherwise an estimate (rollout, evaluation, or both)."""


@dataclass(frozen=True, slots=True)
class Result[A]:
    move: A
    visits: dict[A, int]   # root children's visits, in first-seen order
    iterations: int
    nodes: int             # tree size: memory and diagnostics


def ismcts[S, A: Hashable](
    game: Game[S, A],
    sample: Callable[[Random], S],   # a world consistent with the root information set
    iterations: int,
    rng: Random,
    exploration: float = 0.7,
) -> Result[A]: ...
```

*As built:* `sample` is `Callable[[], S]` and `Result` has `tree` (the root `Node`) in place of `nodes`; see the departures above.

Notes:

- `sample` is a closure over what the searcher knows; for Star Realms, `lambda rng: sample_world(view, rng)` (as built: `lambda: sample_world(view, worlds)`, with the decision's own worlds rng). The search never sees a true state. For a perfect-information test game, `sample` returns the state itself, and ISMCTS reduces to plain UCT.
- The game's `value` holds the playout or evaluation, so the generic code knows nothing about turns, rollout policies or `evaluate`, and #116's rollout evaluator plugs in as one `value`. A helper `rollout(game, policy, stop, rng)` can live in `search/` if two games ever need it; don't add it before then.
- Keep it two-player and zero-sum, as `alphabeta` is. A reward vector per player would generalise it, but nothing here needs that.

### 2.2 The algorithm

The search uses one node class with `__slots__`: `action`, `mover` (the player who chose `action`), `children: dict[A, Node]`, `visits`, `reward` (summed from `mover`'s side) and `available`.

```
ismcts(game, sample, iterations, rng, c):
    root = Node(action=None, mover=None)
    for i in 1..iterations:
        world = sample(rng)                      # a new determinization every iteration
        node, path = root, [root]
        while not game.is_terminal(world):
            legal = game.actions(world)          # this world's actions: restricts the tree
            mover = game.to_move(world)
            for a in legal:                      # availability: every child legal here
                if a in node.children: node.children[a].available += 1
            untried = [a for a in legal if a not in node.children]
            if untried:                          # expansion: one new node per iteration
                a = untried[rng.randrange(len(untried))]
                child = node.children[a] = Node(a, mover, available=1)
                world = game.apply(world, a); path.append(child)
                break
            node = argmax over a in legal (first in `legal` order on ties) of
                   node.children[a].reward / node.children[a].visits
                   + c * sqrt(ln(node.children[a].available) / node.children[a].visits)
            world = game.apply(world, node.action); path.append(node)
        v = game.value(world, rng)               # player 0's reward in [0, 1]
        for n in path[1:]:
            n.visits += 1
            n.reward += v if n.mover == 0 else 1 - v
        root.visits += 1
    move = root child with most visits (ties: higher mean reward, then `legal` order at the root)
```

This follows Cowling et al.'s Algorithm 1 and their detailed pseudocode [1, §IV-E, Appendix A]. Restricting the descent to actions legal in the iteration's determinization makes "the probability of an action being available for selection on a given iteration … precisely the probability of sampling a determinization in which that action is available" [1, §IV-E]. The availability count replaces the parent's visit count in UCB1. Without it, actions available in only a few worlds are over-explored: "whenever they are available for selection their ratio of visits to parent visits is very small, resulting in a disproportionately large UCB value" [1, §IV-B]. Here availability matters at the opponent's nodes (their legal plays depend on their sampled hand) and at our own nodes after a draw. At the root it changes nothing: the root's legal actions are the same in every world, because `legal_actions` for the player to move depends only on their own hand and public state, which is also why the sampler can assert that each world's move is one of the view's actions (`sampler.vote`).

The per-iteration cost is `sample` plus, at every tree level, `actions` + `apply`, plus `value`. States are never cached in nodes: a node's state differs from world to world.

### 2.3 How Star Realms maps onto the tree

- **Micro-actions are tree edges.** Each decision `legal_actions` offers is one edge, so a turn is a chain of nodes with the same mover. This is the "split moves" or move-decomposition structure that Cowling, Ward and Powley found stronger than whole-move nodes in Magic: The Gathering [4], and that Kowalski et al. found "can be greatly beneficial" in general game playing [13].
- **Automatic play is inside `apply`.** The adapter's `apply` is `rules.advance_legal`, so automatic play happens inside the edge, as it does for the cheat and in matches. It never creates nodes.
- **`_forced` decisions are ordinary nodes.** While one is pending, `legal_actions` returns only its actions. At the opponent's forced `Discard`, those actions depend on their sampled hand, and availability counts handle that.
- **Turn ends.** `EndTurn` is an edge. Its `apply` deals the next hand and runs the opponent's automatic play, chance included. The tree carries on into the opponent's turn with them as `mover`, and nodes keep track of whose decision each one is.
- **Chance (draws, reshuffles, Trade Row refills)** is resolved inside `apply` from the world's `rng_seed`, which `sample_world` draws fresh. Each iteration therefore sees new chance outcomes, and the tree doesn't branch on them: it is "open loop" over chance, as open-loop MCTS is for stochastic games [12]. Its nodes after a chance event mix several of the searcher's real information sets (for example our next hand). The price is that the tree can't make a later decision depend on a draw it would really see, so it is slightly pessimistic deep in the tree. Splitting nodes on the searcher's own observations would fix that but multiplies the branching; it is ranked low in §5. Cowling et al. handle chance with explicit environment nodes only where outcomes are few and equally likely [1, §III-C-2]. Star Realms has hundreds of possible hands, so that doesn't fit.
- **The opponent** chooses in the same tree. Its statistics are averaged over its possible hands, but it can only pick actions legal with its sampled hand. This is SO-ISMCTS's opponent model [1, §IV-E]. With fully observable moves, MO-ISMCTS would build the same tree [1, §VII-B].

### 2.4 Hidden information

- The bot is an ordinary `bots.Bot`: `choose(view, actions)`. Its `sample` is `lambda rng: sample_world(view, rng)`. That is the only way worlds enter the search.
- Uniform sampling is the right belief here (§1): the opponent's hand at our decision is a random deal they haven't acted on, and `sample_world` already keeps the known top cards (`deck_top`) in place. No opponent modelling or inference is needed.
- `evaluate` is effectively a function of the information set. It scores Authority, bases and every card a player *owns*, which is the same multiset in every sampled world. So scoring a determinized world leaks nothing.

### 2.5 The leaf value and reward scale

Rewards must be in [0, 1] for the usual exploration constants: Kocsis and Szepesvári's `Cp = 1/√2` assumes that range ([5], citing [6]). A finished game gives 1 or 0.

Evaluate only at **turn boundaries**. The cheat's search only stops at turn ends (`node.ends_turn` as `ply_cost`), for good reason: `evaluate` doesn't score unspent trade or combat, so a mid-turn state looks the same before and after its buys and attacks have been worked out. So `value` should:

1. play out the rest of the current turn (up to `EndTurn`) with a cheap policy;
2. optionally play `extra_turns` more whole turns (default 0, sweep 0 and 1);
3. score `sigmoid(evaluate(world, 0) / scale)`, or 1 or 0 if the game ended.

**The playout policy should be greedy with some randomness, not uniform random.** Measured here (§2.8): from three benchmark positions, a uniformly random policy ends its turn after 3-5 decisions, because `EndTurn` is always legal, leaving trade and combat unspent. The greedy policy plays 3-14 decisions in a turn. Random play can also stall a game for 1,000+ turns (Milestone 4). The Magic study found that a fully deterministic expert playout policy was "clearly inferior" to a weaker rules policy with some randomness, although the expert policy was the stronger player [4]. Lanctot et al. used an ε-greedy playout policy as one of their baselines [7, §4]. A reasonable starting point is ε-greedy on top of `GreedyBot`'s priorities. ε = 0.1 is a guess to tune, **(not from a source)**. If the heuristic proves cheap enough in #116's measurements, try it as well.

**Calibrating `scale`.** `evaluate` returns points of Authority (`evaluate.Weights`), with `WIN = 10**9` for finished games. Fit `P(win) = sigmoid(evaluate / scale)` by one-dimensional maximum likelihood on states at turn starts from a few hundred seeded heuristic-vs-heuristic or greedy-vs-greedy games (these run at about 0.01 ms per move, so it takes seconds). That gives a calibrated win probability with no tournament needed. Record the fitted `scale` with the weights it was fitted for, and refit when #115 changes them. If #116's rollout evaluator returns a win fraction, it is already in [0, 1].

**Implicit minimax backups** [7] (§5) are a later option that uses `evaluate` inside the tree as well.

### 2.6 Final move, budget and matching compute

- **Final move**: the robust child (most visits) [5, §3.1]. Cowling et al. return the root action whose child has the most visits [1, Algorithm 1]. Break ties by mean reward, then by `legal_actions` order, as the sampler's vote does.
- **One legal action**: return it without searching. This is deterministic and free. It changes the ms/decision average, so state it in the results. (As built, the bot does this.)
- **Budget**: `iterations`. `IsmctsBot(seed, iterations=N, exploration=0.7, extra_turns=0, ...)`, whose `settings()` reports all of these and the leaf-value settings.
- **Matching the sampler**: alpha-beta counts nodes (generated children), and ISMCTS counts iterations. These units are not comparable, because one iteration does a descent of several `apply` calls plus a playout. Choose `iterations` so ms/decision in real games, measured alone, matches the sampler's at its defaults: about 101 ms alone, or 125-158 ms contended under `--jobs 3` (Milestone 3: Decisions in real games; the ladders). Also report the mean `advance_legal` calls per decision, counted with a wrapper as `bench` counts nodes, so compute can be compared in a unit close to the sampler's nodes. Also run one larger setting (for example 4×), as #116 does, to see whether ISMCTS improves with budget.

### 2.7 Defaults and why

| Setting | Default | Why |
|---|---|---|
| Variant | SO-ISMCTS, availability counts | equivalent to the others for fully observable moves [1, §VII-B]; availability counts [1, §IV-B] |
| Exploration `c` | 0.7 | used for every algorithm in [1]; they found the algorithms "not particularly sensitive" to it, with strength falling outside a range **(the exact range didn't survive text extraction: unverified)**. A squashed evaluation has lower variance than win/loss, so smaller `c` may do better: sweep 0.35, 0.7, 1.0 |
| Leaf | finish the turn with ε-greedy, then `sigmoid(evaluate/scale)` (as built: ε = 0.25) | turn boundaries as in the cheat; full games cost too much (§2.8); some randomness in playouts [4] |
| `extra_turns` | 0 (sweep 0, 1) | each extra turn roughly doubles the cost of an iteration (§2.8) |
| Expansion | random untried action (seeded) | as in [1, Algorithm 1]; progressive bias in §5 |
| Final move | most visits | [1][5] |
| Iterations | chosen to match the sampler's ms/decision; expect 150-500 (as built: 450) | §2.8 |

### 2.8 Costs measured here (to size the budget)

These are light microbenchmarks at three of `bench`'s positions (`bench.position(1, 7)`, `(1, 15)`, `(2, 13)`), CPython 3.14, run **while another job ran tournaments**: read them as rough upper bounds. The script is in the scratchpad (`scripts/costs.py`), not in the repo.

| Operation | Cost |
|---|---|
| `sample_world(view, rng)` | 20-32 µs |
| `observe(state, p)` | 14-15 µs |
| `legal_actions` | 6-7 µs |
| `advance_legal` | 15-42 µs |
| `evaluate` | 9-14 µs |
| random playout, rest of this turn | 0.2-0.3 ms (3-5 decisions) |
| greedy playout, rest of this turn | 0.3-0.9 ms (3-14 decisions) |
| greedy playout, this turn and the next | 0.6-2.2 ms |
| greedy playout to the end of the game | 3.5-7.5 ms (54-109 decisions) |

So one iteration costs about 25 µs to sample, about 30 µs per tree level, and 0.3-1 ms to finish the turn and evaluate: roughly 0.4-1.2 ms in all. In the sampler's ~100 ms that is about **100-250 iterations with turn-finishing playouts**. With a policy that skips `observe` (§3), perhaps 2× more. The literature's usual 10,000 iterations [1][4] are far out of reach. At this budget the tree will mostly cover the first few decisions of the current turn, and ISMCTS behaves close to a bandit over the root's moves with greedy completion. That is another reason to expect a modest result, and a reason to run long tournaments under PyPy (1.5-1.7× faster, Milestone 5: PyPy re-measured).

### 2.9 Deterministic seeding

- The bot holds `self._rng = random.Random(seed)`, as `SamplerBot` does. Per decision, draw `worlds = Random(self._rng.getrandbits(64))` and `playouts = Random(self._rng.getrandbits(64))` (as built: `sample` closes over `worlds`, and `playouts` is the search's `rng`, drawing expansions and playouts). Changing a playout setting then doesn't change which worlds are sampled, so A/B comparisons stay paired. `sample_world` already draws each world's `rng_seed` from the rng it is given.
- Never iterate over a `set` of actions or cards: `str` hashing is randomised per process (`PYTHONHASHSEED`), and cards hash by name. Dicts keep insertion order, so children found in the same order iterate the same way. Break ties in selection and at the root by `legal_actions` order, never by hash or `id`.
- Use only counts, never `time`, in anything that decides. Timing belongs in the tournament's `Timing`.
- Test it (§4): the same seed gives the same move and the same root visit counts, under `--jobs 1` and `--jobs 3`.

### 2.10 What to measure (following the milestone-3/4 practice)

1. ms/decision and `advance_legal` calls per decision against iterations, at the `bench` positions and in a few real games. Choose the default iteration count to match the sampler, and record it in "Milestone 5: ISMCTS".
2. 500 games, seeds 0-499, alternating seats, `--jobs 3`, as five foreground runs combined with `tournament.combine`, with 95% Wilson intervals: ismcts vs heuristic, vs greedy and vs sampler, at the defaults. Optionally ismcts vs cheat.
3. Sweeps on seeds **outside 0-499** (as #114 and #115 did), so tuning doesn't fit the measurement seeds: `c` ∈ {0.35, 0.7, 1.0}, `extra_turns` ∈ {0, 1}, ε ∈ {0, 0.1, 0.25}, and one larger iteration budget (4×).
4. Diagnostics: tree size, the depth reached (in turns) by the most visited line, and the share of iterations that reach the opponent's turn. These show whether the search looks past the current turn at all.

## 3. Pitfalls

- **Strategy fusion inside the tree.** Keying a node by anything that includes hidden state (the sampled world, its seed, the opponent's hand) brings back PIMC's clairvoyance. Node identity is the action path, nothing else [1, §IV-E].
- **Selecting from children that aren't legal in this world.** Children accumulate actions from many worlds. Selection must only consider actions in `game.actions(world)` for the current iteration [1, §IV-E], or `apply` will be handed an illegal action. (`advance_legal` skips the legality check, so the result would be silent corruption, not an error. Add an `assert` while developing.)
- **Forgetting availability counts**, or counting them only for selected children. Every existing child legal at that visit gets +1, including the selected one [1, Algorithm 1]. Otherwise actions available only in rare worlds are over-explored [1, §IV-B].
- **Caching a state in a node.** States depend on the determinization. Replay from the root every iteration.
- **The playout policy seeing more than its player knows.** In a determinized world the "hidden" cards are made up, so nothing real leaks. Still, a policy that reads the full state lets the simulated opponent react to our sampled hand and deck order (clairvoyance in the playout). Write a state-level policy that reads only the mover's hand and public fields. That is also faster than `observe` + `GreedyBot.choose`, which spends about 15 µs per decision just building the view.
- **Opponent modelling.** SO-ISMCTS averages the opponent's choices over its possible hands [1]. This is a modelling weakness, not a leak, so accept it. Don't "fix" it by letting the opponent's nodes see the sampled world.
- **Non-locality** [1, §III-B, citing [3]] is about opponents steering play using information we lack. Here the hidden information is chance that no player controls, so it barely applies. Don't add belief modelling for it.
- **Evaluating mid-turn.** Unspent trade and combat are invisible to `evaluate`, so always finish the turn first (§2.5).
- **Random playouts.** Uniform random play ends turns early (`EndTurn` is always legal) and can stall the game, so cap any playout by turns. Deterministic expert playouts are a trap too [4].
- **Unscaled rewards.** Raw `evaluate` (Authority points) or `WIN = 10**9` passed into UCB1 would swamp the exploration term. Rewards must be in [0, 1] [5].
- **Unfair budget comparisons.** Iterations ≠ nodes. Compare at equal ms/decision measured alone, and report both units. ISMCTS uses its full budget at every decision, while alpha-beta's late-turn searches finish under budget. Compare averages over real games, not just turn starts.
- **Pure-Python performance traps.** Use node classes with `__slots__` (no dataclass `__init__` overhead in the hot loop); call `math.log` once per child rather than per comparison where possible; never do the `actions`/legality work twice per level (one `legal_actions`, then `advance_legal`); keep the search iterative, not recursive; call `evaluate` only at the leaf; no logging in the loop without the `isEnabledFor(DEBUG)` guard (design doc, Harness). The tree is small (one node per iteration at most, so a few hundred to a few thousand nodes per decision), so memory doesn't matter.
- **Unanchored tuning.** Tune on seeds outside 0-499 (as #114 and #115 do), then measure once on 0-499.

## 4. Tests

### Generic module (`tests/test_ismcts.py`)

1. **Perfect-information sanity (tic-tac-toe).** An adapter over `tic_tac_toe.node.Board` with `sample = lambda rng: board` (one world), and `value` from a uniformly random playout to the end (tic-tac-toe is tiny). Positions: (a) X to move with two in a row and the third cell free → plays the win; (b) X to move while O threatens a line → blocks. 500-2,000 iterations should be plenty, but assert only the move. The test runs in milliseconds.
2. **A hidden-information toy where PIMC fails and ISMCTS succeeds.** This is Cowling et al.'s Fig. 1 game [1, §IV-E], one player and one coin flip. Hidden state: a fair coin `h` ∈ {0, 1} that `sample` draws. Root actions: `safe` (terminal, reward 0.6) or `gamble`, followed by `guess(0)` / `guess(1)` (reward 1 if the guess equals `h`, else 0). The coin is never observed, so `gamble` is worth 0.5 < 0.6. Clairvoyant determinization (PIMC: solve each world, vote) picks `gamble`, worth 1 in every world. ISMCTS merges both worlds in the `guess` node, values `gamble` at about 0.5, and picks `safe`. Assert ISMCTS returns `safe`, and, in the same test with a ten-line PIMC, that PIMC returns `gamble`. This is the same failure mode as the sampler's clairvoyance over draws, so it tests what matters here.
3. **Availability bookkeeping.** A two-step toy where the opponent (player 1) holds one card, A with probability 0.9 or B with probability 0.1 (drawn by `sample`), and can play only the card it holds. After N iterations, the opponent node's `available` counts for A and B equal the number of iterations in which each was legal there (count them in the test's own `sample`), and visits never exceed availability.
4. **Determinism and budget.** For the same seed, `Result` is equal (move and root visits). Root children's visits sum to `iterations`. A different seed is allowed to differ (don't assert that it does).
5. **Terminal handling.** A root where one action wins at once → chosen. An action that loses at once is never chosen when a non-losing alternative exists (with enough iterations).
6. **One legal action** → returned at once, with `iterations` = 0 in the result (if the shortcut lives in the generic code; it could live in the bot instead). (As built, it lives in the bot.)

### Star Realms bot (`tests/star_realms/test_ismcts.py`)

Reuse the sampler's test positions (`tests/star_realms/test_sampler.py`):

1. **Views only**: `IsmctsBot` satisfies `bots.Bot` and not `BenchmarkBot`. The existing "bots only receive views" tests cover the match runner; add `ismcts` to whatever list of bots they parametrise over.
2. **An obvious winning attack, whatever the hidden cards**: the sampler's `test_it_takes_an_obvious_winning_attack_whatever_the_hidden_cards` position → `AttackPlayer` at a small budget (for example 50 iterations).
3. **Seeded replay**: `play_match((IsmctsBot(seed, iterations=small), GreedyBot()), seed)` twice → the same result and the same transcript.
4. **Legal move**: across a few seeded games at a tiny budget, every chosen move is in `actions`. Cheap, and it catches the "child not legal in this world" bug.
5. **Settings and CLI**: `settings()` names the iterations, `c` and leaf settings. `--bots ismcts greedy` works in the single-game and tournament commands (copy `test_cli.py`'s pattern for the sampler, 1-2 games, tiny `--iterations`).
6. **Value**: `value` of a won state is 1.0 for player 0's win and 0.0 for a loss; a non-terminal value is strictly inside (0, 1); and the value is antisymmetric: swapping perspective gives 1 − v.

## 5. Optional enhancements, ranked by expected value for this game

1. **A better leaf value** (#115, #116). The repo's own evidence says the evaluation, not the search, limits strength: the cheat went from 32.8% to 52.0% against the heuristic from a change to `evaluate` alone (Milestone 5: card values), while more nodes or worlds never helped (Milestone 3). Playout policy and cut-off also matter a lot in card games: in Magic, the choice of playout policy shifted results by about 10 points [4]. Tune ε and `extra_turns` here before anything else.
2. **Cheaper iterations.** At 150-500 iterations, every µs counts. Use a state-level greedy policy without `observe`, and avoid recomputing `legal_actions` in playouts where the policy can choose directly. Run under PyPy for long tournaments. Lowest risk, and it multiplies everything else.
3. **Collapsing orderings of a turn's actions.** #105 found buys and attacks in every order dominate the cheat's unfinished searches (Milestone 4: Search speed on the full deck). The same orderings split an MCTS tree's statistics over many nodes. Options: (a) a canonical order for independent buys and attacks in the adapter (the lever #105 named; compare dominated-move pruning, a significant gain in Magic [4], and heuristic move pruning in the card game Lords of War [20]); (b) transpositions keyed by the searcher's *view* within the current turn: the view is hashable, and two orders reaching the same view are the same information set. Share statistics as in UCT with transpositions [9] or UCD [10]; Choe and Kim searched a DAG for Hearthstone's multi-action turns [16] **(abstract only)**. (a) is simpler and safer.
4. **Progressive bias or priors from the heuristic or `evaluate`.** Add `H(a) / (n + 1)` to UCB for a heuristic value `H`, which fades as visits grow [8][5, §5.2.5]. Or give each new child a few virtual visits at the heuristic's or a one-step `evaluate`'s value. With only hundreds of iterations, good priors matter more than usual. Santos et al. report that a heuristic built into MCTS gave "a significant performance gain" in Hearthstone [15] **(abstract only)**.
5. **Implicit minimax backups.** Keep a minimax backup of `evaluate` alongside the playout means, and select on `(1 − α)·mean + α·v_minimax`. Lanctot et al. found α in [0.15, 0.4] "a safe choice" across Kalah, Breakthrough and Lines of Action, with gains in each [7]. Cheap to add once the leaf calls `evaluate` anyway.
6. **Move grouping.** Choose an action kind first (play, buy, attack, scrap, end turn), then which one. Decomposing compound moves helped in Magic (binary trees, significant at 99%) [4] and in Dou Di Zhu (kicker cards as a second decision) [1, §VII-A]. Move groups were proposed by Childs et al. [9]. Most useful if buying from a full Trade Row is the widest decision.
7. **Tree reuse between decisions in the same turn.** About 2.7 decisions per turn (65 per game over about 24 turns, Milestone 3), and the next one is usually a child of the root. Keep that subtree. Its statistics came from the previous information set, so they are slightly biased after a draw. Cheap and deterministic, but it couples decisions, so test replay carefully.
8. **Closed loop for our own draws** (splitting nodes by the searcher's observation after a chance event). This is more faithful to information sets, but it multiplies branching, which at this budget is the scarce resource. Try only after (3).
9. **Ensembles / root parallelisation.** Fern and Lewis found ensembles help per unit time only when parallel; in a single-core model they "did not observe a significant improvement in performance per unit time" [11]. Tournaments already parallelise over games (`--jobs`), so skip it. (In [1], determinized UCT with 40 trees beat SO-ISMCTS in Phantom (4,4,4) but lost to MO-ISMCTS, and lost to every ISMCTS variant in LOTR:C, at 10,000 iterations each [1, §V-C, §VI-B].)

## 6. References

Sources marked "read" were read in full text (or the cited sections). "Abstract" means only the abstract or a catalogue entry was seen.

1. P. I. Cowling, E. J. Powley, D. Whitehouse, "Information Set Monte Carlo Tree Search", *IEEE Trans. Computational Intelligence and AI in Games* 4(2):120-143, 2012. doi:10.1109/TCIAIG.2012.2200894. https://eprints.whiterose.ac.uk/75048/ (read). Used for: SO-ISMCTS, +POM and MO-ISMCTS; availability counts; `c = 0.7`; equivalence of the variants for fully observable moves; results in LOTR:C (ISMCTS beats determinized UCT for the Dark player by a large margin, and for the Light player by about 4.4% (MO-ISMCTS)), Phantom (4,4,4) (MO-ISMCTS > determinized UCT > SO-ISMCTS), and Dou Di Zhu (ISMCTS 42.3% against determinized UCT's 43.6% overall, better where hidden information matters, worse where many moves are unique to a determinization).
2. J. Long, N. R. Sturtevant, M. Buro, T. Furtak, "Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search", AAAI 2010, pp. 134-140. https://ojs.aaai.org/index.php/AAAI/article/view/7562 (read). Used for: leaf correlation, bias, disambiguation factor; PIMC is weakest with low leaf correlation, and improves sharply as disambiguation increases; Skat and Hearts measured at correlation 0.8-1.0, disambiguation about 0.6.
3. I. Frank, D. Basin, "Search in games with incomplete information: a case study using Bridge card play", *Artificial Intelligence* 100, 1998. Cited through [1] for strategy fusion and non-locality **(not read: unverified)**.
4. P. I. Cowling, C. D. Ward, E. J. Powley, "Ensemble Determinization in Monte Carlo Tree Search for the Imperfect Information Card Game Magic: The Gathering", *IEEE TCIAIG* 4(4), 2012. doi:10.1109/TCIAIG.2012.2204883. https://eprints.whiterose.ac.uk/75050/ (read). Used for: binary decision trees for compound moves; dominated-move pruning; 40 determinizations × 250 simulations best at 10,000; stochastic reduced-rules playouts beat deterministic expert playouts by about 10 points; the best players reach 51.2% against the expert rules player.
5. C. B. Browne et al., "A Survey of Monte Carlo Tree Search Methods", *IEEE TCIAIG* 4(1):1-43, 2012. https://repository.essex.ac.uk/4117/ (read the cited sections). Used for: final move selection criteria (§3.1), `Cp = 1/√2` for rewards in [0, 1] (§3.3, UCT), move groups (§5.2.3), transpositions (§5.2.4), progressive bias formula (§5.2.5), progressive widening (§5.5.1), root parallelisation (§6.3.2).
6. L. Kocsis, C. Szepesvári, "Bandit based Monte-Carlo Planning", ECML 2006. Cited through [5] **(not read: unverified)**.
7. M. Lanctot, M. H. M. Winands, T. Pepels, N. R. Sturtevant, "Monte Carlo Tree Search with Heuristic Evaluations using Implicit Minimax Backups", IEEE CIG 2014. https://arxiv.org/abs/1406.0486 (read). Used for: the `(1 − α)·mean + α·v_IM` selection value, α ∈ [0.15, 0.4] "a safe choice", fixed-depth early playout terminations (fet4 best in Kalah, fet20 in Breakthrough), ε-greedy playouts, evaluation in the reward's range.
8. G. M. J.-B. Chaslot, M. H. M. Winands, H. J. van den Herik, J. W. H. M. Uiterwijk, B. Bouzy, "Progressive Strategies for Monte-Carlo Tree Search", *New Mathematics and Natural Computation* 4(3):343-357, 2008. https://cris.maastrichtuniversity.nl/en/publications/progressive-strategies-for-monte-carlo-tree-search (abstract; formula from [5]).
9. B. E. Childs, J. H. Brodeur, L. Kocsis, "Transpositions and Move Groups in Monte Carlo Tree Search", IEEE CIG 2008, pp. 389-395. https://eprints.sztaki.hu/5549/ (catalogue entry; content from [5]).
10. A. Saffidine, T. Cazenave, J. Méhat, "UCD: Upper Confidence bound for rooted Directed acyclic graphs", *Knowledge-Based Systems* 34, 2012. https://www.lamsade.dauphine.fr/~cazenave/papers/kbs.pdf (abstract).
11. A. Fern, P. Lewis, "Ensemble Monte-Carlo Planning: An Empirical Study", ICAPS 2011, pp. 58-65. https://ojs.aaai.org/index.php/ICAPS/article/view/13458 (read abstract and summary).
12. D. Perez Liebana, J. Dieskau, M. Hünermund, S. Mostaghim, S. Lucas, "Open Loop Search for General Video Game Playing", GECCO 2015. https://repository.essex.ac.uk/14734/ (abstract only; the description of open-loop MCTS as storing no states and replaying actions with the forward model comes from a later paper's summary: **unverified against the paper**).
13. J. Kowalski, M. Mika, W. Pawlik, J. Sutowicz, M. Szykuła, M. H. M. Winands, "Split Moves for Monte-Carlo Tree Search", AAAI 2022. https://arxiv.org/abs/2112.07761 (abstract).
14. S. Zhang, M. Buro, "Improving Hearthstone AI by Learning High-Level Rollout Policies and Bucketing Chance Node Events", IEEE CIG 2017, pp. 309-316. doi:10.1109/CIG.2017.8080452. Thesis: https://era.library.ualberta.ca/files/cv979v349b (abstract): determinized MCTS, chance outcomes bucketed, learned high-level playout policies.
15. A. Santos, P. A. Santos, F. S. Melo, "Monte Carlo Tree Search Experiments in Hearthstone", IEEE CIG 2017, pp. 272-279. https://labjogos.tecnico.ulisboa.pt/en/thesis/monte-carlo-tree-search-experiments-in-hearthstone (abstract; **co-authors unverified**): a heuristic built into MCTS beat vanilla MCTS and the then-best Hearthstone AI.
16. J. S. B. Choe, J.-K. Kim, "Enhancing Monte Carlo Tree Search for Playing Hearthstone", IEEE CoG 2019. https://pure.korea.ac.kr/en/publications/enhancing-monte-carlo-tree-search-for-playing-hearthstone/ (abstract): state abstraction, search over a DAG, sparse sampling for chance and hidden information.
17. M. Świechowski, T. Tajmajer, A. Janusz, "Improving Hearthstone AI by Combining MCTS and Supervised Learning Algorithms", IEEE CIG 2018, pp. 445-452. https://arxiv.org/abs/1808.04794 (abstract).
18. R. Tollisen, J. V. Jansen, M. Goodwin, et al., "AIs for Dominion Using Monte-Carlo Tree Search", IEA/AIE 2015, LNCS 9101 (abstract via a catalogue listing; **fourth author and venue unverified**): UCB and UCT agents; the best wins 67% against a good finite-state bot. Thesis: J. V. Jansen, R. Tollisen, "An AI for Dominion based on Monte-Carlo methods", University of Agder, 2014. https://uia.brage.unit.no/uia-xmlui/handle/11250/221382 (abstract; 87.5% against moderately experienced humans).
19. D. Whitehouse, P. I. Cowling, E. J. Powley, J. Rollason, "Integrating Monte Carlo Tree Search with Knowledge-Based Methods to Create Engaging Play in a Commercial Mobile Game", AIIDE 2013. https://ojs.aaai.org/index.php/AIIDE/article/view/12679 (abstract): ISMCTS in a commercial Spades game, objectively stronger than the knowledge-based AI it replaced.
20. N. Sephton, P. I. Cowling, E. J. Powley, N. H. Slaven, "Heuristic Move Pruning in Monte Carlo Tree Search for the Strategic Card Game Lords of War", IEEE CIG 2014. https://pure.york.ac.uk/portal/en/publications/heuristic-move-pruning-in-monte-carlo-tree-search-for-the-strateg/ (title only: **content unverified**).

**Not found or unverified:**

- No published MCTS or ISMCTS bot or paper for **Star Realms** or **Ascension** was found.
- **Fynbo and Nellemann's Dominion work** couldn't be located. One thesis's contents list names a "Fynbo and Nellemann neural network to evaluate cards", so it may be about card evaluation rather than MCTS (**unverified**).
- "Bandits all the way down" (Powley, Whitehouse, Cowling, CIG 2013, UCB1 as a playout policy) wasn't found; it isn't relied on here.
