# Learned play and search: concepts, and where each is strong

A conceptual reference for deciding how to build a strong Star Realms bot. It pins down what "search", "learning" and "knowledge" mean, what properties of a game decide which of them matters most, and what the record of other games says. Section 4 applies it to Star Realms and to the open proposals: #127 (a compiled engine) and #128 (a learned evaluation with a turn planner).

It is not a literature survey. Each claim has a numbered source in [References](#references), with a label:

- **[read]**: the primary source, or its results section, was read.
- **[abstract]**: only the abstract, an official summary or a listing was seen.
- **[unverified]**: from a secondary source or an earlier note, not checked against the original. Treat it as a lead.

Researched 2026-10-21.

## 1. Concepts

### 1.1 Three ways to choose a move

Every game-playing program combines three things in some ratio:

- **Search**: look ahead. Try moves on a model of the game, see where they lead, and pick the move whose consequences look best. It spends computation *at decision time*.
- **Learning**: fit a function (a value, a policy, or both) from experience, usually games the program plays against itself. It spends computation *before* the game, in training, and makes each decision cheap.
- **Knowledge**: hand-written rules or an evaluation function written by a person. It spends *human* effort. Our `HeuristicBot` and `evaluate` are knowledge.

Search almost always needs something to score the positions where it stops (a leaf evaluation). That scorer is either knowledge or learned. So the real question is rarely "search or learning" but **how much of the strength comes from deep lookahead and how much from a good judgement at the leaves**.

### 1.2 Search methods

| Method | What it does | Fits | Source |
|---|---|---|---|
| Minimax / alpha-beta | Full-width lookahead to a depth, both players assumed to play best; alpha-beta skips lines that can't change the result. | Two-player, perfect information, deterministic. Chess, checkers. | Deep Blue [10] [read] |
| Expectimax | Minimax with chance nodes that average over dice or draws. | Small chance branching. Shallow search in backgammon. | TD-Gammon's 2-ply search [5] [read] |
| MCTS / UCT | Grows a tree selectively: a bandit rule (UCB) picks which branch to try next, random or guided playouts score the leaves. Needs no evaluation function in its pure form. | Large branching factor, weak hand evaluation. Go before 2016. | Browne et al. survey [17] [read]; Go [12] [abstract] |
| PIMC (determinisation) | Sample the hidden information to get perfect-information worlds, search each, combine the answers. Our sampler. | Hidden information that is mostly chance and resolves soon. Bridge, Skat. | Long et al. [18] [read]; GIB [13] [abstract] |
| ISMCTS | One MCTS tree over information sets; a fresh world is sampled each iteration. Our `IsmctsBot`. | Like PIMC, plus some protection against "strategy fusion". | Cowling et al. [19] [read] |
| Whole-turn / evolutionary planning | Treat a whole multi-action turn as one plan; search or evolve plans, score the end-of-turn state. | Turns made of many small actions. Hero Academy, Hearthstone. | Justesen et al. [23] [abstract]; Baier & Cowling [24] [read] |
| Re-solving / depth-limited game-theoretic search | At decision time, solve the current subgame for an equilibrium, with a learned or precomputed value at the depth limit. | Poker: hidden information that is the opponent's private *choice*. | DeepStack [26], Libratus [27], Pluribus [28] [abstract] |

### 1.3 Learning methods

| Method | What is learned, from what | Source |
|---|---|---|
| Temporal-difference (TD) learning | A value function V(state), each state's value nudged towards the next state's. Self-play games only. | TD-Gammon [5] [read] |
| Monte Carlo regression | V(state) fitted to the final result of the games it appeared in. The simplest form of the same idea. | (definition) |
| Policy gradient / PPO | A policy π(action given state) pushed towards actions that led to wins. PPO is a stable, widely used variant. | Schulman et al. [31] [abstract]; space-deck-builder [36] [read] |
| Search-guided self-play (AlphaZero, expert iteration) | Policy and value trained to imitate what a search *using them* decided. Search improves the net; the net improves the search. | AlphaGo Zero [7] [read]; Anthony et al. [8] [abstract] |
| Fictitious play, CFR family | Strategies that approach a Nash equilibrium by repeatedly responding to the average of past play (fictitious play) or minimising regret at each information set (CFR). Needed when the opponent's private choices matter. | Zinkevich et al. [25] [abstract]; NFSP [32] [abstract] |
| Evolution of hand-written strategies | Parameters of a rule-based strategy (e.g. a buy list) evolved by playing variants against each other. | Provincial [34] [read] |

A **value function** says how good a position is. A **policy** says which move to make. A policy alone needs no search at all; a value function needs at least a one-move lookahead ("try each move, score the result"), which is the smallest possible search.

### 1.4 The knowledge-search trade-off

The classic framing is a trade-off curve: the same playing strength can come from little knowledge and deep search, or rich knowledge and shallow search.

- Berliner and Ebeling, from the Hitech chess machine: "The further a process can look ahead, the less detailed knowledge it needs" [2] [abstract].
- Junghanns and Schaeffer revisit this for chess, Othello and checkers: "Perfect knowledge about a domain renders search unnecessary and, likewise, exhaustive search obviates heuristic knowledge." They argue that chess's apparently linear gain from each extra ply of depth is a property of chess and chess programs, not a general law [1] [abstract].
- Sutton's "Bitter Lesson" (2019): over 70 years, "general methods that leverage computation are ultimately the most effective, and by a large margin"; the two methods "that seem to scale arbitrarily" are **search and learning**; hand-built knowledge "always helps in the short term" but "plateaus and even inhibits further progress" [3] [read].

The Bitter Lesson does not say "learning beats search". It says both scale with compute and hand-written knowledge doesn't. For this project, the practical reading is that `evaluate`'s hand-chosen card values are the part least likely to keep improving.

### 1.5 Three ideas that connect search and learning

**Evaluation is compressed search.** A good evaluation function predicts what a deep search (or the rest of the game) would conclude, without doing it. Stockfish's NNUE network was trained "on evaluations of millions of positions at moderate search depth": a learned shortcut for search. It added over 80 Elo to Stockfish when it was introduced in 2020 [4] [read]. TD learning does the same against the game's actual outcomes [5] [read].

**Search is policy improvement.** Given any policy and value, a search that uses them usually picks better moves than the policy alone. AlphaGo Zero's authors describe MCTS as "a powerful policy improvement operator" and train the network to match the search's choices [7] [read]. Expert iteration frames the same loop as a fast learner (the net) and a slow expert (the search) teaching each other; on Hex it beat REINFORCE, a plain policy-gradient method [8] [abstract].

**Amortisation.** Learning moves the cost of thinking from play time to training time. A trained policy decides in one forward pass. Search pays its cost at every decision. With enough training, the policy can match what search found, and then search on top only adds what the net still gets wrong. How much that is depends on the game (section 3.2).

## 2. What decides which approach wins

The same techniques rank differently from game to game. These are the properties that matter, with the evidence.

| Property | Favours | Why | Evidence |
|---|---|---|---|
| **Large branching factor** | Selective search (MCTS with priors) or learning; against full-width alpha-beta | Full-width search can't go deep. A learned policy can prune to a few candidates. | Go: alpha-beta stalled, MCTS reached master strength [12] [abstract]; AlphaZero searched 80 thousand positions/s in chess against Stockfish's 70 million and still won, by being selective [6] [read]. Backgammon's "several hundred" moves per ply make deep search impractical [5] [read]. |
| **Long strategic horizon** (decisions pay off many moves later, beyond any search depth) | A good evaluation (learned or expert) | Search can only see consequences inside its horizon. Beyond it, the leaf evaluation is the whole judgement. | Dominion: buys are "by far the more challenging" part, card play simple heuristics [34] [read]. TD-Gammon: strength from the learned evaluation, with 1-3 ply search [5] [read]. |
| **Tactical sharpness** (forcing lines, a single move loses) | Deep search | An evaluation can't reliably see a forced sequence; search finds it by trying. | Chess: Deep Blue's emphasis on search extensions and 100-200 million positions/s [10] [read]; Stockfish still uses alpha-beta, with a learned evaluation [4] [read]. |
| **Chance** (dice, shuffles) | Shallow search plus a strong evaluation | Chance nodes multiply the tree, and the value of a position is smooth. | TD-Gammon: dice make equity "a smooth function" and errors on similar moves correlate and cancel, so 2-ply search suffices [5] [read]. |
| **Hidden information that is chance** (cards nobody chose) | Sampling (PIMC) works well; little gain from fancier methods | Sampling worlds is roughly right when the hidden cards don't depend on the opponent's strategy. | PIMC strong in bridge and Skat [13] [abstract], [14] [abstract]; works when leaf correlation is high [18] [read]. |
| **Hidden information that is the opponent's choice** (bluffs, private plans) | Game-theoretic methods: CFR, re-solving, equilibrium-seeking RL | Sampling assumes the opponent's hand is random; a bluffing opponent exploits that. | Poker: CFR-based blueprints plus decision-time re-solving [26] [27] [28]; Stratego: equilibrium RL without search [30] [abstract]. |
| **Cooperation with hidden information** | A learned policy, with search on top | Partners must agree on conventions; search refines a shared policy. | Hanabi: search on an RL policy reached 24.61/25 against a previous best of 24.08 [15] [abstract]. |
| **Multi-action turns** | Whole-turn planning; budget on the current turn | A turn of many small actions has a huge number of orderings; a tree over single actions spends its budget on them. | Hero Academy: evolving whole-turn plans beat MCTS as actions per turn grew [23] [abstract]; with long turns the best horizon was the current turn only [24] [read]. |
| **Simultaneous moves, real time, huge action spaces** | Learned policies, no tree search | No usable game tree at decision time. | StarCraft II [29] [abstract]; Dota 2 [33] [abstract]. |
| **Cheap perfect simulator** | Both: search can use it at play time, learning can generate millions of games | Without one, search must use a model (MuZero-style) or not search. | space-deck-builder: 5.88M engine decisions/s on a compiled engine, 800k training games per seed [36] [read]. |
| **Small compute per move** | Learning (amortised) | Search quality falls with budget; a policy costs the same at any budget. | AlphaZero scaled better with thinking time than Stockfish or Elmo [6] [read]; every search result in 3.2 is quoted at a stated budget. |

Two cross-cutting points:

- **Search needs a leaf evaluation.** When the horizon is long, search depth buys little unless the leaf is good. Better leaves raise every search method at once.
- **Bias at the root matters more than depth when moves are many and similar.** Selective search lives or dies by its move priors. A learned policy is the most general source of priors [7] [read].

## 3. The record

### 3.1 Case studies

"Role" says where the strength comes from. Labels refer to the source for that row.

| Game | Strongest approach | Role of search vs learning | Source |
|---|---|---|---|
| Checkers | Chinook: alpha-beta, opening book, endgame databases for all positions with ≤ 10 pieces (3.9 × 10^13). Checkers was later solved (a draw). | Search plus precomputed exact knowledge. No learning. | [9] [read] |
| Chess (1997) | Deep Blue: special hardware, 100-200 million positions/s, a hand-tuned evaluation of over 8,000 features | Search-dominated, with heavy hand knowledge. | [10] [read] |
| Chess (now) | Stockfish: alpha-beta with a learned NNUE evaluation (+80 Elo at introduction); AlphaZero/Leela: MCTS with a learned policy and value | Both families combine deep search with a learned evaluation. Chess's tactics keep search essential. | [4] [read], [6] [read] |
| Go | AlphaGo Zero / AlphaZero / KataGo: MCTS guided by a policy-value net from self-play | Learning gives judgement; search adds about 2,100 Elo on top (3.2). MCTS alone reached master level, not top professional. KataGo cut training compute about 50x. | [7] [read], [12] [abstract], [11] [abstract] |
| Backgammon | TD-Gammon: a neural value learned by TD self-play, 2-ply (later 3-ply) search | Learning-dominated; shallow search fixes some endgame errors. Near the best humans. | [5] [read] |
| Poker (heads-up no-limit) | DeepStack, Libratus: equilibrium strategy plus re-solving at decision time | Both: a precomputed or learned strategy, refined by game-theoretic search every hand. Libratus beat four top professionals over 120,000 hands. | [26] [abstract], [27] [unverified] |
| Poker (6-player) | Pluribus: MCCFR self-play blueprint (8 days, 64 cores, under $150 of cloud) plus depth-limited search in play (2 CPUs) | Search on top of a self-play blueprint. Won about 5 bb/100 against elite pros. | [28] [abstract] |
| Bridge (card play) | GIB: Monte Carlo sampling of deals, each solved by double-dummy search (PIMC) | Search over sampled worlds. "Approximately expert calibre" (2001). | [13] [abstract] |
| Skat | PIMC search with a state evaluation learned from human games, plus inference about hidden cards | Search plus learned evaluation; first expert-level Skat program. | [14] [abstract] |
| Hanabi | RL policy with single- or multi-agent search on top | Learned policy; search adds 0.53 points to the state of the art (24.61/25). | [15] [abstract] |
| StarCraft II | AlphaStar: imitation then league self-play RL | Learned policy; the official summary describes no search. Grandmaster, top 0.2% on Battle.net. | [29] [abstract] |
| Dota 2 | OpenAI Five: self-play RL at scale (10 months) | Learned policy; the abstract describes no search. Beat the world champions. | [33] [abstract] |
| Stratego | DeepNash: equilibrium-seeking RL (R-NaD), "without search" | Learned policy only. Top 3 on Gravon. | [30] [abstract] |
| Diplomacy | Cicero: a strategic planning module plus a language model | Planning over predicted moves of other players; top 10% of players with more than one game (40 games). | [35] [abstract] |
| Magic: The Gathering | Ensemble determinisation MCTS | Search; the best players about 51% against an expert rule-based player. | [20] [read, for docs/ismcts.md] |
| Hearthstone | Competition entries (2018-2020): MCTS, rolling-horizon evolution, pruned BFS, dynamic lookahead, each with a state-scoring function; 2019 winner used ISMCTS with state abstraction. In 2020 shallow lookahead bots placed 1st and 2nd in the premade-deck track (2-Step Lookahead 72.3%, Dynamic Lookahead 70.9%, MCTS 3rd at 68.7%) and Dynamic Lookahead 1st and 2nd in the user-deck track | Shallow search over the turn with a hand-written scorer; MCTS competitive but not ahead. | [21] [read, the paper's summary], [22] [abstract], [38] [read] |
| Race for the Galaxy | Keldon Jones's AI: neural networks trained by TD learning from self-play | Learning-dominated. The commercial app's AI uses TD learning and self-play. | [37] [unverified] |
| Dominion | Provincial: buy lists evolved by self-play, simple card-play heuristics. Gerigk & Engels: deep RL, 73% (plus 8.5% draws) against Big Money, but 42% against a Double Witch bot | Strategy (buys) is the hard part. Learned agents beat simple bots but not always strong scripted ones. Earlier MCTS reached 68.5% against Witch bots on one kingdom but never used action-card combinations. | [34] [read], [16] [read] |
| Star Realms (another project) | space-deck-builder: PPO self-play policy, 92.5% against its heuristic at temperature 0, no search | Learning-dominated. Turn-bounded search on top: +44 Elo. | [36] [read] |
| Star Realms (this repo) | Hand-written heuristic; no legitimate search bot beats it | See section 4. | [R1] |

### 3.2 What search adds on top of a learned net, when measured

| System | Without search | With search | Source |
|---|---|---|---|
| AlphaGo Zero (Go) | Raw network, most probable move: Elo 3,055 | Full system, 5 s/move on 4 TPUs: Elo 5,185 | [7] [read] |
| AlphaZero (chess) | n/a | 80 thousand positions/s, beat Stockfish at 70 million; scaled with thinking time better than Stockfish | [6] [read] |
| Hanabi | Previous best 24.08/25 | 24.61/25 with search on an RL policy | [15] [abstract] |
| Legends of Code and Magic (card game) | Imitation-learned policy and value: 26.8% against the champion ByteRL | Sampling opponent decks ("unsound" search, PIMC-like) on top: 51.35% over 10,000 pre-registered games | [39] [abstract] |
| Pluribus (poker) | Blueprint from self-play | Real-time depth-limited search on top, used in all matches | [28] [abstract] |
| space-deck-builder (Star Realms) | PPO policy, greedy: 50% against itself | Top-8 candidates, sequential halving, rollouts finishing the turn, critic at the leaf: 56.4% (+44 Elo) over 2,000 paired deals. 16 rollouts +33 Elo, 64 +48, 256 +47: no gain past 64. One round of expert iteration: +19 Elo to the policy. | [36] [read] |

The pattern: **in sharp, deep games (Go, chess) search on a learned net adds a great deal. In a stochastic deck-builder with a learned policy, it adds a little and saturates quickly.** That matches section 2: where the payoff of a decision lies far beyond what any search can reach, the leaf judgement carries the strength.

## 4. Synthesis

### 4.1 Decision guide

| If the game has... | ...expect | Spend effort on |
|---|---|---|
| Short horizon, sharp tactics, perfect information | Deep search wins; evaluation quality matters less | Search speed, pruning, move ordering |
| Long horizon, smooth value, chance | A good evaluation wins; search of 1-3 plies is enough | The evaluation (learned by self-play); a shallow search on top |
| Hidden information that is mostly chance | PIMC is about as good as anything; ISMCTS adds little | The per-world search and the leaf, not the sampling method |
| Hidden information that is opponents' choices | Equilibrium methods needed | CFR / re-solving / equilibrium RL |
| Multi-action turns | Trees over single actions waste budget on orderings | Whole-turn planning, canonical orderings, current-turn budget |
| Huge action space, real time | Search impractical | A learned policy |
| A cheap simulator and modest hardware | Both are open; learning needs millions of games | A fast engine first |

Most games mix rows. The question to ask is: **where in this game do decisions get decided, inside the search horizon or beyond it?**

### 4.2 Star Realms against the guide

| Property | Star Realms | Points to |
|---|---|---|
| Strategic horizon | Long. A buy pays off as the deck cycles, over roughly 5-15 turns; a 2-turn search can't see most of it. | The evaluation |
| Tactics | Shallow, mostly within a turn: lethal checks, splitting combat between bases and Authority, play order for draws and ally abilities, scrap choices. | A small search: one turn |
| Chance | Large: shuffles, the Trade Row. | Shallow search, smooth value |
| Hidden information | Mostly chance: our deck order, the opponent's fresh hand. The opponent's private choices are few. | PIMC is enough; ISMCTS not expected to beat it |
| Turn structure | Many micro-actions per turn, with many orderings reaching the same end of turn. | Whole-turn planning |
| Simulator | Perfect and cheap, but slow in Python (about 45k alpha-beta nodes/s on CPython [R1]; #127). | Compiled engine for learning at scale |

By the guide, Star Realms sits close to backgammon and Dominion, not chess: **strength should come mostly from the evaluation, with a short tactical search inside the turn**.

### 4.3 What this repo's results say

From docs/star-realms-results.md (Milestone 5) [R1]:

- **The cheat, alpha-beta on the true state including deck order, only ties the hand-written heuristic**: 52.0% (47.6-56.3%) on seeds 0-499, and 43-44% on two other seed ranges. Perfect information plus a 2-turn search buys about as much as hand-written card values without search.
- **The legitimate search bots are far below the heuristic and level with each other**: ISMCTS 34.6%, the sampler (PIMC) 33.4%; ISMCTS against the sampler 48.6%. That is what section 2 predicts for hidden information that is mostly chance.
- **4x ISMCTS iterations gained nothing** (+1 on 61 discordant seeds). Leaf settings moved it far more (up to 24 games of 200).
- **4x nodes lifted the cheat from 43.2% to 51.4%** on seeds 2000-2499 (+41 on 81 discordant, p < 0.001). Budget helps alpha-beta when its 2-turn search otherwise fails to finish. That is consistent with the micro-action tree spending nodes on orderings (Milestone 4: search speed on the full deck), though no ablation isolated it.
- **Tuning the linear weights and greedy rollouts didn't beat the hand-chosen card values.** That suggests the linear form, more than its weights, is the limit; it hasn't been tested directly.

Read together with section 3:

1. **The flat budget curve is the expected shape.** A check for one suspected bias in ISMCTS's leaves (scoring at different turn boundaries; #128) found no measurable effect at the default budget. space-deck-builder's search on a learned net saturated at 64 rollouts [36] [read]; ours saturates with a weaker leaf. More search with the same leaf won't close a 15-point gap.
2. **The cheat tying the heuristic is the clearest signal.** Even with no hidden information, 2 turns of lookahead on a hand-valued leaf can't see what makes a buy good. The heuristic encodes that judgement directly. A better leaf is the lever.
3. **The only published Star Realms evidence points to learning.** A PPO policy without search beats its heuristic 92.5%, and search adds +44 Elo [36] [read]. Caveats: one unreviewed project, a different heuristic of unknown strength relative to ours, and about 800k self-play games per seed on a GPU with a compiled engine.
4. **Learned agents don't automatically beat strong scripts.** Gerigk & Engels' Dominion agent beat Big Money 73% but lost to a Double Witch bot (42%) unless that bot was in training [16] [read]. Our heuristic should be in the training opponents, and the bar stays "clearly above 50% against the heuristic".
5. **Multi-action turns call for a turn-level search.** Evidence from Hero Academy says long turns are best searched within the turn [24] [read], with whole-turn plans scored by an evaluation [23] [abstract]. In Star Realms that means listing or searching distinct end-of-turn states and scoring each with the value function.

### 4.4 Implications for #127 and #128

These are readings of the evidence, for the reviews of those issues to challenge, not decisions.

- **#128 (learned evaluation plus turn planner) matches the guide.** The game's long horizon, chance-dominated hidden information and multi-action turns all point to "a strong evaluation, a short search within the turn, simple sampling for the hidden cards". The cheap test it lists first, a whole-turn planner on the *current* evaluation, separates "the tree wastes budget" from "the leaf is too weak"; the cheat's gain from 4x nodes suggests both are true.
- **A learned value is the general replacement for hand-chosen card values.** TD learning or Monte Carlo regression on self-play games [5] [read] is the simplest form; a policy trained by PPO is what worked for space-deck-builder [36] [read]; search-guided training (expert iteration) is the next step if search on top helps [8] [abstract].
- **#127 (compiled engine) is a precondition for learning at the published scale.** space-deck-builder needed about 800k games per seed and a compiled engine [36] [read]. At our current speed (about 45k alpha-beta nodes/s and 0.2 ms per ISMCTS iteration on CPython, per #127), data generation for a learned value would be the bottleneck. Search on top saturates quickly, so the engine matters more for training than for play.
- **Sampling stays adequate for hidden information.** Nothing in the record suggests Star Realms needs equilibrium methods; PIMC or ISMCTS at whole-turn granularity, with the learned value at the leaves, is the expected end state [18] [read], [19] [read].

## References

Labels: [read], [abstract], [unverified], as defined at the top. "Read for docs/ismcts.md" means read in this repo's earlier ISMCTS research and not re-read here.

1. A. Junghanns, J. Schaeffer, "Search Versus Knowledge in Game-Playing Programs Revisited", IJCAI 1997, pp. 692-697. https://mlanthology.org/ijcai/1997/junghanns1997ijcai-search [abstract]
2. H. Berliner, C. Ebeling, "Pattern Knowledge and Search: The SUPREM Architecture", *Artificial Intelligence* 38(2):161-198, 1989. https://www.chessprogramming.org/HiTech [abstract, via secondary listings]
3. R. Sutton, "The Bitter Lesson", 13 March 2019. http://www.incompleteideas.net/IncIdeas/BitterLesson.html [read]
4. Stockfish team, "Introducing NNUE Evaluation", 7 August 2020. https://stockfishchess.org/blog/2020/introducing-nnue-evaluation/ [read]
5. G. Tesauro, "Temporal Difference Learning and TD-Gammon", *Communications of the ACM* 38(3), 1995. https://www.bkgm.com/articles/tesauro/tdl.html [read]
6. D. Silver et al., "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm", arXiv:1712.01815, 2017 (published in *Science* 362, 2018). https://arxiv.org/abs/1712.01815 [read: the preprint's evaluation section]
7. D. Silver et al., "Mastering the game of Go without human knowledge", *Nature* 550, 2017. Author manuscript: https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf [read: the evaluation and MCTS sections]
8. T. Anthony, Z. Tian, D. Barber, "Thinking Fast and Slow with Deep Learning and Tree Search" (expert iteration), NeurIPS 2017. https://arxiv.org/abs/1705.08439 [abstract]
9. J. Schaeffer et al., "Checkers Is Solved", *Science* 317:1518-1522, 2007. https://www.cs.mcgill.ca/~dprecup/courses/AI/Materials/checkers_is_solved.pdf [read: introduction and method]
10. M. Campbell, A. J. Hoane Jr., F.-h. Hsu, "Deep Blue", *Artificial Intelligence* 134:57-83, 2002. https://www.cefns.nau.edu/~edo/Classes/CS470-570_WWW/Readings/Campbell2002.pdf [read: abstract and system sections]
11. D. J. Wu, "Accelerating Self-Play Learning in Go" (KataGo), arXiv:1902.10565, 2019. https://arxiv.org/abs/1902.10565 [abstract]
12. S. Gelly et al., "The Grand Challenge of Computer Go: Monte Carlo Tree Search and Extensions", *Communications of the ACM* 55(3):106-113, 2012. https://research.google/pubs/the-grand-challenge-of-computer-go-monte-carlo-tree-search-and-extensions/ [abstract]
13. M. L. Ginsberg, "GIB: Imperfect Information in a Computationally Challenging Game", *JAIR* 14:303-358, 2001. https://arxiv.org/abs/1106.0669 [abstract]
14. M. Buro, J. R. Long, T. Furtak, N. Sturtevant, "Improving State Evaluation, Inference, and Search in Trick-Based Card Games", IJCAI 2009, pp. 1407-1413. https://www.ijcai.org/Abstract/09/236 [abstract]
15. A. Lerer, H. Hu, J. Foerster, N. Brown, "Improving Policies via Search in Cooperative Partially Observable Games", AAAI 2020. https://arxiv.org/abs/1912.02318 [abstract]
16. J. Gerigk, S. Engels, "Playing Various Strategies in Dominion with Deep Reinforcement Learning", AIIDE 2023. https://ojs.aaai.org/index.php/AIIDE/article/view/27518 [read: related work and results]
17. C. B. Browne et al., "A Survey of Monte Carlo Tree Search Methods", *IEEE TCIAIG* 4(1):1-43, 2012. https://repository.essex.ac.uk/4117/ [read for docs/ismcts.md]
18. J. Long, N. R. Sturtevant, M. Buro, T. Furtak, "Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search", AAAI 2010. https://ojs.aaai.org/index.php/AAAI/article/view/7562 [read for docs/ismcts.md]
19. P. I. Cowling, E. J. Powley, D. Whitehouse, "Information Set Monte Carlo Tree Search", *IEEE TCIAIG* 4(2):120-143, 2012. https://eprints.whiterose.ac.uk/75048/ [read for docs/ismcts.md]
20. P. I. Cowling, C. D. Ward, E. J. Powley, "Ensemble Determinization in Monte Carlo Tree Search for the Imperfect Information Card Game Magic: The Gathering", *IEEE TCIAIG* 4(4), 2012. https://eprints.whiterose.ac.uk/75050/ [read for docs/ismcts.md]
21. J. Kowalski et al., "Introducing Tales of Tribute AI Competition", arXiv:2305.08234, 2023-2024 (its section II-A summarises the Hearthstone AI Competition, 2018-2020). https://arxiv.org/abs/2305.08234 [read: that summary only]
22. J. S. B. Choe, J.-K. Kim, "Enhancing Monte Carlo Tree Search for Playing Hearthstone", IEEE CoG 2019. https://pure.korea.ac.kr/en/publications/enhancing-monte-carlo-tree-search-for-playing-hearthstone/ [abstract]
23. N. Justesen, T. Mahlmann, S. Risi, J. Togelius, "Playing Multi-Action Adversarial Games: Online Evolutionary Planning versus Tree Search", *IEEE TCIAIG*, 2017, pp. 281-291. https://pure.itu.dk/en/publications/playing-multi-action-adversarial-games-online-evolutionary-planni/ [abstract]
24. H. Baier, P. I. Cowling, "Evolutionary MCTS with Flexible Search Horizon", AIIDE 2018. https://ojs.aaai.org/index.php/AIIDE/article/view/13023 [read: experiments and table 1]
25. M. Zinkevich, M. Johanson, M. Bowling, C. Piccione, "Regret Minimization in Games with Incomplete Information" (CFR), NIPS 2007. https://proceedings.neurips.cc/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html [abstract]
26. M. Moravčík et al., "DeepStack: Expert-Level Artificial Intelligence in Heads-Up No-Limit Poker", *Science* 356, 2017. https://arxiv.org/abs/1701.01724 [abstract]
27. N. Brown, T. Sandholm, "Superhuman AI for heads-up no-limit poker: Libratus beats top professionals", *Science* 359, 2018. https://www.science.org/doi/10.1126/science.aao1733 [unverified: abstract text seen only in mirrors; module description from the IJCAI 2017 demo paper, https://www.ijcai.org/proceedings/2017/0772.pdf]
28. Meta AI, "Facebook, Carnegie Mellon build first AI that beats pros in 6-player poker" (Pluribus), 2019; paper: N. Brown, T. Sandholm, *Science* 365, 2019. https://ai.meta.com/blog/pluribus-first-ai-to-beat-pros-in-6-player-poker/ [abstract: official blog]
29. Google DeepMind, "AlphaStar: Grandmaster level in StarCraft II using multi-agent reinforcement learning", 2019; paper: O. Vinyals et al., *Nature* 575, 2019. https://deepmind.google/discover/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/ [abstract: official blog]
30. J. Perolat et al., "Mastering the Game of Stratego with Model-Free Multiagent Reinforcement Learning" (DeepNash), *Science* 378, 2022. https://arxiv.org/abs/2206.15378 [abstract]
31. J. Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347, 2017. https://arxiv.org/abs/1707.06347 [abstract]
32. J. Heinrich, D. Silver, "Deep Reinforcement Learning from Self-Play in Imperfect-Information Games" (NFSP), arXiv:1603.01121, 2016. https://arxiv.org/abs/1603.01121 [abstract]
33. OpenAI et al., "Dota 2 with Large Scale Deep Reinforcement Learning", arXiv:1912.06680, 2019. https://arxiv.org/abs/1912.06680 [abstract]
34. M. Fisher, "Provincial: A Kingdom-Adaptive AI for Dominion". https://graphics.stanford.edu/~mdfisher/DominionAI.html [read]
35. Meta AI, "CICERO". https://ai.meta.com/research/cicero/ ; paper: Meta FAIR Diplomacy Team et al., "Human-level play in the game of Diplomacy by combining language models with strategic reasoning", *Science* 378, 2022, https://www.science.org/doi/10.1126/science.ade9097 [abstract: official page]
36. T. Stewart (TimStewartJ), space-deck-builder (a Star Realms core-set clone with PPO training), GitHub, 2026. https://github.com/TimStewartJ/space-deck-builder ; results: `results/greedy_inference.md` (92.51% against its heuristic at temperature 0, 400 updates × 2,000 episodes), `results/search_uplift.md` (+44 Elo, budgets 0/16/64/256), README (expert iteration +19 Elo; vector engine 5.88M decisions/s). Single author, not peer-reviewed. [read]
37. K. Jones, Race for the Galaxy AI. http://keldon.net/rftg/ (the page itself doesn't describe the method); T. Duringer, "Race for the Galaxy" (TD learning and self-play in the Temple Gates app), GDC 2018, https://gdcvault.com/play/1025226/-Race-for-the-Galaxy [unverified: from the talk's listing]

38. Hearthstone AI Competition, bot downloads and results (2018-2020). https://hearthstoneai.github.io/botdownloads.html [read: the 2020 tables]
39. D. Rubin, "Unsound Search with Policy and Value Networks in Legends of Code and Magic", arXiv:2609.06816, 2026. https://arxiv.org/abs/2609.06816 [abstract]

Repo sources:

- R1. docs/star-realms-results.md: Milestone 5 (card values, tuning, rollouts, ISMCTS, the ladder). Also issues #127 and #128.
- R2. docs/research/strong-star-realms-bot.md and docs/research/engine-language.md: the research notes behind #127 and #128.
