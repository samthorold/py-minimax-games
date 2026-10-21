# What would make a strong Star Realms bot? (research notes, 2026-10-21)

> **Checked since (2026-10-21):** written as research notes for #128 and kept as they were, apart from issue numbers. Two claims were rechecked against their sources: the Hearthstone AI Competition 2020 results (official page, read) and Rubin 2026, arXiv:2609.06816 (abstract), both as stated here. On Baier & Cowling 2018, docs/learning-and-search.md gives the fuller reading: searching beyond the current turn helped most with short turns, and only with the longest turns was the current turn alone best. That doc is the reviewed summary; treat these notes as the working record.


Status legend: **[read]** = I read the primary source (full text or the relevant results section) in this session. **[abstract]** = only abstract / listing / secondary summary seen. **[unverified]** = from memory or second-hand, not checked this session. No citation below is invented; anything not marked [read] should be checked before it goes into project docs.

## 0. Bottom line

- **ISMCTS is not the answer by itself.** Across deckbuilders and card games, the strong agents either have a **learned evaluation/policy** (TD self-play MLP, PPO, AlphaZero-style nets) or a **simulation-optimised buy policy**. Search adds a smaller amount on top of a good evaluator. Pure MCTS/ISMCTS with a hand-made evaluation usually lands near a good rule bot, not clearly above it.
- **Your own numbers point the same way.** The cheat sees the true state and still only ties the heuristic (52%), so hidden information is not the main limit. The evaluation (hand-chosen card values) and the micro-action granularity are. ISMCTS uses `sigmoid(eval)` at its leaves, so it inherits the same ceiling. 4x iterations adding nothing means the error is bias, not variance. The one Star Realms search-on-a-net study found the same flat budget curve (below).
- **The closest prior art is a Star Realms base-set clone with PPO self-play.** It beats its rule-based heuristic **92.5%** with no search (greedy decoding). Adding turn-bounded search on top gains only **+44 Elo**, and nothing beyond 64 rollouts per decision.
- **Recommendation, short form:** (1) learn a state-value function V from self-play, on deck-composition features, TD-Gammon / Race-for-the-Galaxy style, in numpy. (2) Use it for buys via 1-ply afterstate evaluation of whole-turn outcomes. (3) Only then put PIMC or ISMCTS on top, at turn granularity, with V at the leaves. Full PPO is the high-ceiling option, but the evidence for it comes from GPU + Numba compute.

## 1. Published bots: what technique produced the strongest play

### Star Realms specifically
- **TimStewartJ/space-deck-builder** (GitHub, 2025-2026, single author, not peer-reviewed) **[read: README, results/*.md, heuristic_agent.py]**. It implements Star Realms' base set ("49 canonical cards", including Stealth Needle). The strongest results in this note come from here:
  - **Training:** PPO, MLP policy + critic over a 505-float observation. Self-play against a snapshot league (PFSP) plus fixed Random, Heuristic and Simple opponents. 400 updates × 2,000 episodes = **800k games per seed, about 1 hour per seed**. That used a Numba vector engine (5.9M engine decisions/s, 587k/s with the policy) and a GPU.
  - **Strength:** the self-play policy beats their rule-based `HeuristicAgent` **86.4%** when sampling and **92.5% ± 0.9pp** with greedy (T=0) play, over 5,000 games per seed and 3 seeds. That heuristic plays everything, then draws, checks for lethal, attacks outposts, bases and the player, and buys by score. It looks broadly similar to a hand-written Star Realms heuristic. Whether it is as strong as yours is unknown.
  - **Turn-bounded search on top of the net** (`results/search_uplift.md`). Each decision compares the policy's top 8 legal actions by sequential halving. A rollout re-deals the opponent's unseen hand, continues the searcher's turn greedily, and scores the result by the critic. At 16, 64 and 256 rollouts per decision this beats the same net playing greedily 54.8%, **56.9%** and 56.7%. Pooled over 3 seeds at 64 rollouts: **56.4%, +44 Elo**. It changes only about 3% of decisions, and **256 rollouts is no better than 64**. The author's reading is that leaf evaluation, not sampling noise, is the limit. This is the same pattern as your 4x ISMCTS run.
  - **One round of expert iteration** (distilling search into the policy): +19 Elo (CI +4 to +35). It missed the author's pre-registered +20 threshold.
- **icesphere/star-realms-simulator** (Java, 2016) **[read: README, results file, SimulatorBot.java]**. Its rule bots are buy-priority scorers (Velocity, Attack, Defense, Tortoise, Hare, Expensive). On the base set, VelocityBot wins 63% overall against the others, and ExpensiveBot (buy the most expensive card) 29%. `SimulatorBot` picks the buy by **flat Monte Carlo**: it simulates whole games for each candidate card, with a rule bot finishing the game, and keeps the card with the highest win%. That is the Dominion "buy simulator" idea applied to Star Realms. No head-to-head strength is published.
- **Official app AI (White Wizard Games):** I found no technical information. Store pages only list 3 difficulty levels **[abstract]**.
- Other GitHub Star Realms engines (hardiepiennar/starrealms_ai with an LLM agent, SalahKouhen explorer-only bot, and several plain engines) have no strength results **[read: READMEs]**.

### Dominion
- **Provincial** (M. Fisher, Stanford 2014) **[read]**. It splits strategy into a **buy strategy**, an ordered "buy menu" of (card, count) plus thresholds for switching to victory cards, and a **play strategy** of hand-written general heuristics with a simple lookahead. Buy menus are found by **competitive co-evolution**: millions of C++ self-play games, about 40k games/s on 8 cores, minutes per kingdom. Fisher's quote: "Determining good buy strategies is by far the more challenging of these two components ... almost all of the AI's compute time is spent working out which combinations of cards are most effective." He claims it is "easily competitive with experienced players", with no formal evaluation.
- **Gerigk & Engels, AIIDE 2023, "Playing Various Strategies in Dominion with Deep RL"** **[read: most of the text]**. Multiset card embeddings with discrete Soft Actor-Critic. It wins 73% against a strong Big Money bot. With a trashing heuristic, 78% against Big Money, but only 39% / 32% against Single / Double Witch Big Money bots. Engine strategies needed curriculum hacks and were still weak (14% against Big Money). About one day of training per agent. Prior work as summarised in this paper (all **[unverified]** beyond that summary):
  - Jansen & Tollisen 2014, MCTS: 68.5% against Big Money + Witch, but never found action-card combos.
  - Winder 2014, neural nets trained by TD, hill-climbing or a GA: the GA with 2 nets won 74.7% against Big Money.
  - Fynbo & Nellemann 2010: neural-net card evaluation.
  - Angelopoulos & Metafas, Q-learning: 57%.
- **Temple Gates Dominion app** **[read: their blog and a Gametek article; marketing-level detail]**. A neural value net with a learned card-embedding layer (cards described by about 10 latent "DNA" components) and weights shared across players, trained by self-play with TD, plus AlphaZero-style net-guided tree search. Claims "win rate over eighty percent", with no stated opponent or protocol, so treat it as anecdotal.

### Race for the Galaxy (closest well-known shipped analogue)
- **Keldon Jones' RftG AI**, later shipped by Temple Gates. Source: Gametek / Temple Gates write-up **[read, secondary]**. A small MLP **trained by TD learning from self-play, starting from random play**, scoring each candidate move by **one-turn lookahead** with the net. The write-up reports 30,000+ training games and a "97% win rate", with no stated opponent. Keldon's site confirms the AI ships neural networks with its source **[read]**. This is TD-Gammon's recipe in a hidden-information card game, and it is widely regarded as strong against humans (that last point **[unverified]**).

### Hearthstone / CCGs
- **Hearthstone AI Competition**, premade-deck track 2020 **[read: bot list with win rates]**. The top two were **2-step Lookahead (72.3%)** and **Dynamic Lookahead (70.9%)**: shallow enumeration of the whole turn plus a hand-tuned board evaluation. MCTS came 3rd (68.7%). UCT came 8th, and other MCTS entries 10th and 15th. In the user-deck track, Dynamic Lookahead took 1st and 2nd. The 2018 premade winner (Frick & Akkaya) is listed as "PredatorMCTS" **[abstract]**. García-Sánchez et al. (arXiv 2410.19681) evolved such agents' evaluation weights by competitive coevolution. They placed runner-up and were "competitive with MCTS" **[abstract]**.
- **Zhang & Buro, CIG 2017** **[abstract]**. Determinized MCTS for Hearthstone, improved by (a) **bucketing chance outcomes** with similar effects and (b) **learned high-level rollout policies**. They report significant gains over the then state-of-the-art Hearthstone AI.
- **ByteRL (Xi et al., arXiv 2303.04096)** **[abstract]**. An end-to-end policy trained with Optimistic Smooth Fictitious Play, **no search**. It won both COG 2022 LOCM tracks and beat a top-10 Chinese Hearthstone player. Haluska & Schmid (arXiv 2404.16689) show it is "highly exploitable" **[abstract]**.
- **Rubin, arXiv 2609.06816 (Sept 2026), "Unsound Search with Policy and Value Networks in LOCM"** **[abstract]**. The agent imitates the LOCM runner-up's policy, then runs **PIMC** over worlds sampled from a deck prior. Without search it beats ByteRL **26.8%**; with search **51.35%** (10k games). That is +24.6 points from PIMC search on top of nets, and the searched agent was also more robust to best-response attacks. This is good evidence that "unsound" determinization is fine when there are nets at the leaves.

### Slay the Spire (single-player, so only loosely relevant)
- bottled_ai: rule-based, with combat simulation, "52% win rate" for its best Watcher strategy, ascension level unstated **[abstract]**.
- sts_lightspeed: a C++ simulator with MCTS for combat. A learned non-combat (pathing / card-pick) policy on top raised win rate from 2% to 14% in one configuration **[abstract]**. The deck-building (strategic) choices were where learning paid.

### Magic
- Cowling, Ward & Powley 2012, ensemble determinization: already in your notes. Forge's heuristic AI **[unverified]**: no strong learned agent is known to me.

## 2. ISMCTS vs PIMC vs determinized MCTS when hidden info is mostly chance

- Long et al. 2010 (in your notes): PIMC works well when leaf correlation is high and disambiguation is fast. Star Realms' hidden information is mostly shuffle order, which fits that case.
- Whitehouse, Powley & Cowling 2011, Dou Di Zhu: determinized UCT and ISMCTS showed no significant difference **[unverified; reported second-hand in arXiv 1906.04439]**. Cowling, Powley & Whitehouse 2012 (in your notes): ISMCTS's advantage shows where strategy fusion matters. In games where the opponent's hidden information is mostly chance, the strategy-fusion fix buys little. Your result (ISMCTS ≈ PIMC, both 34%) fits this.
- Rubin 2026 (above): PIMC + nets gave +24.6 points in a CCG, so determinization is not the bottleneck once the evaluation is good.
- **Enhancements known to help ISMCTS/MCTS in practice:**
  - MAST / NAST / "Bandits all the way down" (UCB1 as the simulation policy; Powley, Whitehouse & Cowling, ~CIG 2013) **[unverified]**.
  - RAVE.
  - Bucketing chance nodes (Zhang & Buro).
  - Learned rollout / prior policies (Zhang & Buro; AlphaZero-style PUCT).
  - Split moves (below).
  - In the Star Realms PPO study, the effective search was restricted to the **policy's top-k actions**, with **sequential halving** and **critic-valued leaves at the end of the turn**, not deep rollouts **[read]**.

## 3. Multi-action turns: granularity and budget

- **Justesen, Mahlmann, Risi & Togelius, IEEE TCIAIG 2017, Online Evolutionary Planning** (Hero Academy, 5 action points per turn, branching 30-60 per action, so about 10^7-10^8 per turn) **[abstract]**. A genome is a whole-turn action sequence, and fitness is the heuristic evaluation at the end of the turn. OEP beat MCTS variants and greedy search "by a large margin", and its advantage grew with actions per turn. Non-exploring MCTS (C=0, rollouts to the end of the turn, then evaluation) and "bridge-burning" MCTS helped MCTS a lot.
- **Baier & Cowling, AIIDE 2018, Flexible-Horizon Evolutionary MCTS** **[read in full]**:
  - EMCTS: the tree is over whole-turn genomes, edges are mutations, leaves are evaluated with no rollouts.
  - FH-EMCTS extends the genome past the current turn, filling the opponent's moves from a cheap greedy model.
  - FH-EMCTS beats vanilla EMCTS 57-85% and greedy OEP 60-90% at 1 s per turn, for 2-15 action points.
  - **Searching beyond the current turn mattered most for short turns.** For long turns (15 action points) the best horizon was just the current turn, because a limited budget is better spent on getting this turn right.
  - All agents used a **hand-tuned linear evaluation**. The authors list learning it as future work.
- **Kowalski et al., AAAI 2022, "Split Moves for MCTS"** **[abstract]**. Splitting a turn into sub-decisions inside MCTS "can be greatly beneficial" in both single- and multi-action games. The right split granularity matters, and it combines with MAST and RAVE.
- **Hearthstone competition:** whole-turn enumeration with an evaluation of the end-of-turn state (dynamic / 2-step lookahead) won outright against MCTS entries **[read]**.
- **Implications for Star Realms:**
  - Most micro-action orderings are transpositions (your docs measured 2-3,900x).
  - Many plays are dominant: play every ship with no choice, and take ally abilities when free.
  - The real per-turn decisions are a handful: which cards to buy, scrap and discard choices, the attack split between bases and the player, and the order of a few interacting choices.
  - So the right unit of search is the **set of distinct end-of-turn afterstates**, which your transposition table already enumerates. Evaluate those with a good V, rather than spending ISMCTS iterations on orderings. If a turn has too many afterstates, OEP/EMCTS over the remaining real choices is the published fallback.

## 4. Learned evaluation for deckbuilders: feasible on a laptop in numpy?

- **Precedent:**
  - Race for the Galaxy: a TD self-play MLP, about 30k games reported, in C, used as a 1-ply evaluator.
  - Winder 2014, Dominion: TD-trained nets, 1 kingdom **[unverified]**.
  - TD-Gammon: the classic template.
  - Space-deck-builder: a PPO MLP on a 505-float observation beat its heuristic 92%. But that was 800k games per seed on a vector engine. Sample efficiency: every seed was above 70% against the heuristic after 50 updates, about 100k games **[read]**.
- **Features that work** in Temple Gates, Gerigk and space-deck-builder:
  - Per-card (or per card-embedding) **counts by zone** for both players: deck + discard, hand, bases in play. Your opponent's deck composition is public in Star Realms.
  - Authority for both players, turn number, trade-row contents.
  - Embeddings, or simple hand features per card (trade / combat / authority / draw / scrap / faction), give generalisation with few weights.
- **A laptop-feasible recipe** (my estimate, not measured):
  - Start with **linear or logistic V** on about 100-200 features, fit by TD(λ) or Monte-Carlo-return regression on games from heuristic-vs-heuristic play with ε-noise. Then iterate: games played by "1-ply afterstate-greedy on V" against itself and the heuristic (generalised policy iteration).
  - Then a one-hidden-layer numpy MLP (64-128 units), as in RftG.
  - Data: 20-100k games. At your engine speed, a rule-bot game is probably tens of ms to a few hundred ms (not measured here). That is roughly 1-10 CPU-hours per round with `--jobs`.
  - Train time in numpy is negligible next to game generation.
- **Expected strength** (inference, so honest uncertainty):
  - A good V used 1-ply plausibly beats the hand-valued evaluation clearly: 60-75% against your heuristic. That rests on RftG, the 92% PPO result against a similar heuristic, and Dominion learned agents beating Big Money 73-78%.
  - Linear V will cap lower than an MLP, because synergies (faction ally pairs) are nonlinear. Add pairwise faction-count features if staying linear.

## 5. Ranked recommendation for this project

**(a) Strategic layer: what to buy, i.e. long-term deck value.** This is where the evidence says the strength is.

1. **Learned V(state) from self-play (TD / MC regression), used as afterstate evaluation for buy and scrap decisions.** Highest expected gain: my guess is +10-25 points against the heuristic. Cost: about 1-2 weeks of work. Feature encoder, numpy TD trainer, a generate→fit→evaluate loop on `tournament --jobs`, a few CPU-hours per iteration. Gate each round on a fixed tournament against the heuristic, as space-deck-builder does with paired seeds.
2. **Cheap stopgap: flat Monte Carlo buy choice**, as in icesphere's `SimulatorBot`. For each affordable buy option, play N full-game rollouts with the heuristic bot from sampled worlds, and buy the best win%. Medium gain. It tends to beat its own rollout policy, but it is slow per decision and capped by the rollout policy's quality. The rollout-evaluation contender (#116, docs/star-realms-results.md, Milestone 5: rollouts) is related but differs: it scored alpha-beta leaves with 2-turn greedy playouts, not full games per buy, and did not beat the linear evaluation.
3. **Provincial-style evolved buy priorities** (co-evolve the heuristic's card-value table and phase thresholds by tournament). Weight tuning (#115) already found no change that beat the noise, so expect diminishing returns. Cheapest of the three, lowest ceiling.

**(b) Tactical layer: playing out the turn.**

4. **Collapse the turn.**
   - Auto-play dominant actions.
   - Enumerate distinct end-of-turn afterstates (your transposition table), or OEP / EMCTS when there are too many.
   - Pick the turn by V averaged over a few determinizations of any mid-turn draws.
   - Expected gain is moderate. Hearthstone evidence: whole-turn lookahead beat MCTS; Hero Academy: whole-turn optimisation beat action-by-action MCTS. Cost: days, and most of the machinery exists.
5. **Then PIMC (simplest) or ISMCTS at turn granularity**, with V (and, if learned, a policy prior / top-k pruning) at the leaves, and a horizon of the current turn plus the opponent's reply.
   - Expected gain: +30-50 Elo on top of whatever (1) gives (space-deck-builder: +44 Elo; LOCM: +25 points with a weaker base). Do not expect gains from budget scaling past a small number of rollouts per candidate.
   - ISMCTS is not expected to beat PIMC here, because the hidden information is chance-like.
6. **Full PPO / actor-critic self-play** (high ceiling, high cost). The best direct Star Realms evidence (92% against a heuristic), but that took 800k games per seed on a Numba engine plus a GPU. In pure CPython it is probably impractical without a much faster engine, and the published code is GPU / Windows-oriented. Consider it only after (1) shows learning works and plateaus. Reusing their feature design (505-float observation) and gauntlet protocol is cheap.

**Where the evidence is thin:**
- Nobody has published a peer-reviewed Star Realms bot.
- The space-deck-builder heuristic may be weaker than yours.
- Commercial "80% / 97%" claims have no protocol.
- I did not read the full OEP, Split Moves, Zhang & Buro, ByteRL or Rubin papers; their numbers above are from abstracts.

## Sources
- TimStewartJ/space-deck-builder: https://github.com/TimStewartJ/space-deck-builder (results/optimized_selfplay_baseline.md, greedy_inference.md, search_uplift.md, expert_iteration_round1.md, batched_search.md; src/ai/heuristic_agent.py)
- icesphere/star-realms-simulator: https://github.com/icesphere/star-realms-simulator (simulator_results.txt, bots/SimulatorBot.java)
- Fisher, Provincial: https://graphics.stanford.edu/~mdfisher/DominionAI.html
- Gerigk & Engels, AIIDE 2023: https://ojs.aaai.org/index.php/AIIDE/article/download/27518/27291/31569
- Temple Gates, Dominion AI: https://www.templegatesgames.com/dominion-ai/
- Gametek, "Big Brain, Small Phone": https://gametek.substack.com/p/big-brain-small-phone-ai-for-dominion
- Keldon Jones RftG AI: http://keldon.net/rftg/
- Baier & Cowling, FH-EMCTS, AIIDE 2018: https://ojs.aaai.org/index.php/AIIDE/article/view/13023
- Justesen et al., OEP: https://pure.itu.dk/en/publications/playing-multi-action-adversarial-games-online-evolutionary-planni/
- Kowalski et al., Split Moves, AAAI 2022: https://arxiv.org/abs/2112.07761
- Hearthstone AI Competition bot list / results: https://hearthstoneai.github.io/botdownloads.html, https://hearthstoneai.github.io/results2018.html
- García-Sánchez et al., arXiv 2410.19681
- Zhang & Buro, CIG 2017 (thesis: https://era.library.ualberta.ca/files/cv979v349b/Zhang_Shuyi_201709_MSc.pdf)
- Xi et al., ByteRL, arXiv 2303.04096; Haluska & Schmid, arXiv 2404.16689; Rubin, arXiv 2609.06816
- Bravi et al., Rinascimento (Splendor), arXiv 1904.01883 (not read; MCTS/RHEA tuning-sensitivity only)
- Slay the Spire: https://github.com/xaved88/bottled_ai, https://github.com/Jialeiv/sts-rl-agent
- Card-game AI survey (Dou Di Zhu claim): arXiv 1906.04439
