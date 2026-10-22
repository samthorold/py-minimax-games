# Policy representation for the end-state player: claims checked

How should the end-state player's learned policy be represented: as a value of positions (afterstates), as an explicit preference over moves, or both? This note checks eight claims against primary sources. Labels follow docs/learning-and-search.md: **[read]**, **[abstract]**, **[unverified]**. "L&S [n]" means reference n in that doc.

Researched 2026-10-22.

## 1. Sutton & Barto: afterstates (6.8) and explicit policies (13.1)

**Afterstates: holds.** Section 6.8 says afterstates help "when we have knowledge of an initial part of the environment's dynamics but not necessarily of the full dynamics". In games "we typically know the immediate effects of our moves". The reason given is sharing: "many position–move pairs produce the same resulting position", and "a conventional action-value function would have to separately assess both pairs, whereas an afterstate value function would immediately assess both equally" [1] [read].

**Explicit policies (13.1): holds, with two more reasons.** The book gives four advantages of a parameterised policy (soft-max over action preferences) over action-value methods [1] [read]:

- It can approach a deterministic policy. Soft-max over action values can't, and ε-greedy always keeps an ε chance of a random move.
- It can find stochastic optimal policies. "In problems with significant function approximation, the best approximate policy may be stochastic. For example, in card games with imperfect information the optimal play is often to do two different things with specific probabilities, such as when bluffing in Poker." The worked example (13.1) is state aliasing: states that look identical to the function approximator. That is the claim's "imperfect state information", stated more narrowly.
- "The policy may be a simpler function to approximate" (Tetris is the example).
- The policy's form is a way to inject prior knowledge, "often the most important reason for using a policy-based learning method". The claim left this out.

Continuous action spaces are a fifth reason, treated in 13.7.

## 2. TD-Gammon chose moves by valuing the resulting positions

**Holds.** Tesauro: "the neural network scores every possible legal move", interpreted as "the 'equity' of the position", and "the move that is selected is then the move with maximum expected outcome" (L&S [5]) [read]. Sutton & Barto state the afterstate framing explicitly. TD-Gammon "considered each of the 20 or so ways it could play its dice roll and the corresponding positions that would result. The resulting positions are afterstates as discussed in Section 6.8" [1, §16.1] [read]. Correction to keep in mind: later versions played with a 2-ply search at decision time, so the 1-ply afterstate choice describes training (L&S [5]).

## 3. AlphaGo Zero: one network with two heads

**Holds, but the 600 Elo was measured with search.** The text says "combining policy and value together into a single network slightly reduced the move prediction accuracy, but reduced the value error and boosted playing performance in AlphaGo by around another 600 Elo" (L&S [7]) [read]. Qualifications from the same section:

- All four networks were trained on one fixed dataset of AlphaGo Zero self-play. Each was "combined with AlphaGo Zero's search", 5 seconds per move. So it is 600 Elo for the player with search, not for the raw network.
- Accuracy and error were measured on human professional games (GoKifu), not on self-play.
- The authors attribute the gain partly to efficiency, "but more importantly the dual objective regularises the network to a common representation".
- "Another" 600: separately, residual instead of convolutional layers gave "over 600 Elo".

**Targets: holds.** The policy is trained toward the search probabilities π, "proportional to N^(1/τ)" of the root visit counts. The value is trained toward "the game winner z" (L&S [7]) [read].

**Raw network vs search.** "The raw neural network, without using any lookahead, achieved an Elo rating of 3,055. AlphaGo Zero achieved a rating of 5,185" (L&S [7]) [read]. So search added about 2,100 Elo to the same network.

## 4. Expert iteration (Anthony, Tian, Barber)

**Partly holds. The paper's evidence is tree-policy versus chosen-move targets, not search versus outcomes.** It compares two imitation targets: the move MCTS chose (chosen-action targets, CAT) and the root visit distribution n(s,a)/n(s) (tree-policy targets, TPT) [2] [read]. TPT is "cost-sensitive: when MCTS is less certain between two moves (because they are of similar strength), TPT penalises misclassifications less severely". Results: similar top-1 prediction error (47.0% and 47.7%), "however, despite the very similar prediction errors, the TPT network is 50 ± 13 Elo stronger than the CAT network". The final apprentice roughly matched its MCTS expert (87 of 162 games).

On outcomes versus search: the paper says RL algorithms "often suffer from slow convergence, and high variance", and in Hex, compared with REINFORCE, "ExIt learns stronger policies faster" and "shows no sign of instability" [2] [read]. That is ExIt against policy gradient, not a measured variance of targets. "Lower variance" is the authors' motivation, not a result. Note also that their value head was trained on game outcomes of the apprentice's games, because enough expert games were too costly, in a shared network with separate policy and value heads.

## 5. DRRN (He et al.)

**Holds.** DRRN "represents action and state spaces with separate embedding vectors, which are combined with an interaction function to approximate the Q-function" [3] [read]. The interaction is usually an inner product; bilinear and a deep network over the concatenated embeddings were also tried. The motivation is a variable set: "the feasible action set A_t at each time t is an unknown subset of the unbounded action space A that varies over time" [3] [read].

## 6. Dulac-Arnold et al.: large discrete action spaces

**Holds, with one addition.** The actor outputs a "proto-action" in an embedding space. An approximate nearest-neighbour search finds the k closest real actions "in logarithmic time", and the critic Q then picks the best of those k [4] [read]. Their reason for not using a value alone: with Q(s,a), "|A| evaluations are necessary to choose an action. This quickly becomes intractable", although value-based policies generalise over actions better and can "reason about previously unseen actions". A plain actor avoids the cost but "cannot extend to previously unseen actions" [4] [read]. Tested up to a million actions.

## 7. AlphaStar: autoregressive policy and pointer network

**Holds. The unit-selection detail comes from the follow-up paper.** The Nature text says: "To manage the structured, combinatorial action space, the agent uses an auto-regressive policy and recurrent pointer network." An action picks a type, "who to issue that action to, for any subset of the agent's units", a target and a delay, "approximately 10^26 possible choices at each step". The joint probability is "decomposed via the chain rule" [5] [read: main text and methods; L&S [29] saw only the blog]. That the pointer network picks the units comes from DeepMind's AlphaStar Unplugged: two arguments "are sampled from the vector and units modalities using pointer networks", and "unit_tags is actually obtained by sampling the pointer network 64 times autoregressively" [6] [read]. The Nature paper's own figure and supplementary data weren't read.

## 8. When an explicit policy is needed rather than a value plus a model

Found in sources:

- **No model.** "If only state values are estimated, then either a model or a separate policy (as in actor–critic methods) is required for action selection" [1, §8.13] [read].
- **Too many actions to evaluate each.** Dulac-Arnold's |A| evaluations argument above [4] [read].
- **Mixed strategies.** Sutton & Barto's poker-bluffing and aliasing argument above [1, §13.1] [read]. The poker programs in L&S [25]-[28] compute mixed strategies (CFR-based), but that is their equilibrium method, not a stated rule about policy representation.
- **Decision-time cost.** "Decision-time planning is most useful in applications in which fast responses are not required ... if low latency action selection is the priority, then one is generally better off doing planning in the background to compute a policy that can then be rapidly applied" [1, §8.8] [read].

Not found: a well-cited source that sets out these conditions together as a rule. The list above is assembled from separate sources.

## What this means for Star Realms

Star Realms is a game with a perfect simulator, hidden information that behaves like chance, and turns whose end positions can mostly be enumerated.

- **A value over end-of-turn positions is the textbook fit.** The turn's effects are known and only what follows is uncertain, which is the afterstate case [1]. Many orderings of plays reach the same end position, which is the sharing argument. TD-Gammon is the precedent [1, §16.1].
- **None of the four reasons for an explicit policy applies strongly.** There is a model. Enumeration works for most turns. Decisions are offline, with no latency limit. Hidden information comes from shuffles, not from an opponent bluffing, so Sutton & Barto's stochastic-policy argument is weak here. That last point is our inference, not a source's.
- **Scoring each end position with V already has the DRRN shape**, with the "action" being its afterstate. No pointer or embedding head is needed while every end position can be listed.
- **Where a policy would earn its place:** turns whose end positions can't all be listed (a prior to order or prune them, as in AlphaGo Zero and ExIt), and training. If a policy is trained from the planner, train it on the planner's soft scores over end positions (TPT-like [2]), not only on the move it chose. A shared network with value and policy heads regularised AlphaGo Zero, but that evidence is from Go, with search.

## References

1. R. S. Sutton, A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018 (2020 online edition): §6.8 pp. 136-137, §8.8 pp. 180-181, §8.13 p. 189, §13.1 pp. 322-324, §16.1 pp. 421-425. http://incompleteideas.net/book/RLbook2020.pdf [read]
2. T. Anthony, Z. Tian, D. Barber, "Thinking Fast and Slow with Deep Learning and Tree Search", NeurIPS 2017 (L&S [8]). https://arxiv.org/abs/1705.08439 [read: sections 4-6]
3. J. He, J. Chen, X. He, J. Gao, L. Li, L. Deng, M. Ostendorf, "Deep Reinforcement Learning with a Natural Language Action Space", ACL 2016. https://arxiv.org/abs/1511.04636 [read: abstract, introduction, DRRN section]
4. G. Dulac-Arnold, R. Evans, H. van Hasselt et al., "Deep Reinforcement Learning in Large Discrete Action Spaces", arXiv:1512.07679, 2015. https://arxiv.org/abs/1512.07679 [read: sections 1-4]
5. O. Vinyals et al., "Grandmaster level in StarCraft II using multi-agent reinforcement learning", *Nature* 575:350-354, 2019 (L&S [29]). Author manuscript: https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf [read: main text and methods; not the supplementary architecture]
6. M. Mathieu et al., "AlphaStar Unplugged: Large-Scale Offline Reinforcement Learning", arXiv:2308.03526, 2023. https://arxiv.org/abs/2308.03526 [read: architecture section]

Also cited: L&S [5] (Tesauro, TD-Gammon) [read], L&S [7] (Silver et al., AlphaGo Zero, author manuscript) [read: the architecture comparison, evaluation and training sections].
