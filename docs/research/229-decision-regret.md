# Regret by kind of decision: what the literature supports (#229)

The opinion behind the design's *Learned values: regret by kind of decision*, and its grounding. Researched 2026-11-24, briefed from the owner's question (which kinds of decision the player loses most on, measured with rollouts) without candidate answers. Tags: **[full]** read in full or the relevant sections; **[extract]** read as an extract, passage quoted; **[abstract]** abstract or snippet only; **[recalled]** not read.

## The opinion

1. **Measure regret at the level of the whole turn first, against a shortlist rolled out with the player's own policy**, then attribute only the turns that lose. This is established practice for stochastic games: on-line policy improvement in backgammon rolls out each candidate with the base policy and grades moves by equity lost (Tesauro & Galperin 1996 [extract]), a rollout policy is no worse than its base (Bertsekas 2019 [abstract]), and backgammon analysis reports error per decision against a rollout or search reference, by decision type, over unforced decisions (GNU Backgammon manual [extract]; eXtreme Gammon [extract]; Strehl 2026 [extract]). A shortlist from a decent screen loses little (eXtreme Gammon's search-interval study: 0.0034 equity per game against the evaluator's own 0.0271 [extract]); sequential halving narrows one at a fixed budget (Ontañón 2017 [full]).
2. **Correct for the best of noisy estimates**: the apparent best falls short of its estimate even when each estimate is unbiased (Smith & Winkler 2006 [abstract]), so the best is re-estimated on fresh rollouts before differencing.
3. **Share randomness across options, per source of randomness, and check it holds**: common random numbers need synchronised seeds and systems that respond alike to the same input (Glasserman & Yao 1992 [full]). In a deck-builder two options that buy different cards break the player's own shuffles, so the help is least for buys, which is why the design measures it.
4. **Report loss per game (frequency × size), by phase and by how decided the position is, without forced or trivial decisions**: error grows with the size of the advantage and is compressed in decided positions (Regan & Haworth 2011 [full]); measures ignoring difficulty mislead (Guid & Bratko 2006, 2011 [extract]); the reference budget must be fixed across compared decisions (Guid & Bratko 2011 [extract]).
5. **Within-turn attribution has no established method in game analysis.** Backgammon splits checker play from cube action, which are separate decisions; card-game work reached treats compound turns as a search problem. The nearest tools are from multi-agent and combinatorial-action RL: component-at-a-time rollout (Bertsekas 2019 [abstract]), counterfactual baselines (Foerster et al. 2018 [abstract]), additive decomposition (Ontañón 2017 [full]; Tang et al. 2022 [abstract]), Shapley credit [abstract]. They disagree when sub-choices interact, as in a deck-builder; an exact Shapley split over the few sub-choices that differ is order-free and affordable on the turns that lose. Using them for diagnosis is our inference.
6. **Known blind spot**: a self-policy reference sees only errors the player could fix in one step, and the shortlist is chosen by the judgement under test (Tesauro & Galperin flag the same about grading with the network that generated the data [extract]).

**Don't**: take the max of noisy estimates uncorrected; rank by mean loss without frequency and difficulty; read agreement with a self-policy reference as truth; vary the reference budget between compared decisions; count forced decisions; trust evaluator-based luck correction or truncated rollouts in the positions the judgement misjudges, which are the ones sought (GNU Backgammon rollouts manual [extract]; Tesauro & Galperin found truncated rollouts weaker [extract]).

## What wasn't reached

Forums (not attempted); eXtreme Gammon's manual on pairing dice across candidates; full texts of the multi-agent attribution papers and of Dominion and Hearthstone theses (snippets only). The picture leans on backgammon and chess, where a decision is atomic and the reference is stronger than the player; "no game method for within-turn attribution" is absence of evidence.

## Sources

- Tesauro, G. & Galperin, G. R. (1996). On-line Policy Improvement using Monte-Carlo Search. NIPS 9. arXiv:2501.05407
- Bertsekas, D. P. (2019). Multiagent Rollout Algorithms and Reinforcement Learning. arXiv:1910.00120
- Smith, J. E. & Winkler, R. L. (2006). The Optimizer's Curse. Management Science 52(3). doi:10.1287/mnsc.1050.0451
- Glasserman, P. & Yao, D. D. (1992). Some Guidelines and Guarantees for Common Random Numbers. Management Science 38(6)
- GNU Backgammon manual: "Output of an analysis", "Rollouts in GNU Backgammon" (gnu.org/software/gnubg/manual)
- eXtreme Gammon, "Search interval" study (extremegammon.com/SearchInterval.aspx)
- Strehl, A. L. (2026). PureTD: RL for Backgammon Money Games with No Evaluation-time Search. arXiv:2608.15146
- Regan, K. W. & Haworth, G. McC. (2011). Intrinsic Chess Ratings. AAAI-11
- Guid, M. & Bratko, I. (2006, 2011). ICGA Journal 29(2), 34(2)
- Ontañón, S. (2017). Combinatorial Multi-armed Bandits for Real-Time Strategy Games. JAIR 58. arXiv:1710.04805
- Foerster, J. et al. (2018). Counterfactual Multi-Agent Policy Gradients. AAAI-18. arXiv:1705.08926
- Tang, S. et al. (2022). Leveraging Factored Action Spaces for Efficient Offline RL in Healthcare. NeurIPS. arXiv:2305.01738
