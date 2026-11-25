# Where a self-play player is weak: what the literature supports (#234)

The opinion behind the design's test of What it learns from's third source (*opponents unlike itself*) and *Learned values: the exploiter*, and its grounding. Researched 2026-11-25, briefed from the owner's question (how game-AI research finds where a self-play player is weak, beyond comparisons within its own games) without candidate answers. Tags: **[full]** read in full or the relevant sections; **[extract]** read as an extract, passage quoted; **[abstract]** abstract or snippet only; **[recalled]** not read.

## The opinion

1. **Rollouts with the player on both sides can't see an error it makes on both sides.** Backgammon practice: a rollout's systematic error arises "when the bot misplays one side of the position more than it misplays the other", and "the only way to eliminate systematic errors is to use a different type of rollout" (Zare [extract]); comparisons between two plays are the better for it, since shared error "is likely to be somewhat correlated" (GNU Backgammon manual [extract]). So `pf01`'s regret near zero (#229, #232) says it chooses well *given how it plays on*, not that it is near the best player: one step of policy improvement with itself finds nothing.
2. **Beating every opponent you have is weak evidence of how good a player is.** In Timbers et al. the AlphaZero variant that won 77% head to head was the more exploitable one (about 40% against its best response against 20%) [full]; Wang et al.'s adversary beat superhuman KataGo 97% of the time while losing to amateurs [full]. The measure is **exploitability**: how much a player built to beat the frozen player wins, a lower bound on how far it is from unexploitable (Timbers et al. [full]).
3. **A cheap exploiter starts from a copy of the player and trains only against it.** Haluška & Schmid cloned ByteRL's behaviour from its self-play games and fine-tuned against it; a card-game champion proved "highly exploitable" [full]. For a planner with a learned judgement the copy is free (its own fit), and its training games are against the frozen champion only (our inference). The exploiter's search should model the victim's actual replies (Wang et al.'s A-MCTS [full]); here the planner's lookahead ends with its own turn, so this is a later refinement, not a prerequisite (our inference).
4. **The reading is one-sided.** An exploiter that wins shows a gap; one that doesn't proves nothing, since a weak learner finds nothing (Timbers et al. [full]). An exploiter that beats the player while losing to weaker ones has found a specific flaw, not general strength (Wang et al.'s non-transitivity [full]).
5. **Cheaper checks the sources offer, and why they come second here.** Consistency checks on the judgement (Fluri et al.: evaluations that must agree, such as before and after the recommended move; adversarial search finds up to 9× more failures than random sampling [full]), and calibration on games against other opponents (Wang et al.: the victim's prediction loss stayed high for most of each game against the adversary [full]). Both read where the judgement is wrong, not whether its choices lose for it; and the opponents this project has (the heuristic, earlier champions) are already a half of every policy-iteration round's games (`iterate.mix`), so their positions are inside `pf01`'s training data, not outside it. That last point is the project's fact, not a source's.

**Don't**: choose between players by head-to-head results alone (Timbers et al. [full]); expect more search to remove the gaps ("will not produce fully robust agents on its own", Wang et al. [full]; consistency improves "slowly" with search, Fluri et al. [full]); assume that patching one exploit removes the weakness (KataGo's adversarial training stopped the fixed adversary, a fine-tuned one won 47% again, Wang et al. §5.4 [full]); read absolute values from rollouts with the player on both sides (Zare [extract]).

## What wasn't reached

The full AlphaStar Nature paper (league exploiters, snippets only [abstract]); forums (rec.games.backgammon, bkgm, Leela and KataGo Discords), so practitioners' experience of rolling out with a different bot is thin, which could overstate how cleanly that remedy works; Lisý & Bowling's local best response (known through Timbers only); PSRO and population-based evaluation; any Star Realms or Dominion exploitability study (none found). Every source is chess, Go, backgammon, poker or a collectible card game, none a deck-builder with compound turns, so how the methods carry over is inferred throughout.

## Sources

- Timbers, F. et al. (2022). Approximate Exploitability: Learning a Best Response. IJCAI 2022. arXiv:2004.09677
- Wang, T. T., Gleave, A. et al. (2023). Adversarial Policies Beat Superhuman Go AIs. ICML 2023. arXiv:2211.00241 (§§1-6)
- Fluri, L., Paleka, D. & Tramèr, F. (2023). Evaluating Superhuman Models with Consistency Checks. arXiv:2306.09983 (§5, App. B)
- Haluška, R. & Schmid, M. (2024). Learning to Beat ByteRL: Exploitability of Collectible Card Game Agents. ALA Workshop 2024. arXiv:2404.16689
- Zare, D. (2002). Rollouts. GammonVillage (bkgm.com/articles/Zare/Rollouts/)
- GNU Backgammon manual, "Rollouts in GNU Backgammon" (gnu.org/software/gnubg/manual)
- DeepMind (2019), AlphaStar blog [extract]; Vinyals, O. et al. (2019), Nature 575 [abstract]
