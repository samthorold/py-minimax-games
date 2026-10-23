# Star Realms: the register

The status of every hypothesis in the [End state](../star-realms.md#end-state), as of today's champion. The End state says what each hypothesis is, its test and its reading rule; this says what the test reads today. It is a reading of the current player, not a verdict: every test stays whatever its status, and promoting a new champion makes every row due.

**Before any Star Realms change**, review each row past its *Due* date: test it now, defer it with a reason (in *Note*, with the new due date), or revise the End state. A review that changes nothing still moves *Due* on by the hypothesis's N.

**A status comes only from a reading made under the rule**, on seeds the rule wasn't written against. The rules were rewritten on 2026-10-23 to say how each treats its instrument's error (End state, and [results, Milestone 10: pf01 read again under the reading rules](../star-realms-results.md#milestone-10-pf01-read-again-under-the-reading-rules), where a rule without it failed on noise). Every row whose rule changed is due again; its last reading, under the earlier rule, stays as context.

This file is kept by hand until it can be generated from the instruments and the results doc (#214). Until then, a row changes only with a reading recorded in [the results](../star-realms-results.md), or a review.

**Champion:** `pf01`. Statuses: *supported*, *refuted* (both by the reading rule, and both provisional), *undecided* (a reading too close to call given its error), *due* (a rule with no reading under it yet), *untested* (no reading possible yet), *deferred* (untested, with a reason).

| Part | Hypothesis | Status | Last reading on `pf01` | Read | Due | Note |
|---|---|---|---|---|---|---|
| The whole player | Stronger than every player built before it | due | Under the earlier rule, supported: loses to none of the population (74.0%, 61.6%, 56.6%); against `pi03` on fresh seeds 58.0% (52.3-63.4%) | 2026-10-23 | 2026-10-23 | |
| The whole player | Learns at the speed the feedback loop needs | due | Every instrument 1.7-13.4 minutes on 2 threads; training a candidate not timed | 2026-10-23 | 2026-11-22 | Deferred: the training half is read when the next candidate is trained. |
| The whole player | A person could play it | due | Under the earlier rule, supported: about 0.25 s of thinking a game | 2026-10-23 | 2026-10-23 | N 90. |
| How it chooses | Most strength from judgement; lookahead adds little | due | Under the earlier rule, supported: more examination 51.3% (47.3-55.3%); the reply 40.3% (34.9-46.0%) | 2026-10-23 | 2026-10-23 | |
| How it chooses | Within the turn, tactics are calculated | due | Under the earlier rule, supported: 2.5% of plans change their choice | 2026-10-23 | 2026-10-23 | |
| How it chooses | Beyond the turn, hidden cards are chance | supported | SPRT accepted H0 after 200 games (16 worlds 43.5%), under a rule set before the run and unchanged since. [M10](../star-realms-results.md#milestone-10-16-worlds-against-4-settled) | 2026-10-23 | 2026-11-22 | |
| What it judges | Judging the position a move leads to (no move preference) | due | Under the earlier rule, supported: the examination match 51.3% (47.3-55.3%) | 2026-10-23 | 2026-10-23 | |
| What it judges | The decisions that win are investments | due | Under the earlier rule, supported: early, a unit of trade +0.0050 and +0.0194; economy 54.0% of buys in turns 1-4, 34.4-44.6% from turn 13 | 2026-10-23 | 2026-10-23 | |
| What it judges | A judgement of positions can be checked and explained | due | Under the earlier rule, refuted: the 0.1-0.2 tenth off by +0.031 ± 0.014 | 2026-10-23 | 2026-10-23 | The reading that prompted the rewrite; read again under the new rule. |
| What it judges | One understanding shared with a move preference | deferred | | | 2026-11-22 | No move preference while the afterstate hypothesis holds. |
| What it judges | A move preference learns from the whole comparison | deferred | | | 2026-11-22 | As above. |
| What it judges | The judgement knows how sure it is | untested | | | 2026-11-22 | The judgement gives no confidence yet (#212). |
| What it sees | The deck's cycle | due | Under the earlier rule, refuted: near above far within a phase only from turn 25 (0.197 against 0.143) | 2026-10-23 | 2026-11-22 | Deferred until `diagnose` reports each phase's error (#217). |
| What it sees | Cards as proportions | due | Under the earlier rule, supported: sign z 2.26, paired against `pi03` | 2026-10-23 | 2026-10-23 | In today's player (`features.EXTRA`). Head to head, the fits are level (50.8%). |
| What it sees | Cards by what they do | untested | Rare-card arm, under the earlier rule: never above without, by phase | 2026-10-23 | 2026-11-22 | The rare-card arm needs #217; the synergy arm needs #211. |
| What it sees | Both players described alike | untested | | | 2026-11-21 | No test yet. |
| What it sees | The Trade Row as it bears on whoever buys next | untested | Without it, `pf01` sees 2.2% of a row card's worth to the next buyer. [Denial](../star-realms-results.md#milestone-9-probe-denial) | | 2026-11-22 | #209. |
| What it sees | Never the order of anything hidden | supported | `test_shuffling_a_deck_or_discard_pile_does_not_change_the_row` passes | 2026-10-23 | 2026-11-22 | |
| What it learns from | The positions it played | untested | | | 2026-11-21 | No test yet. |
| What it learns from | Positions it compared and rejected | due | Under the earlier rule, refuted: gap −0.001 ± 0.005 overall; +0.014 ± 0.014 from turn 25 | 2026-10-23 | 2026-10-23 | Milestone 9's late-game gap (+0.033) didn't hold on fresh seeds. |
| What it learns from | Opponents unlike itself | untested | | | 2026-11-21 | No test yet. |
| How it explores | Self-taught players learn only the strategies they try | untested | | | 2026-11-22 | #212. |
| How it explores | Exploration is directed, not random | untested | | | 2026-11-22 | #212. |
| How it explores | In matches, being unsure is a reason for caution | untested | | | 2026-11-22 | #212. |
