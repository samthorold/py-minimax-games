# Star Realms: the register

The status of every hypothesis in the [End state](../star-realms.md#end-state), as of today's champion. The End state says what each hypothesis is, its test and its reading rule; this says what the test reads today. It is a reading of the current player, not a verdict: every test stays whatever its status.

**A row is due when a change could move its reading**, not when time has passed (End state: a new champion makes every row due; a change to what a test reads makes its rows due; a candidate trained for an experiment makes due only the rows that experiment reads; a docs change makes none due). *Due* says *no*, or *yes* with the change that made it due.

**Before any Star Realms change**, review each row whose *Due* is *yes*: read it now, defer it with a reason (in *Note*), or revise the End state. A review is a decision, not a run. Then mark due whatever the change makes due.

**A status comes only from a reading made under the rule**, on seeds the rule wasn't written against. The rules were rewritten on 2026-10-23 to say how each treats its instrument's error (End state, and [results, Milestone 10: pf01 read again under the reading rules](../star-realms-results.md#milestone-10-pf01-read-again-under-the-reading-rules), where a rule without it failed on noise). A rewritten rule alone doesn't make a row due, since `pf01` hasn't changed: those rows show their last reading under the earlier rule, and are read under the new one when a change makes them due.

This file is kept by hand until it can be generated from the instruments and the results doc (#214). Until then, a row changes only with a reading recorded in [the results](../star-realms-results.md), a review, or a change that makes it due.

**Champion:** `pf01`. Statuses: *supported*, *refuted* (both by the reading rule, and both provisional), *undecided* (a reading too close to call given its error), *earlier rule* (last read under a rule since rewritten; the reading is in its column), *untested* (no reading possible yet), *deferred* (untested, with a reason).

| Part | Hypothesis | Status | Last reading on `pf01` | Read | Due | Note |
|---|---|---|---|---|---|---|
| The whole player | Stronger than every player built before it | earlier rule | Supported: loses to none of the population (74.0%, 61.6%, 56.6%); against `pi03` on fresh seeds 58.0% (52.3-63.4%) | 2026-10-23 | no | |
| The whole player | Learns at the speed the feedback loop needs | earlier rule | Training half, under the current rule: **supported**, a candidate (#209's `t32`) trained in 32 s on 2 threads. Instruments' half, under the earlier rule: every instrument 1.7-13.4 minutes on 2 threads. [M10](../star-realms-results.md#milestone-10-the-trade-row-as-it-bears-on-whoever-buys-next) | 2026-11-24 | no | The instruments' half is read under the current rule with the next champion. |
| The whole player | A person could play it | earlier rule | Supported: about 0.25 s of thinking a game | 2026-10-23 | no | |
| How it chooses | Most strength from judgement; lookahead adds little | earlier rule | Supported: more examination 51.3% (47.3-55.3%); the reply 40.3% (34.9-46.0%) | 2026-10-23 | no | |
| How it chooses | Within the turn, tactics are calculated | earlier rule | Supported: 2.5% of plans change their choice | 2026-10-23 | no | |
| How it chooses | Beyond the turn, hidden cards are chance | supported | SPRT accepted H0 after 200 games (16 worlds 43.5%), under a rule set before the run and unchanged since. [M10](../star-realms-results.md#milestone-10-16-worlds-against-4-settled) | 2026-10-23 | no | |
| What it judges | Judging the position a move leads to (no move preference) | earlier rule | Supported: the examination match 51.3% (47.3-55.3%) | 2026-10-23 | no | |
| What it judges | The decisions that win are investments | earlier rule | Supported: early, a unit of trade +0.0050 and +0.0194; economy 54.0% of buys in turns 1-4, 34.4-44.6% from turn 13 | 2026-10-23 | no | |
| What it judges | A judgement of positions can be checked and explained | earlier rule | Refuted: the 0.1-0.2 tenth off by +0.031 ± 0.014 | 2026-10-23 | no | The reading that prompted the rewrite; read under the new rule with the next champion. |
| What it judges | One understanding shared with a move preference | deferred | | | no | No move preference while the afterstate hypothesis holds. |
| What it judges | A move preference learns from the whole comparison | deferred | | | no | As above. |
| What it judges | The judgement knows how sure it is | untested | | | no | The judgement gives no confidence yet (#212). |
| What it sees | The deck's cycle | earlier rule | Refuted: near above far within a phase only from turn 25 (0.197 against 0.143) | 2026-10-23 | no | Can't be read under the new rule until `diagnose` reports each phase's error (#217). |
| What it sees | Cards as proportions | earlier rule | Supported: sign z 2.26, paired against `pi03` | 2026-10-23 | no | In today's player (`features.EXTRA`). Head to head, the fits are level (50.8%). |
| What it sees | Cards by what they do | untested | Rare-card arm, under the earlier rule: never above without, by phase | 2026-10-23 | no | The rare-card arm needs #217; the synergy arm needs #211. |
| What it sees | Both players described alike | untested | | | no | No test yet. |
| What it sees | The Trade Row as it bears on whoever buys next | undecided | A judgement shown the row as hand-made features (`t32`, `features.TRADE_ROW`) sees 3.0% ± 0.1% of a row card's worth to the next buyer (`pf01` 2.4% on the same seeds), far below a quarter; sign z 0.54 over every member against `pf01`. [M10](../star-realms-results.md#milestone-10-the-trade-row-as-it-bears-on-whoever-buys-next) | 2026-11-24 | no | Undecided because the description didn't get through: by the rule, the next step is a better description, not more games. |
| What it sees | …because it lets the player deny | undecided | By `pf01`'s judgement, `t32`'s buys deny 39.2% ± 0.8%, `pf01`'s 36.7% ± 0.8% (+2.5 ± 1.1 points); the heuristic's 44.1%. [M10](../star-realms-results.md#milestone-10-the-trade-row-as-it-bears-on-whoever-buys-next) | 2026-11-24 | no | Undecided while the parent isn't supported. Split from the row above before its reading (2026-11-24): strength can come from the row without denial. |
| What it sees | Never the order of anything hidden | supported | `test_shuffling_a_deck_or_discard_pile_does_not_change_the_row` passes | 2026-10-23 | no | |
| What it learns from | The positions it played | untested | | | no | No test yet. |
| What it learns from | Positions it compared and rejected | earlier rule | Refuted: gap −0.001 ± 0.005 overall; +0.014 ± 0.014 from turn 25 | 2026-10-23 | no | Milestone 9's late-game gap (+0.033) didn't hold on fresh seeds. |
| What it learns from | Opponents unlike itself | untested | | | no | No test yet. |
| How it explores | Self-taught players learn only the strategies they try | untested | | | no | #212. |
| How it explores | Exploration is directed, not random | untested | | | no | #212. |
| How it explores | In matches, being unsure is a reason for caution | untested | | | no | #212. |
