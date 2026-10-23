# Star Realms: the register

The status of every hypothesis in the [End state](../star-realms.md#end-state), as of today's champion. The End state says what each hypothesis is, its test and its reading rule; this says what the test reads today. It is a reading of the current player, not a verdict: every test stays whatever its status, and promoting a new champion makes every row due.

**Before any Star Realms change**, review each row past its *Due* date: test it now, defer it with a reason (in *Note*, with the new due date), or revise the End state. A review that changes nothing still moves *Due* on by the hypothesis's N.

This file is kept by hand until it can be generated from the instruments and the results doc (#214). Until then, a row changes only with a reading recorded in [the results](../star-realms-results.md), or a review.

**Champion:** `pf01`. Statuses: *supported*, *refuted* (both by the reading rule, and both provisional), *untested*, *deferred* (untested, with a reason). **†** marks a reading made before its rule was declared (the rules were written on 2026-10-23, after Milestone 9's readings); those statuses are the rule applied to an earlier reading, and the next champion's reading is the first the rule binds.

| Part | Hypothesis | Status | Reading on `pf01` | Read | Due | Note |
|---|---|---|---|---|---|---|
| The whole player | Stronger than every player built before it | supported † | Loses to none of the population (74.0%, 61.6%, 56.6%); beat `pi03` 54.0% (49.6-58.3%) on fresh seeds after the SPRT gate. [M8](../star-realms-results.md#milestone-8-hand-made-features-at-pi03s-data-size-offline), [M9](../star-realms-results.md#milestone-9-the-reference-population-and-pf01s-baseline) | 2026-10-22 | 2026-11-22 | |
| The whole player | Learns at the speed the feedback loop needs | supported † | Every instrument 2-15 minutes on 2 contended threads; `pf01` fitted offline in minutes. [M9 baseline](../star-realms-results.md#milestone-9-the-baseline) | 2026-10-22 | 2026-11-22 | |
| The whole player | A person could play it | supported † | Estimated, not read: 13.8 ms per plan and about 12 turns a game each, far under 60 s. The population run doesn't report time per game. [M10](../star-realms-results.md#milestone-10-16-worlds-against-4-settled) | 2026-10-23 | 2027-01-21 | N 90. Read directly once the population run reports time per game. |
| How it chooses | Most strength from judgement; lookahead adds little | supported † | More examination 50.8% (46.8-54.8%); the reply 44.7% (39.1-50.3%). [More examination](../star-realms-results.md#milestone-9-more-examination-where-the-budget-runs-out), [reply](../star-realms-results.md#milestone-9-looking-through-the-reply) | 2026-10-22 | 2026-11-22 | |
| How it chooses | Within the turn, tactics are calculated | supported † | 2.6% of plans change their choice. Reported beside it: 18% of turns not examined in full, 57% from turn 25. [More examination](../star-realms-results.md#milestone-9-more-examination-where-the-budget-runs-out) | 2026-10-22 | 2026-11-22 | |
| How it chooses | Beyond the turn, hidden cards are chance | supported | SPRT accepted H0 after 200 games (16 worlds 43.5%); rule set before the run. [M10](../star-realms-results.md#milestone-10-16-worlds-against-4-settled) | 2026-10-23 | 2026-11-22 | |
| What it judges | Judging the position a move leads to (no move preference) | supported † | The examination match: 50.8% (46.8-54.8%). | 2026-10-22 | 2026-11-22 | |
| What it judges | The decisions that win are investments | supported † | Early, a unit of trade +0.004 and +0.019; economy 55.8% of buys in turns 1-4, 33.8-38.8% from turn 13. [Investment](../star-realms-results.md#milestone-9-probe-investment-against-the-race) | 2026-10-22 | 2026-11-22 | The probe's own reading, against the earlier lore (invest more and longer than bots), was a failure; this rule asks only whether it invests early. |
| What it judges | A judgement of positions can be checked and explained | supported † | Every tenth of judged chance within 0.02 on chosen positions; all four probes read. [Gap](../star-realms-results.md#milestone-9-the-exploitation-gap) | 2026-10-22 | 2026-11-22 | |
| What it judges | One understanding shared with a move preference | deferred | | | 2026-11-22 | No move preference while the afterstate hypothesis is supported. |
| What it judges | A move preference learns from the whole comparison | deferred | | | 2026-11-22 | As above. |
| What it judges | The judgement knows how sure it is | untested | | | 2026-11-22 | The judgement gives no confidence yet (#212). |
| What it sees | The deck's cycle | refuted † | Squared miss near a reshuffle against far: no consistent direction within phases. [Errors](../star-realms-results.md#milestone-9-the-judgements-errors-by-reshuffle-distance-and-rare-cards-and-turns-not-examined) | 2026-10-22 | 2026-11-22 | Not built while refuted. |
| What it sees | Cards as proportions | supported † | A fit with the hand-made features against the same fit without: 71 seeds won only with them, 36 only without, z ≈ 3.4. [M8](../star-realms-results.md#milestone-8-hand-made-features-at-pi03s-data-size-offline) | 2026-10-22 | 2026-11-22 | In today's player (`features.EXTRA`). |
| What it sees | Cards by what they do | untested | One arm read †: positions with a rare card judged no worse (0.211 against 0.232 in turns 1-6). The synergy arm needs a judgement shown what cards do. | 2026-10-22 | 2026-11-22 | #211. |
| What it sees | Both players described alike | untested | | | 2026-11-21 | No test yet. |
| What it sees | The Trade Row as it bears on whoever buys next | untested | Without it, `pf01` sees 2.2% of a row card's worth to the next buyer. [Denial](../star-realms-results.md#milestone-9-probe-denial) | | 2026-11-22 | #209. |
| What it sees | Never the order of anything hidden | supported | `test_shuffling_a_deck_or_discard_pile_does_not_change_the_row` passes. | 2026-10-23 | 2026-11-22 | |
| What it learns from | The positions it played | untested | | | 2026-11-21 | No test yet. |
| What it learns from | Positions it compared and rejected | refuted † | Gap +0.002 ± 0.005 overall. [Gap](../star-realms-results.md#milestone-9-the-exploitation-gap) | 2026-10-22 | 2026-11-22 | From turn 25 the gap is +0.033 ± 0.012: a finer hypothesis to declare in the End state before it is read again. |
| What it learns from | Opponents unlike itself | untested | | | 2026-11-21 | No test yet. |
| How it explores | Self-taught players learn only the strategies they try | untested | | | 2026-11-22 | #212. |
| How it explores | Exploration is directed, not random | untested | | | 2026-11-22 | #212. |
| How it explores | In matches, being unsure is a reason for caution | untested | | | 2026-11-22 | #212. |
