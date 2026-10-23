# minimax-games

Search and learning applied to games. The terms below are those of the Star Realms work, where learned play and search are compared.

## Language

### The player

**Position**:
The public state of a game at a turn boundary, the moment a player ends their turn: what every player could know, never the order of anything hidden.
_Avoid_: state (for the public part alone), board

**Judgement**:
A player's learned sense of how likely the player who has just ended their turn is to win from a **position**, with how confident it is in that.
_Avoid_: evaluation (the hand-written `evaluate`), value function (as a design term)

**Move preference**:
A learned leaning towards some moves over others, used only to decide which outcomes a player examines when it can't examine them all.
_Avoid_: policy (ambiguous: also the whole rule a player chooses by)

**Champion**:
The current best learned player; each new one must beat it and lose to none of the **reference population**.
_Avoid_: best model, incumbent

**Candidate**:
A player trained to challenge the **champion**, not yet promoted.

### Measuring

**Reference population**:
The fixed set of opponents every candidate plays on fixed seeds: the heuristic, every champion, and any hand-crafted player kept as an opponent.
_Avoid_: ladder (the older, champion-only chain), gauntlet

**Probe**:
A question put to a trained **judgement** about one idea from Star Realms strategy, answered by comparing **positions** that differ in that one fact.

**Exploitation gap**:
How much a player's **judgement** of the positions it chose exceeds how often it went on to win from them.

**Hypothesis**:
A claim about Star Realms or about choosing moves that a part of the **end state** rests on, with the cheapest test that could show it false; a hypothesis because the end state itself is uncertain.
_Avoid_: belief, claim

**Reading rule**:
What result of a **hypothesis**'s test supports it and what refutes it, declared with the hypothesis before the test is read.

**Register**:
The status of every **hypothesis** (untested, supported or refuted) as of today's **champion**, with the reading behind it and when it is next due for review. A reading of today's player, not a verdict: a test stays whatever its status.

**End state**:
The player the work moves towards, described by what it knows and does and the **hypotheses** behind them, never by how it is built.

## Example dialogue

> **Dev:** The candidate beat the champion 56%. Promote it?
> **Expert:** Only if it lost to nobody in the reference population. And which hypothesis did it test?
> **Dev:** The deck's cycle. The judgement's error near a reshuffle fell by half.
> **Expert:** Then strength and the diagnostic agree: that's a result. Did the probes change?
> **Dev:** The faction-pair probe still finds no pairs. And we aren't learning from rejected positions: the exploitation gap was small.
