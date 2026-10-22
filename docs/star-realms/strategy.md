# What decides a Star Realms game

How the game is won, as players understand it, independent of any bot. Each idea is labelled:

- **[lore]**: held by experienced players, learned by playing and rarely stated with a reason; not checked here.
- **[measured]**: shown by an experiment in this repo, with the result cited.

The end state ([star-realms.md](../star-realms.md), End state) asks how far a learned player arrives at these ideas by itself, so they are also the questions its understanding is judged by.

## The race and the investment

A game ends when one player's Authority reaches zero, and nearly all damage is combat played from hand, so every game is in part a **damage race**: each turn, a player's combat is spent against the opponent's Authority and bases. But a deck is built over the game, and a card bought now is played only when the deck cycles round to it. So every buy is also an **investment**, trading this turn's tempo for a stronger deck later.

- **Strong players invest early, then turn to combat quickly.** [lore] Humans buy trade (economy) in the opening turns and switch to buying combat soon after.
- **`pf01` judges the same way.** [measured] Its judgement values combat above trade in every phase, and after turn 8 a card that adds only trade lowers its judged chance of winning; it buys economy most in turns 1-4 and less in every later band ([results](../star-realms-results.md), Milestone 9: probe, investment against the race).

## Synergy

- **Cards of the same function are worth more together.** [lore] A deck gains from a second and third card that draws, scraps, or makes the opponent discard, beyond the worth of each alone: a deck can be built around a function, not only around a faction.
- **`pf01` doesn't judge cards that draw or scrap worth more together.** [measured] A second such card adds less to its judged chance than the first, and falls off faster than a card of no function does; only a second card that makes the opponent discard adds a little more than the first, early in the game ([results](../star-realms-results.md), Milestone 9: probe, same-function synergy). It says what the judgement believes, not whether the lore is true.
- **Some factions pair particularly well.** [lore] Beyond a single faction's ally abilities, some pairs of factions suit each other: their strengths cover each other's gaps, so a two-faction deck of the right pair beats either faction alone.
- **`pf01` values a faction's own cards far above any pairing.** [measured] A card adds most to a deck mostly of its own faction, in every faction, and of the twelve pairs of different factions its judgement singles out only one, a Machine Cult card added to a deck mostly of Trade Federation, and that only slightly ([results](../star-realms-results.md), Milestone 9: probe, faction pairs). It says what the judgement believes, not which pairs are best.

## Denial

- **Keeping a card from the opponent has value, even when you don't want it.** [lore] A card in the Trade Row is worth what it would add to the deck of whoever buys it next, so buying (or scrapping) a card the opponent wants can be right even when it adds little to your own deck.
- **`pf01` holds denial weakly: in its scraps, not its buys.** [measured] Its judgement sees about 2% of what a Trade Row card is worth to the next buyer in its being there for them; enough that three in four of its Trade Row scraps take a card worth more to the opponent than to itself, more than the heuristic's scripted denial does, but its buys deny no more often than chance would ([results](../star-realms-results.md), Milestone 9: probe, denial). It says what the judgement believes, not whether denial wins games.
