"""The immutable game state.

Every pile is a tuple, and the deck's order is concrete: the top card is
`deck[0]`. Randomness lives in `GameState.rng_seed`, an integer. Each shuffle
seeds a fresh `random.Random` from it, shuffles, and draws the next seed from
that generator, so the state stays immutable and hashable and one starting
seed fixes every shuffle in the game.

The Trade Row is `trade_row`, refilled from the top of `trade_deck`; the
Explorer pile holds `explorers` identical Explorers, so it is just a count.
Scrapped cards go to `scrap_heap`, in the order they were scrapped (except
Explorers, which go back to their pile).

`allies_used` lists the cards in play whose ally ability has been used this
turn, one entry per use. Identical copies are interchangeable, so a card's
ally ability is still available while it has more copies in play than entries
here. The list empties at the end of the turn.

A player's bases stay in play (`PlayerState.bases`) across turns, unlike
ships, which are discarded at the end of the turn. `bases_used` lists the
current player's bases whose primary ability has been used this turn (bases
played this turn among them), in the same one-entry-per-use way.

`may_scrap` counts the scraps from hand or discard pile the current player
may still make this turn (`ScrapHandOrDiscard` effects): optional, usable at
any time in the turn, and lost at its end. `must_scrap` counts the scraps
from hand they owe right now (Machine Base: "draw a card, then scrap a card
from your hand"), which they make before anything else.

`may_destroy` counts the opponent's bases the current player may still
destroy this turn (`DestroyBase` effects), like `may_scrap`: optional,
usable at any time in the turn, and lost at its end.

`may_scrap_row` counts the cards the current player may still scrap from
the Trade Row this turn (`ScrapTradeRow` effects), like `may_scrap`:
optional, usable at any time in the turn, and lost at its end.

`choosing` holds the cards whose "or" choice (a `Choice` effect in their
primary ability) the current player must make right now, before anything
else: the first is chosen first.

`recycling` is how many cards the current player may discard right now,
before anything else, to draw as many (Recycling Station's
`DiscardThenDraw`, once chosen).

`factions_played` counts the cards of each faction (in `Faction` order) the
current player has played this turn, whatever became of them since (Blob
World draws a card per Blob card played); it empties at the end of the turn.

`may_on_top` counts the ships the current player may still put on top of
their deck as they acquire them this turn (`NextShipOnTop`), instead of in
the discard pile, like `may_scrap`: optional, and lost at the end of the
turn.

`free_ships` counts the ships the current player must acquire right now,
before anything else, for free from the Trade Row onto their deck (Blob
Carrier's `AcquireShipFreeOnTop`).

`scrap_to_draw` is how many cards the current player may scrap right now
from hand and discard pile, before anything else, to draw as many (Brain
World's `ScrapThenDraw`).

`copying` counts the Stealth Needles the current player has just played
and must now make a copy of another ship in play, before anything else
(`CopyShip`). A Needle that copies a ship is, while in play, a card of its
own (`rules._copy`), turned back into Stealth Needle when it leaves play.

`PlayerState.to_discard` counts the discards an opponent's `OpponentDiscards`
effects have left a player to make at the start of their next turn. It is
public, and capped at the hand's size when their turn starts.

`PlayerState.known_top` counts the cards on top of a player's deck that both
players know (`deck[:known_top]`): ships put there when acquired
(`NextShipOnTop`, `AcquireShipFreeOnTop`). They stay known until drawn; the
deck is only shuffled once empty, so a shuffle never hides one.
"""

from dataclasses import dataclass

from minimax_games.games.star_realms.card import Card, Faction

STARTING_AUTHORITY = 50
HAND_SIZE = 5
FIRST_HAND_SIZE = 3
TRADE_ROW_SIZE = 5
NO_FACTIONS_PLAYED = (0,) * len(Faction)


@dataclass(frozen=True, slots=True)
class PlayerState:
    authority: int
    deck: tuple[Card, ...]
    hand: tuple[Card, ...] = ()
    in_play: tuple[Card, ...] = ()
    discard: tuple[Card, ...] = ()
    bases: tuple[Card, ...] = ()
    to_discard: int = 0
    known_top: int = 0

    def replace(
        self,
        *,
        authority: int | None = None,
        deck: tuple[Card, ...] | None = None,
        hand: tuple[Card, ...] | None = None,
        in_play: tuple[Card, ...] | None = None,
        discard: tuple[Card, ...] | None = None,
        bases: tuple[Card, ...] | None = None,
        to_discard: int | None = None,
        known_top: int | None = None,
    ) -> "PlayerState":
        """`dataclasses.replace(self, ...)`, without its introspection of the
        fields on every call (see `GameState.replace`)."""
        return PlayerState(
            self.authority if authority is None else authority,
            self.deck if deck is None else deck,
            self.hand if hand is None else hand,
            self.in_play if in_play is None else in_play,
            self.discard if discard is None else discard,
            self.bases if bases is None else bases,
            self.to_discard if to_discard is None else to_discard,
            self.known_top if known_top is None else known_top,
        )


@dataclass(frozen=True, slots=True)
class GameState:
    players: tuple[PlayerState, PlayerState]
    rng_seed: int
    trade_deck: tuple[Card, ...] = ()
    trade_row: tuple[Card, ...] = ()
    explorers: int = 0
    scrap_heap: tuple[Card, ...] = ()
    current: int = 0
    turn: int = 1
    trade: int = 0
    combat: int = 0
    allies_used: tuple[Card, ...] = ()
    bases_used: tuple[Card, ...] = ()
    may_scrap: int = 0
    must_scrap: int = 0
    may_destroy: int = 0
    may_scrap_row: int = 0
    choosing: tuple[Card, ...] = ()
    recycling: int = 0
    factions_played: tuple[int, ...] = NO_FACTIONS_PLAYED
    may_on_top: int = 0
    free_ships: int = 0
    scrap_to_draw: int = 0
    copying: int = 0

    def replace(
        self,
        *,
        players: tuple[PlayerState, PlayerState] | None = None,
        rng_seed: int | None = None,
        trade_deck: tuple[Card, ...] | None = None,
        trade_row: tuple[Card, ...] | None = None,
        explorers: int | None = None,
        scrap_heap: tuple[Card, ...] | None = None,
        current: int | None = None,
        turn: int | None = None,
        trade: int | None = None,
        combat: int | None = None,
        allies_used: tuple[Card, ...] | None = None,
        bases_used: tuple[Card, ...] | None = None,
        may_scrap: int | None = None,
        must_scrap: int | None = None,
        may_destroy: int | None = None,
        may_scrap_row: int | None = None,
        choosing: tuple[Card, ...] | None = None,
        recycling: int | None = None,
        factions_played: tuple[int, ...] | None = None,
        may_on_top: int | None = None,
        free_ships: int | None = None,
        scrap_to_draw: int | None = None,
        copying: int | None = None,
    ) -> "GameState":
        """`dataclasses.replace(self, ...)`, without its introspection of the
        fields on every call: the search builds several states per node, so
        `rules` builds them with this. Every field is a tuple or an int, so
        `None` means "unchanged"."""
        return GameState(
            self.players if players is None else players,
            self.rng_seed if rng_seed is None else rng_seed,
            self.trade_deck if trade_deck is None else trade_deck,
            self.trade_row if trade_row is None else trade_row,
            self.explorers if explorers is None else explorers,
            self.scrap_heap if scrap_heap is None else scrap_heap,
            self.current if current is None else current,
            self.turn if turn is None else turn,
            self.trade if trade is None else trade,
            self.combat if combat is None else combat,
            self.allies_used if allies_used is None else allies_used,
            self.bases_used if bases_used is None else bases_used,
            self.may_scrap if may_scrap is None else may_scrap,
            self.must_scrap if must_scrap is None else must_scrap,
            self.may_destroy if may_destroy is None else may_destroy,
            self.may_scrap_row if may_scrap_row is None else may_scrap_row,
            self.choosing if choosing is None else choosing,
            self.recycling if recycling is None else recycling,
            self.factions_played if factions_played is None else factions_played,
            self.may_on_top if may_on_top is None else may_on_top,
            self.free_ships if free_ships is None else free_ships,
            self.scrap_to_draw if scrap_to_draw is None else scrap_to_draw,
            self.copying if copying is None else copying,
        )
