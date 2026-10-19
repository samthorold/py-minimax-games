"""Observation views: what one player knows about a game.

A bot never sees the `GameState`, only `observe(state, player)`: everything
public, the observer's own hand, and the contents but not the order of the
hidden piles. Hidden piles are multisets, held as tuples sorted by card name,
so two states that differ only in hidden information give equal (and
hashable) views.

What is hidden:

- the order of every deck, the observer's own included, and of the Trade Deck,
  except for cards known to be on top (`PlayerView.deck_top`): ships put
  there when acquired, known to both players until drawn;
- the opponent's hand. The observer knows every card the opponent owns, so
  the opponent's hand and deck together are known as one multiset
  (`PlayerView.unseen`) with known sizes, but not which card is where;
- the seed for future shuffles.

For the observer, `unseen` is just their deck's contents. A sampler can build
a full world consistent with a view by dealing each player's `unseen` cards,
`hand_size` of them to the hand (none for the observer, whose `hand` is
known) and the rest to the deck under its known top cards, and shuffling
`trade_deck`.
"""

import random
from dataclasses import dataclass

from minimax_games.games.star_realms.card import Card
from minimax_games.games.star_realms.state import GameState, PlayerState


def _multiset(cards: tuple[Card, ...]) -> tuple[Card, ...]:
    """`cards` in a canonical order, forgetting their own."""
    return tuple(sorted(cards, key=lambda card: card.name))


@dataclass(frozen=True, slots=True)
class PlayerView:
    """One player as the observer sees them.

    `hand` holds the cards the observer can see in this player's hand: all of
    them for the observer, none for the opponent. `unseen` holds the rest of
    the hand and the whole deck, as a multiset. `hand_size` and `deck_size`
    are public, as are the discards the player has pending (`to_discard`)
    and the cards known to be on top of the deck (`deck_top`, top first,
    `PlayerState.known_top`), which `unseen` leaves out.
    """

    authority: int
    hand_size: int
    deck_size: int
    hand: tuple[Card, ...]
    unseen: tuple[Card, ...]
    in_play: tuple[Card, ...]
    discard: tuple[Card, ...]
    bases: tuple[Card, ...]
    to_discard: int
    deck_top: tuple[Card, ...]


@dataclass(frozen=True, slots=True)
class View:
    """What `observer` knows about the game. `players` is indexed like
    `GameState.players`; the turn state (whose turn, trade, combat, abilities
    used, scraps left and owed, bases left to destroy, Trade Row scraps
    left, the choices and recycling being made, the factions played, ships
    left to put on top of the deck and free ships owed) is all public."""

    observer: int
    players: tuple[PlayerView, PlayerView]
    trade_deck: tuple[Card, ...]
    trade_row: tuple[Card, ...]
    explorers: int
    scrap_heap: tuple[Card, ...]
    current: int
    turn: int
    trade: int
    combat: int
    allies_used: tuple[Card, ...]
    bases_used: tuple[Card, ...]
    may_scrap: int
    must_scrap: int
    may_destroy: int
    may_scrap_row: int
    choosing: tuple[Card, ...]
    recycling: int
    factions_played: tuple[int, ...]
    may_on_top: int
    free_ships: int


def _player_view(player: PlayerState, *, hand_seen: bool) -> PlayerView:
    deck_top, deck = player.deck[: player.known_top], player.deck[player.known_top :]
    if hand_seen:
        hand, unseen = player.hand, deck
    else:
        hand, unseen = (), player.hand + deck
    return PlayerView(
        authority=player.authority,
        hand_size=len(player.hand),
        deck_size=len(player.deck),
        hand=_multiset(hand),
        unseen=_multiset(unseen),
        in_play=player.in_play,
        discard=player.discard,
        bases=player.bases,
        to_discard=player.to_discard,
        deck_top=deck_top,
    )


def observe(state: GameState, observer: int) -> View:
    """What `observer` (0 or 1) knows about `state`."""
    first, second = (
        _player_view(player, hand_seen=index == observer)
        for index, player in enumerate(state.players)
    )
    return View(
        observer=observer,
        players=(first, second),
        trade_deck=_multiset(state.trade_deck),
        trade_row=state.trade_row,
        explorers=state.explorers,
        scrap_heap=state.scrap_heap,
        current=state.current,
        turn=state.turn,
        trade=state.trade,
        combat=state.combat,
        allies_used=state.allies_used,
        bases_used=state.bases_used,
        may_scrap=state.may_scrap,
        must_scrap=state.must_scrap,
        may_destroy=state.may_destroy,
        may_scrap_row=state.may_scrap_row,
        choosing=state.choosing,
        recycling=state.recycling,
        factions_played=state.factions_played,
        may_on_top=state.may_on_top,
        free_ships=state.free_ships,
    )


def _shuffled(cards: tuple[Card, ...], rng: random.Random) -> tuple[Card, ...]:
    return tuple(rng.sample(cards, len(cards)))


def _sample_player(player: PlayerView, rng: random.Random) -> PlayerState:
    unseen = _shuffled(player.unseen, rng)
    dealt = player.hand_size - len(player.hand)
    return PlayerState(
        authority=player.authority,
        deck=player.deck_top + unseen[dealt:],
        hand=player.hand + unseen[:dealt],
        in_play=player.in_play,
        discard=player.discard,
        bases=player.bases,
        to_discard=player.to_discard,
        known_top=len(player.deck_top),
    )


def sample_world(view: View, rng: random.Random) -> GameState:
    """A full state consistent with `view`, its hidden information drawn from
    `rng`: `observe(sample_world(view, rng), view.observer) == view`.

    Each player's `unseen` cards are shuffled and dealt, enough to fill the
    hand to `hand_size` (none for the observer, whose hand is known) and the
    rest to the deck. The Trade Deck is shuffled, and the seed for future
    shuffles is fresh from `rng`, in the range the rules draw seeds from.
    """
    first, second = (_sample_player(player, rng) for player in view.players)
    return GameState(
        players=(first, second),
        rng_seed=rng.getrandbits(64),
        trade_deck=_shuffled(view.trade_deck, rng),
        trade_row=view.trade_row,
        explorers=view.explorers,
        scrap_heap=view.scrap_heap,
        current=view.current,
        turn=view.turn,
        trade=view.trade,
        combat=view.combat,
        allies_used=view.allies_used,
        bases_used=view.bases_used,
        may_scrap=view.may_scrap,
        must_scrap=view.must_scrap,
        may_destroy=view.may_destroy,
        may_scrap_row=view.may_scrap_row,
        choosing=view.choosing,
        recycling=view.recycling,
        factions_played=view.factions_played,
        may_on_top=view.may_on_top,
        free_ships=view.free_ships,
    )
