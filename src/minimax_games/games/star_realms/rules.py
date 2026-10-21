"""The rules: dealing, legal actions and applying them."""

import functools
import itertools
import random
from collections.abc import Callable, Iterator
from typing import assert_never

from minimax_games.games.star_realms.action import (
    Acquire,
    AcquireFreeShip,
    AcquireOnTop,
    Action,
    AttackBase,
    AttackPlayer,
    Choose,
    Copy,
    DestroyTarget,
    Discard,
    EndTurn,
    PlayCard,
    Recycle,
    Scrap,
    ScrapAndDraw,
    ScrapDiscardCard,
    ScrapHandCard,
    ScrapTradeRowCard,
    UseAlly,
    UseBase,
)
from minimax_games.games.star_realms.card import Card, CardType, Faction
from minimax_games.games.star_realms.catalogue import (
    CATALOGUE,
    EXPLORER,
    EXPLORER_PILE,
    PLAYABLE_TRADE_DECK,
    STARTING_DECK,
)
from minimax_games.games.star_realms.effect import (
    AcquireShipFreeOnTop,
    AllyForAllFactions,
    Authority,
    Choice,
    Combat,
    CopyShip,
    DestroyBase,
    DiscardThenDraw,
    Draw,
    DrawIfBases,
    DrawPerFactionCard,
    Effect,
    NextShipOnTop,
    OpponentDiscards,
    ScrapFromHand,
    ScrapHandOrDiscard,
    ScrapThenDraw,
    ScrapTradeRow,
    ShipsGainCombat,
    Trade,
)
from minimax_games.games.star_realms.state import (
    FIRST_HAND_SIZE,
    HAND_SIZE,
    NO_FACTIONS_PLAYED,
    STARTING_AUTHORITY,
    TRADE_ROW_SIZE,
    GameState,
    PlayerState,
)


def _shuffle(cards: tuple[Card, ...], seed: int) -> tuple[tuple[Card, ...], int]:
    """Return `cards` shuffled, and the seed for the next shuffle."""
    rng = random.Random(seed)
    shuffled = tuple(rng.sample(cards, len(cards)))
    return shuffled, rng.getrandbits(64)


def _draw(player: PlayerState, n: int, seed: int) -> tuple[PlayerState, int]:
    """Draw `n` cards, shuffling the discard pile into the deck if it runs out."""
    deck, hand, discard = player.deck, player.hand, player.discard
    for _ in range(n):
        if not deck:
            if not discard:
                break
            deck, seed = _shuffle(discard, seed)
            discard = ()
        hand, deck = (*hand, deck[0]), deck[1:]
    # The known top cards are drawn first (`PlayerState.known_top`).
    known_top = max(player.known_top - n, 0)
    return player.replace(
        deck=deck, hand=hand, discard=discard, known_top=known_top
    ), seed


def new_game(
    seed: int,
    trade_deck: tuple[Card, ...] = PLAYABLE_TRADE_DECK,
) -> GameState:
    """Deal a game from `seed`.

    `trade_deck` is shuffled and its top 5 cards revealed as the Trade Row,
    beside a pile of 10 Explorers. Each player gets a shuffled starting deck
    and 50 Authority. The first player draws 3 cards and the second 5.
    """
    shuffled, seed = _shuffle(trade_deck, seed)
    trade_row, rest = shuffled[:TRADE_ROW_SIZE], shuffled[TRADE_ROW_SIZE:]
    players = []
    for hand_size in (FIRST_HAND_SIZE, HAND_SIZE):
        deck, seed = _shuffle(STARTING_DECK, seed)
        player, seed = _draw(PlayerState(STARTING_AUTHORITY, deck), hand_size, seed)
        players.append(player)
    first, second = players
    return GameState(
        players=(first, second),
        rng_seed=seed,
        trade_deck=rest,
        trade_row=trade_row,
        explorers=EXPLORER_PILE,
    )


def _without(cards: tuple[Card, ...], card: Card) -> tuple[Card, ...]:
    """`cards` with one copy of `card` (which it holds) removed."""
    i = cards.index(card)
    return cards[:i] + cards[i + 1 :]


def _players(
    state: GameState, index: int, player: PlayerState
) -> tuple[PlayerState, PlayerState]:
    """`state.players` with `player` in place of player `index`."""
    first, second = state.players
    return (player, second) if index == 0 else (first, player)


_FACTION_INDEX = {faction: i for i, faction in enumerate(Faction)}
"""Each faction's place in `GameState.factions_played`."""


def _played(state: GameState, card: Card) -> tuple[int, ...]:
    """`state.factions_played` once `card` is played too."""
    played = state.factions_played
    if card.faction is None:
        return played
    i = _FACTION_INDEX[card.faction]
    return (*played[:i], played[i] + 1, *played[i + 1 :])


def _resolve(state: GameState, effect: Effect, card: Card) -> GameState:
    """`state` after `effect`, one of `card`'s, applies."""
    match effect:
        case Trade(amount):
            return state.replace(trade=state.trade + amount)
        case Combat(amount):
            return state.replace(combat=state.combat + amount)
        case Authority(amount):
            player = state.players[state.current]
            player = player.replace(authority=player.authority + amount)
            return state.replace(players=_players(state, state.current, player))
        case Draw(cards):
            player, seed = _draw(state.players[state.current], cards, state.rng_seed)
            players = _players(state, state.current, player)
            return state.replace(players=players, rng_seed=seed)
        case OpponentDiscards(cards):
            them = 1 - state.current
            opponent = state.players[them]
            opponent = opponent.replace(to_discard=opponent.to_discard + cards)
            return state.replace(players=_players(state, them, opponent))
        case ScrapHandOrDiscard(cards):
            return state.replace(may_scrap=state.may_scrap + cards)
        case DestroyBase():
            # The opponent gains no bases during this turn, so with none now
            # there is nothing to destroy.
            if not state.players[1 - state.current].bases:
                return state
            return state.replace(may_destroy=state.may_destroy + 1)
        case ScrapTradeRow(cards):
            # The row only empties once the Trade Deck has run out, and then
            # never refills: with it empty there is nothing to scrap.
            if not state.trade_row:
                return state
            return state.replace(may_scrap_row=state.may_scrap_row + cards)
        case ScrapFromHand():
            # Machine Base draws first, so the hand is empty only when the
            # player has no cards left to draw: then there is nothing to scrap.
            if not state.players[state.current].hand:
                return state
            return state.replace(must_scrap=state.must_scrap + 1)
        case DrawPerFactionCard():
            if card.faction is None:
                return state
            cards = state.factions_played[_FACTION_INDEX[card.faction]]
            return _resolve(state, Draw(cards), card)
        case DiscardThenDraw(cards):
            # With an empty hand there is nothing to discard, so nothing to draw.
            if not state.players[state.current].hand:
                return state
            return state.replace(recycling=cards)
        case NextShipOnTop():
            return state.replace(may_on_top=state.may_on_top + 1)
        case Choice():
            # The player chooses an option before anything else (`_forced`).
            return state.replace(choosing=(*state.choosing, card))
        case AcquireShipFreeOnTop():
            # With no ship in the row there is nothing to acquire.
            if not any(c.type is CardType.SHIP for c in state.trade_row):
                return state
            return state.replace(free_ships=state.free_ships + 1)
        case DrawIfBases(bases, cards):
            if len(state.players[state.current].bases) < bases:
                return state
            return _resolve(state, Draw(cards), card)
        case AllyForAllFactions() | ShipsGainCombat():
            # Always on while the base is in play (`_ship_bonus`, `_allied`):
            # nothing happens when it is played.
            return state
        case ScrapThenDraw(cards):
            # With nothing to scrap there is nothing to draw.
            player = state.players[state.current]
            if not player.hand and not player.discard:
                return state
            return state.replace(scrap_to_draw=cards)
        case CopyShip():
            # With no other ship in play there is nothing to copy.
            if not _copy_targets(state.players[state.current].in_play):
                return state
            return state.replace(copying=state.copying + 1)
        case _:
            assert_never(effect)


@functools.cache
def _options(card: Card) -> tuple[tuple[Effect, ...], ...]:
    """The options of `card`'s "or" choice, its one `Choice` effect (cached)."""
    choice = next(
        e for e in card.primary + card.ally + card.scrap if isinstance(e, Choice)
    )
    return choice.options


@functools.cache
def _primary_usable(card: Card) -> bool:
    """Whether `card` has a primary ability to use: not none, nor one that is
    always on (Fleet HQ, Mech World). Cached by card, which hashes by its
    name."""
    return bool(card.primary) and not all(
        isinstance(e, _ALWAYS_ON) for e in card.primary
    )


_ALWAYS_ON = AllyForAllFactions | ShipsGainCombat
"""The effects that work while their base is in play, without being used."""


@functools.cache
def _ship_bonus(card: Card) -> int:
    """The combat each ship played gains while `card` is in play (Fleet HQ's
    `ShipsGainCombat`), cached."""
    return sum(e.amount for e in card.primary if isinstance(e, ShipsGainCombat))


def _allied(card: Card, in_play: tuple[Card, ...]) -> bool:
    """Whether another card that is an ally for `card`'s faction is among
    `in_play`, the cards in play (`card` among them): one of that faction,
    or one that is an ally for other factions too (`_OTHER_FACTIONS`: Mech
    World, and Stealth Needle copying a ship of another faction). A
    scrapped card is no longer in play, so no longer allies anything."""
    faction = card.faction
    if faction is None:
        return False
    allies = 0
    for c in in_play:
        if c.faction is faction or (
            c.name in _OTHER_FACTIONS and faction in _OTHER_FACTIONS[c.name]
        ):
            allies += 1
    return allies >= 2


_OTHER_FACTIONS: dict[str, frozenset[Faction]] = {
    card.name: frozenset(Faction) - {card.faction}
    for card in CATALOGUE.values()
    if any(isinstance(e, AllyForAllFactions) for e in card.primary)
}
"""The cards that are allies for factions besides their own, by name (a
`str` hashes faster than a `Card`), and those factions: Mech World
(`AllyForAllFactions`), and each copy Stealth Needle makes of a ship of
another faction (added by `_copy`)."""

_NEEDLES: dict[Card, Card] = {}
"""Each copy Stealth Needle has made (`_copy`), mapped to the Needle."""


@functools.cache
def _copy(needle: Card, target: Card) -> Card:
    """`needle` (Stealth Needle) as a copy of `target`: a ship with
    `target`'s abilities and faction (the Needle's if it has none), named
    for both, which `original` turns back into the Needle. Cached, so each
    copy is made once."""
    copy = Card(
        name=f"{needle.name} ({target.name})",
        cost=needle.cost,
        faction=target.faction or needle.faction,
        type=CardType.SHIP,
        primary=target.primary,
        ally=target.ally,
        scrap=target.scrap,
        copies=needle.copies,
    )
    _NEEDLES[copy] = needle
    if needle.faction is not None and needle.faction is not copy.faction:
        _OTHER_FACTIONS[copy.name] = frozenset((needle.faction,))
    return copy


def original(card: Card) -> Card:
    """The card `card` is in every pile but in play: Stealth Needle for a copy
    it has made (`_copy`), else `card` itself."""
    return _NEEDLES.get(card, card)


@functools.cache
def _copies_ships(card: Card) -> bool:
    """Whether `card`'s primary ability copies a ship (Stealth Needle)."""
    return any(isinstance(e, CopyShip) for e in card.primary)


def _copy_targets(in_play: tuple[Card, ...]) -> tuple[Card, ...]:
    """The distinct ships among `in_play` that Stealth Needle may copy: any
    but itself (or a copy it has made)."""
    return tuple(
        card
        for card in dict.fromkeys(in_play)
        if not _copies_ships(card) and card not in _NEEDLES
    )


def winner(state: GameState) -> int | None:
    """The index of the player who has won, or `None` while the game goes on."""
    for i, player in enumerate(state.players):
        if player.authority <= 0:
            return 1 - i
    return None


def legal_actions(state: GameState) -> tuple[Action, ...]:
    """The actions open to the current player.

    Cards in play include the player's bases. One `PlayCard` per distinct
    card in hand; one `UseAlly` per distinct card in play with an ally
    ability, another card in play that is an ally for its faction
    (`_allied`), and a copy whose ally ability is unused this turn; one
    `UseBase` per distinct base with a primary ability that isn't always on
    and a copy whose primary is unused this turn; one `Scrap` per distinct
    card in play with a scrap ability; while the player may still scrap from
    hand or discard pile (`GameState.may_scrap`), one `ScrapHandCard` per
    distinct card in hand and one `ScrapDiscardCard` per distinct card in
    the discard pile; while the player may still scrap from the Trade Row
    (`GameState.may_scrap_row`), one `ScrapTradeRowCard` per distinct card
    in the row; one `Acquire` per distinct affordable card
    in the Trade Row or on the Explorer pile, and, while the player may
    still put a ship on top of their deck (`GameState.may_on_top`), one
    `AcquireOnTop` per affordable ship among them; while the player may still
    destroy a base (`GameState.may_destroy`), one `DestroyTarget` per
    distinct opponent's base, Outposts or not; one `AttackBase` per distinct
    opponent's base whose defense the combat covers, only Outposts while the
    opponent has one; `AttackPlayer` while there is combat to spend and the
    opponent has no Outpost; and `EndTurn`. Empty once the game is over.

    A decision the player must make before anything else replaces all of
    those (`_forced`): while discards are pending at the start of their turn,
    one `Discard` per distinct card in hand; while a scrap from hand is
    owed (Machine Base), one `ScrapHandCard` per distinct card in hand;
    while recycling (Recycling Station), one `Recycle` per distinct set of
    cards; while a free ship is owed (Blob Carrier), one `AcquireFreeShip`
    per distinct ship in the Trade Row; while Brain World's scraps are
    being made, one `ScrapAndDraw` per distinct choice of cards; while
    Stealth Needle's copy is being made, one `Copy` per distinct other ship
    in play; and while an "or" choice is being made, one `Choose` per
    option.
    """
    if winner(state) is not None:
        return ()
    if (forced := _forced(state)) is not None:
        return forced
    player = state.players[state.current]
    actions: list[Action] = list(_maybe_automatic(state))
    in_play = player.in_play + player.bases
    actions.extend(Scrap(card) for card in dict.fromkeys(in_play) if card.scrap)
    if state.may_scrap:
        actions.extend(ScrapHandCard(card) for card in dict.fromkeys(player.hand))
        actions.extend(ScrapDiscardCard(card) for card in dict.fromkeys(player.discard))
    if state.may_scrap_row:
        actions.extend(
            ScrapTradeRowCard(card) for card in dict.fromkeys(state.trade_row)
        )
    market = state.trade_row + ((EXPLORER,) if state.explorers else ())
    affordable = [card for card in dict.fromkeys(market) if card.cost <= state.trade]
    actions.extend(Acquire(card) for card in affordable)
    if state.may_on_top:
        actions.extend(
            AcquireOnTop(card) for card in affordable if card.type is CardType.SHIP
        )
    targets = state.players[1 - state.current].bases
    if state.may_destroy:
        actions.extend(DestroyTarget(card) for card in dict.fromkeys(targets))
    outposts = tuple(c for c in targets if c.type is CardType.OUTPOST)
    actions.extend(
        AttackBase(card)
        for card in dict.fromkeys(outposts or targets)
        if (card.defense or 0) <= state.combat
    )
    if state.combat > 0 and not outposts:
        actions.append(AttackPlayer())
    actions.append(EndTurn())
    return tuple(actions)


def _forced(state: GameState) -> tuple[Action, ...] | None:
    """The actions of a decision the current player must make before any
    other, if one is pending, else `None`.

    A pending discard (`PlayerState.to_discard`), one `Discard`
    per distinct card in hand (`EndTurn` caps the count at the hand's size,
    so while it is positive the hand isn't empty); an owed scrap from
    hand (`GameState.must_scrap`, Machine Base), one `ScrapHandCard` per
    distinct card in hand (owed only with a card in hand, and nothing else
    can happen until it is made); recycling (`GameState.recycling`,
    Recycling Station's `DiscardThenDraw` once chosen, only with a card in
    hand), one `Recycle` per distinct set of one card up to that many from
    hand; a free ship (`GameState.free_ships`, Blob Carrier's
    `AcquireShipFreeOnTop`, owed only while the row holds a ship), one
    `AcquireFreeShip` per distinct ship in the row; Brain World's scraps
    (`GameState.scrap_to_draw`, its `ScrapThenDraw`, only with a card in
    hand or discard pile), one `ScrapAndDraw` per distinct choice of up to
    that many cards from both, none included; Stealth Needle's copy
    (`GameState.copying`, its `CopyShip`, only with another ship in play),
    one `Copy` per distinct other ship in play; and an "or" choice
    (`GameState.choosing`), one `Choose` per option of the first card's
    `Choice`, in the card's order.
    """
    player = state.players[state.current]
    if player.to_discard:
        return tuple(Discard(card) for card in dict.fromkeys(player.hand))
    if state.must_scrap:
        return tuple(ScrapHandCard(card) for card in dict.fromkeys(player.hand))
    if state.recycling:
        return tuple(
            Recycle(cards) for cards in _discard_sets(player.hand, state.recycling)
        )
    if state.scrap_to_draw:
        return tuple(
            ScrapAndDraw(hand, discard)
            for hand, discard in _scrap_sets(
                player.hand, player.discard, state.scrap_to_draw
            )
        )
    if state.copying:
        return tuple(Copy(card) for card in _copy_targets(player.in_play))
    if state.free_ships:
        return tuple(
            AcquireFreeShip(card)
            for card in dict.fromkeys(state.trade_row)
            if card.type is CardType.SHIP
        )
    if state.choosing:
        card = state.choosing[0]
        return tuple(Choose(card, option) for option in _options(card))
    return None


def _discard_sets(hand: tuple[Card, ...], most: int) -> list[tuple[Card, ...]]:
    """Each distinct set of one card up to `most` cards from `hand`, its cards
    sorted by name."""
    cards = sorted(hand, key=lambda card: card.name)
    return list(
        dict.fromkeys(
            combo
            for size in range(1, min(most, len(cards)) + 1)
            for combo in itertools.combinations(cards, size)
        )
    )


def _scrap_sets(
    hand: tuple[Card, ...], discard: tuple[Card, ...], most: int
) -> list[tuple[tuple[Card, ...], tuple[Card, ...]]]:
    """Each distinct choice of up to `most` cards from `hand` and `discard`
    together, none included: the cards from each, sorted by name."""

    def subsets(cards: tuple[Card, ...], size: int) -> list[tuple[Card, ...]]:
        ordered = sorted(cards, key=lambda card: card.name)
        return list(dict.fromkeys(itertools.combinations(ordered, size)))

    return [
        (from_hand, from_discard)
        for total in range(min(most, len(hand) + len(discard)) + 1)
        for in_hand in range(total + 1)
        for from_hand in subsets(hand, in_hand)
        for from_discard in subsets(discard, total - in_hand)
    ]


def _maybe_automatic(state: GameState) -> Iterator[Action]:
    """The legal `PlayCard`, `UseAlly` and `UseBase` actions, the only kinds
    that can be automatic, in `legal_actions` order (they come first there),
    generated lazily so the automatic-play scan can stop early. Assumes the
    game isn't over."""
    player = state.players[state.current]
    for card in dict.fromkeys(player.hand):
        yield PlayCard(card)
    in_play = player.in_play + player.bases
    for card in dict.fromkeys(in_play):
        if (
            card.ally
            and _allied(card, in_play)
            and in_play.count(card) > state.allies_used.count(card)
        ):
            yield UseAlly(card)
    for card in dict.fromkeys(player.bases):
        if _primary_usable(card) and player.bases.count(card) > state.bases_used.count(
            card
        ):
            yield UseBase(card)


def _pure_gain(effects: tuple[Effect, ...]) -> bool:
    """Whether `effects` only add trade, combat or Authority, make the
    opponent discard, or let the player scrap from hand or discard pile or
    from the Trade Row, destroy one of the opponent's bases or put a ship
    they acquire on top of their deck, or are "or" choices between options
    that do only that."""
    return all(
        isinstance(
            e,
            Trade
            | Combat
            | Authority
            | OpponentDiscards
            | ScrapHandOrDiscard
            | DestroyBase
            | ScrapTradeRow
            | NextShipOnTop,
        )
        or (isinstance(e, Choice) and all(map(_pure_gain, e.options)))
        for e in effects
    )


@functools.cache
def _pure_gain_ship(card: Card) -> bool:
    """Whether `card` is a ship whose primary effects are pure gain. Cached by
    card, which hashes by its name: cheaper than rescanning its effects."""
    return card.type is CardType.SHIP and _pure_gain(card.primary)


@functools.cache
def _bases_to_draw(card: Card) -> int | None:
    """For a ship whose primary effects are pure gain but for drawing with
    enough bases in play (Embassy Yacht's `DrawIfBases`), the fewest bases
    that draw; else `None` (cached)."""
    if card.type is not CardType.SHIP:
        return None
    thresholds = [e.bases for e in card.primary if isinstance(e, DrawIfBases)]
    rest = tuple(e for e in card.primary if not isinstance(e, DrawIfBases))
    if not thresholds or not _pure_gain(rest):
        return None
    return min(thresholds)


def _cannot_draw(state: GameState, card: Card) -> bool:
    """Whether `card` is a ship that would be pure gain but for a draw with
    enough bases in play (`_bases_to_draw`), and the current player can't
    have that many this turn: the bases in play and in hand fall short.
    (Like scrappers, bases a later draw might bring aren't counted.)"""
    bases = _bases_to_draw(card)
    if bases is None:
        return False
    player = state.players[state.current]
    in_hand = sum(c.type is not CardType.SHIP for c in player.hand)
    return len(player.bases) + in_hand < bases


@functools.cache
def _pure_gain_ally(card: Card) -> bool:
    """Whether `card`'s ally effects are pure gain (cached)."""
    return _pure_gain(card.ally)


@functools.cache
def _pure_gain_primary(card: Card) -> bool:
    """Whether `card`'s primary effects are pure gain (cached)."""
    return _pure_gain(card.primary)


def automatic(state: GameState, action: Action) -> bool:
    """Whether `action`, legal in `state`, is applied automatically, never
    offered as a decision.

    The rule for conservative automatic play (docs/star-realms.md): an action
    is automatic when taking it now can't make any later option worse.

    - `PlayCard` of a **ship** whose primary effects are pure gain (trade,
      combat, Authority, making the opponent discard: that only adds to the
      discards they make at the start of their next turn, so nothing this
      turn depends on when it happens; or letting the player scrap a card
      from hand or discard pile or from the Trade Row, or destroy one of
      the opponent's bases, or put a ship they acquire on top of their
      deck, which they may do at any time in the turn: the opponent's bases
      and the Trade Row only change when the player destroys, scraps or
      buys, and a `DestroyBase` with no base to target or a
      `ScrapTradeRow` with an empty row is lost either way),
      unless something could scrap it from hand this turn instead
      (`_hand_scrap_ahead`). A ship in hand and a ship in play both go to
      the discard pile at the end of the turn; in play it only adds options
      (ally abilities, scrapping).
    - `UseAlly` whose ally effects are pure gain, and `UseBase` whose primary
      effects are pure gain: the card stays where it is, and the gain can't
      be lost by taking it early.

    An "or" choice between pure-gain options counts as pure gain: using the
    ability is automatic (Patrol Mech, and Trading Post, Barter World and
    Defense Center on the turns after they are played), and the `Choose`
    it leaves is the decision. Choosing an option early or late gives the
    same gain, so every outcome stays reachable; what is lost is only
    information a view-based bot would have had later in the turn (a
    draw), the accepted price of not branching on when to use it.

    Everything else is a decision. `Draw` never is automatic: drawing can
    reshuffle the discard pile, so drawing before or after an `Acquire` (the
    acquired card goes to the discard pile) changes the deck. Playing a base
    is a decision too: a base left in hand goes to the discard pile, so it
    changes what is reshuffled. `Scrap` loses the card, and `Acquire`,
    attacks and `EndTurn` are the turn's real choices. `Discard`,
    `ScrapHandCard`, `ScrapDiscardCard` and `ScrapTradeRowCard` are the
    player's choice of card, as are `AcquireOnTop` and `AcquireFreeShip`
    (Blob Carrier's ally ability is a decision too: the row it takes from
    changes as the player buys and scraps), `DestroyTarget` their choice of the
    opponent's base, `Choose` their choice of option and `Recycle` of the
    cards to discard. A choice with a draw among its options (Blob World,
    Recycling Station) isn't pure gain, so using it is a decision too.

    Embassy Yacht's draw with two bases (`DrawIfBases`) makes it a
    decision, unless the player can't have that many bases this turn (in
    play and in hand): then it is pure gain (`_cannot_draw`). Ships wait in
    hand while a Fleet HQ is there too (`_bonus_ahead`): played after it,
    they gain combat. Stealth Needle (`CopyShip`) and Brain World
    (`ScrapThenDraw`) are decisions, as are `Copy` and `ScrapAndDraw`; Mech
    World and Fleet HQ have nothing to use once played.

    This is sound for every base-set effect, with accepted gaps: a draw may
    bring a card that scraps from hand (or a base, for Embassy Yacht, or a
    Fleet HQ), and a ship already played can't be scrapped by it (or
    can't use it). Every new effect or action must revisit it.
    """
    match action:
        case PlayCard(card):
            if not _pure_gain_ship(card) and not _cannot_draw(state, card):
                return False
            return not _bonus_ahead(state) and not _hand_scrap_ahead(state, card)
        case UseAlly(card):
            return _pure_gain_ally(card)
        case UseBase(card):
            return _pure_gain_primary(card)
        case (
            Scrap()
            | Acquire()
            | AcquireOnTop()
            | AcquireFreeShip()
            | AttackPlayer()
            | AttackBase()
            | DestroyTarget()
            | Discard()
            | ScrapHandCard()
            | ScrapDiscardCard()
            | ScrapTradeRowCard()
            | Choose()
            | Copy()
            | Recycle()
            | ScrapAndDraw()
            | EndTurn()
        ):
            return False
        case _:
            assert_never(action)


def _bonus_ahead(state: GameState) -> bool:
    """Whether a ship the current player plays later this turn could gain
    combat it wouldn't now: a Fleet HQ (`ShipsGainCombat`) is in hand."""
    return any(card.name in _BOOSTERS for card in state.players[state.current].hand)


_BOOSTERS = frozenset(card.name for card in CATALOGUE.values() if _ship_bonus(card))
"""The names of the cards whose ships played gain combat (Fleet HQ): a
name is quicker to look up than a `Card`, whose hash is a Python call."""

_COPIERS = frozenset(card.name for card in CATALOGUE.values() if _copies_ships(card))
"""The names of the cards that copy a ship (Stealth Needle)."""


def _scraps_hand(effects: tuple[Effect, ...]) -> bool:
    """Whether `effects` include scrapping a card from hand."""
    return any(
        isinstance(e, ScrapHandOrDiscard | ScrapFromHand | ScrapThenDraw)
        for e in effects
    )


@functools.cache
def _primary_scraps_hand(card: Card) -> bool:
    """Whether `card`'s primary ability scraps a card from hand. Cached by
    card, which hashes by its name: cheaper than rescanning its effects."""
    return _scraps_hand(card.primary)


@functools.cache
def _ally_scraps_hand(card: Card) -> bool:
    """Whether `card`'s ally ability scraps a card from hand (cached)."""
    return _scraps_hand(card.ally)


def _hand_scrap_ahead(state: GameState, card: Card) -> bool:
    """Whether the current player could scrap `card`, in hand, from hand this
    turn if they didn't play it: they have a scrap from hand or discard pile
    left (`may_scrap`), another card in hand scraps from hand when played,
    or a card in play has an unused primary (bases) or ally ability that
    does, or Stealth Needle in hand could copy a ship in play whose primary
    or ally ability does. Conservative: it doesn't check that an ally
    ability is allied. Cards a later draw might bring aren't counted (see
    `automatic`)."""
    if state.may_scrap:
        return True
    player = state.players[state.current]
    scrappers = sum(map(_primary_scraps_hand, player.hand))
    if scrappers > _primary_scraps_hand(card):
        return True
    for base in player.bases:
        if _primary_scraps_hand(base) and player.bases.count(
            base
        ) > state.bases_used.count(base):
            return True
    for other in player.in_play + player.bases:
        if _ally_scraps_hand(other) and (player.in_play + player.bases).count(
            other
        ) > state.allies_used.count(other):
            return True
    for other in player.hand:
        if other.name in _COPIERS:
            return any(
                _primary_scraps_hand(ship) or _ally_scraps_hand(ship)
                for ship in player.in_play
            )
    return False


type OnAction = Callable[[GameState, Action], None]
"""Called with each automatic action and the state it is taken in."""


def auto_play(state: GameState, on_action: OnAction | None = None) -> GameState:
    """Apply automatic actions, in `legal_actions` order, until none is legal.

    The order doesn't change the outcome, but a fixed one keeps games and
    their logs reproducible. `on_action`, if given, is told of each one
    before it is applied (the match runner logs them through it).
    """
    while (action := _next_automatic(state)) is not None:
        if on_action is not None:
            on_action(state, action)
        state = _apply(state, action)
    return state


def _next_automatic(state: GameState) -> Action | None:
    """The first automatic action in `legal_actions(state)`, if any. It scans
    only the kinds of action that can be automatic, which `legal_actions`
    lists first, instead of building the whole list."""
    if winner(state) is not None or _forced(state) is not None:
        return None
    return next((a for a in _maybe_automatic(state) if automatic(state, a)), None)


def advance(
    state: GameState, action: Action, on_action: OnAction | None = None
) -> GameState:
    """Apply `action`, a decision, then any automatic play it opens up.

    This is the move generator bots and search share: from a state `advance`
    returns, `legal_actions` holds decisions only. `apply` is the bare rules,
    without automatic play.
    """
    return auto_play(apply(state, action), on_action)


def advance_legal(state: GameState, action: Action) -> GameState:
    """`advance` for an action taken from `legal_actions(state)`, as the search
    takes them: it skips `apply`'s legality check, which would recompute
    `legal_actions` just to find the action there. An action from anywhere
    else, such as a bot, goes through `advance`."""
    return auto_play(_apply(state, action))


def apply_legal(state: GameState, action: Action) -> GameState:
    """`apply` for an action taken from `legal_actions(state)`, without its
    legality check: the bare rules, no automatic play. The whole-turn planner
    takes `EndTurn` with it, so an end of turn is the state as `EndTurn`
    leaves it, before the next player's automatic play."""
    return _apply(state, action)


def apply(state: GameState, action: Action) -> GameState:
    """Return the state after the current player takes `action`.

    `state` itself is never changed. Raises `ValueError` if `action` isn't legal.
    """
    if action not in legal_actions(state):
        msg = f"{action} is not legal now"
        raise ValueError(msg)
    return _apply(state, action)


def _to_scrap_heap(state: GameState, card: Card) -> GameState:
    """`state` with `card`, taken from wherever it was, scrapped: on the scrap
    heap, or, for an Explorer, back on the Explorer pile."""
    if card == EXPLORER:
        return state.replace(explorers=state.explorers + 1)
    return state.replace(scrap_heap=(*state.scrap_heap, card))


def _take_from_row(state: GameState, card: Card) -> GameState:
    """`state` with `card` taken from the Trade Row and its place filled from
    the top of the Trade Deck, if any is left."""
    i = state.trade_row.index(card)
    return state.replace(
        trade_row=state.trade_row[:i] + state.trade_deck[:1] + state.trade_row[i + 1 :],
        trade_deck=state.trade_deck[1:],
    )


def _take_from_market(state: GameState, card: Card) -> GameState:
    """`state` with `card` taken from the Trade Row (`_take_from_row`) or,
    for an Explorer, from the Explorer pile."""
    if card == EXPLORER:
        return state.replace(explorers=state.explorers - 1)
    return _take_from_row(state, card)


def _on_top(state: GameState, card: Card) -> GameState:
    """`state` with `card`, which the current player acquires, on top of
    their deck, where both players know it (`PlayerState.known_top`)."""
    player = state.players[state.current]
    player = player.replace(deck=(card, *player.deck), known_top=player.known_top + 1)
    return state.replace(players=_players(state, state.current, player))


def _destroy(state: GameState, card: Card) -> GameState:
    """`state` with `card`, one of the opponent's bases, destroyed: in its
    owner's discard pile. Destroy-base abilities left over once the opponent
    has no bases are dropped (`may_destroy`), as nothing could use them."""
    them = 1 - state.current
    opponent = state.players[them]
    opponent = opponent.replace(
        bases=_without(opponent.bases, card),
        discard=(*opponent.discard, card),
    )
    players = _players(state, them, opponent)
    if not opponent.bases:
        return state.replace(players=players, may_destroy=0)
    return state.replace(players=players)


def _apply(state: GameState, action: Action) -> GameState:
    """`apply` for an action already known to be legal."""
    me = state.current
    player = state.players[me]
    match action:
        case PlayCard(card):
            hand = _without(player.hand, card)
            if card.type is CardType.SHIP:
                player = player.replace(hand=hand, in_play=(*player.in_play, card))
                state = state.replace(
                    players=_players(state, me, player),
                    factions_played=_played(state, card),
                    combat=state.combat + sum(map(_ship_bonus, player.bases)),
                )
            else:
                # Playing a base uses its primary ability for this turn.
                player = player.replace(hand=hand, bases=(*player.bases, card))
                state = state.replace(
                    players=_players(state, me, player),
                    bases_used=(*state.bases_used, card),
                    factions_played=_played(state, card),
                )
            for effect in card.primary:
                state = _resolve(state, effect, card)
            return state
        case UseAlly(card):
            state = state.replace(allies_used=(*state.allies_used, card))
            for effect in card.ally:
                state = _resolve(state, effect, card)
            return state
        case UseBase(card):
            state = state.replace(bases_used=(*state.bases_used, card))
            for effect in card.primary:
                state = _resolve(state, effect, card)
            return state
        case Scrap(card):
            if card.type is CardType.SHIP:
                player = player.replace(in_play=_without(player.in_play, card))
            else:
                player = player.replace(bases=_without(player.bases, card))
            # Of identical copies, scrap one whose primary (for a base) and
            # ally abilities were used, so an unused one keeps its abilities.
            bases_used, allies_used = state.bases_used, state.allies_used
            if card in bases_used:
                bases_used = _without(bases_used, card)
            if card in allies_used:
                allies_used = _without(allies_used, card)
            state = _to_scrap_heap(
                state.replace(
                    players=_players(state, me, player),
                    bases_used=bases_used,
                    allies_used=allies_used,
                ),
                original(card),
            )
            for effect in card.scrap:
                state = _resolve(state, effect, card)
            return state
        case Acquire(card):
            player = player.replace(discard=(*player.discard, card))
            state = state.replace(
                players=_players(state, me, player), trade=state.trade - card.cost
            )
            return _take_from_market(state, card)
        case AcquireOnTop(card):
            state = _on_top(state, card).replace(
                trade=state.trade - card.cost, may_on_top=state.may_on_top - 1
            )
            return _take_from_market(state, card)
        case AcquireFreeShip(card):
            state = _on_top(state, card).replace(free_ships=state.free_ships - 1)
            return _take_from_row(state, card)
        case AttackPlayer():
            opponent = state.players[1 - me]
            opponent = opponent.replace(authority=opponent.authority - state.combat)
            return state.replace(players=_players(state, 1 - me, opponent), combat=0)
        case AttackBase(card):
            state = _destroy(state, card)
            return state.replace(combat=state.combat - (card.defense or 0))
        case DestroyTarget(card):
            state = _destroy(state, card)
            return state.replace(may_destroy=max(state.may_destroy - 1, 0))
        case Discard(card):
            player = player.replace(
                hand=_without(player.hand, card),
                discard=(*player.discard, card),
                to_discard=player.to_discard - 1,
            )
            return state.replace(players=_players(state, me, player))
        case ScrapHandCard(card):
            player = player.replace(hand=_without(player.hand, card))
            players = _players(state, me, player)
            if state.must_scrap:
                state = state.replace(players=players, must_scrap=state.must_scrap - 1)
            else:
                state = state.replace(players=players, may_scrap=state.may_scrap - 1)
            return _to_scrap_heap(state, card)
        case ScrapDiscardCard(card):
            player = player.replace(discard=_without(player.discard, card))
            return _to_scrap_heap(
                state.replace(
                    players=_players(state, me, player),
                    may_scrap=state.may_scrap - 1,
                ),
                card,
            )
        case ScrapTradeRowCard(card):
            state = _to_scrap_heap(_take_from_row(state, card), card)
            if not state.trade_row:
                return state.replace(may_scrap_row=0)
            return state.replace(may_scrap_row=state.may_scrap_row - 1)
        case Choose(card, option):
            state = state.replace(choosing=state.choosing[1:])
            for effect in option:
                state = _resolve(state, effect, card)
            return state
        case Copy(target):
            in_play = player.in_play
            i = next(i for i, c in enumerate(in_play) if _copies_ships(c))
            copy = _copy(in_play[i], target)
            player = player.replace(in_play=(*in_play[:i], copy, *in_play[i + 1 :]))
            played = state.factions_played
            if target.faction is not None and target.faction is not in_play[i].faction:
                # A card of the target's faction too (Blob World counts it).
                played = _played(state, target)
            state = state.replace(
                players=_players(state, me, player),
                copying=state.copying - 1,
                factions_played=played,
            )
            for effect in copy.primary:
                state = _resolve(state, effect, copy)
            return state
        case Recycle(cards):
            hand = player.hand
            for card in cards:
                hand = _without(hand, card)
            player = player.replace(hand=hand, discard=player.discard + cards)
            player, seed = _draw(player, len(cards), state.rng_seed)
            return state.replace(
                players=_players(state, me, player), rng_seed=seed, recycling=0
            )
        case ScrapAndDraw(from_hand, from_discard):
            hand, discard = player.hand, player.discard
            for card in from_hand:
                hand = _without(hand, card)
            for card in from_discard:
                discard = _without(discard, card)
            scrapped = from_hand + from_discard
            player, seed = _draw(
                player.replace(hand=hand, discard=discard),
                len(scrapped),
                state.rng_seed,
            )
            state = state.replace(
                players=_players(state, me, player), rng_seed=seed, scrap_to_draw=0
            )
            for card in scrapped:
                state = _to_scrap_heap(state, card)
            return state
        case EndTurn():
            player = player.replace(
                hand=(),
                in_play=(),
                discard=player.discard
                + tuple(map(original, player.in_play))
                + player.hand,
            )
            player, seed = _draw(player, HAND_SIZE, state.rng_seed)
            # The next player discards no more cards than they hold.
            opponent = state.players[1 - me]
            if opponent.to_discard > len(opponent.hand):
                opponent = opponent.replace(to_discard=len(opponent.hand))
            players = (player, opponent) if me == 0 else (opponent, player)
            return state.replace(
                players=players,
                current=1 - me,
                turn=state.turn + 1,
                trade=0,
                combat=0,
                allies_used=(),
                bases_used=(),
                may_scrap=0,
                may_destroy=0,
                may_scrap_row=0,
                factions_played=NO_FACTIONS_PLAYED,
                may_on_top=0,
                rng_seed=seed,
            )
        case _:
            assert_never(action)
