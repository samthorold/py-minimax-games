"""The rules: dealing, legal actions and applying them."""

import functools
import random
from collections.abc import Callable, Iterator
from typing import assert_never

from minimax_games.games.star_realms.action import (
    Acquire,
    Action,
    AttackBase,
    AttackPlayer,
    DestroyTarget,
    Discard,
    EndTurn,
    PlayCard,
    Scrap,
    ScrapDiscardCard,
    ScrapHandCard,
    UseAlly,
    UseBase,
)
from minimax_games.games.star_realms.card import Card, CardType
from minimax_games.games.star_realms.catalogue import (
    EXPLORER,
    EXPLORER_PILE,
    IMPLEMENTED,
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
    return player.replace(deck=deck, hand=hand, discard=discard), seed


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


def _resolve(state: GameState, effect: Effect) -> GameState:
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
            return state.replace(players=players, rng_seed=seed, discard_scraps=())
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
        case ScrapFromHand():
            # Machine Base draws first, so the hand is empty only when the
            # player has no cards left to draw: then there is nothing to scrap.
            if not state.players[state.current].hand:
                return state
            return state.replace(must_scrap=state.must_scrap + 1)
        case (
            AcquireShipFreeOnTop()
            | AllyForAllFactions()
            | Choice()
            | CopyShip()
            | DiscardThenDraw()
            | DrawIfBases()
            | DrawPerFactionCard()
            | NextShipOnTop()
            | ScrapThenDraw()
            | ScrapTradeRow()
            | ShipsGainCombat()
        ):
            # Only playable cards (`catalogue.IMPLEMENTED`) are dealt; later
            # slices implement the rest (docs/star-realms.md).
            msg = f"{effect} is not implemented yet"
            raise NotImplementedError(msg)
        case _:
            assert_never(effect)


def _implemented(effect: Effect) -> bool:
    """Whether the rules implement `effect` (`catalogue.IMPLEMENTED`)."""
    return type(effect) in IMPLEMENTED


def _scrappable(card: Card) -> bool:
    """Whether `card` has a scrap ability whose effects are all implemented."""
    return bool(card.scrap) and all(_implemented(e) for e in card.scrap)


def _ally_usable(card: Card) -> bool:
    """Whether `card` has an ally ability whose effects are all implemented."""
    return bool(card.ally) and all(_implemented(e) for e in card.ally)


def _primary_usable(card: Card) -> bool:
    """Whether `card` has a primary ability whose effects are all implemented."""
    return bool(card.primary) and all(_implemented(e) for e in card.primary)


def _allied(card: Card, in_play: tuple[Card, ...]) -> bool:
    """Whether another card of `card`'s faction is among `in_play`, the cards
    in play (`card` among them). A scrapped card is no longer in play, so no
    longer allies anything."""
    if card.faction is None:
        return False
    return sum(c.faction is card.faction for c in in_play) >= 2


def winner(state: GameState) -> int | None:
    """The index of the player who has won, or `None` while the game goes on."""
    for i, player in enumerate(state.players):
        if player.authority <= 0:
            return 1 - i
    return None


def legal_actions(state: GameState) -> tuple[Action, ...]:
    """The actions open to the current player.

    Cards in play include the player's bases. One `PlayCard` per distinct
    card in hand; one `UseAlly` per distinct card in play with an ally ability
    the engine implements, another card of its faction in play, and a copy
    whose ally ability is unused this turn; one `UseBase` per distinct base
    with a primary ability the engine implements and a copy whose primary is
    unused this turn; one `Scrap` per distinct card in play with a scrap
    ability the engine implements; while the player may still scrap from
    hand or discard pile (`GameState.may_scrap`), one `ScrapHandCard` per
    distinct card in hand and one `ScrapDiscardCard` per distinct card in
    the discard pile; one `Acquire` per distinct affordable card
    in the Trade Row or on the Explorer pile; while the player may still
    destroy a base (`GameState.may_destroy`), one `DestroyTarget` per
    distinct opponent's base, Outposts or not; one `AttackBase` per distinct
    opponent's base whose defense the combat covers, only Outposts while the
    opponent has one; `AttackPlayer` while there is combat to spend and the
    opponent has no Outpost; and `EndTurn`. Empty once the game is over.

    Discard-pile scraps come last (`_after_discard_scrap`): once the player
    has made one, only automatic actions, draws that reshuffle the discard
    pile, more of them and `EndTurn` are legal, until they next draw.

    A decision the player must make before anything else replaces all of
    those (`_forced`): while discards are pending at the start of their turn,
    one `Discard` per distinct card in hand, and while a scrap from hand is
    owed (Machine Base), one `ScrapHandCard` per distinct card in hand.
    """
    if winner(state) is not None:
        return ()
    if (forced := _forced(state)) is not None:
        return forced
    if state.discard_scraps:
        return _after_discard_scrap(state)
    player = state.players[state.current]
    actions: list[Action] = list(_maybe_automatic(state))
    in_play = player.in_play + player.bases
    actions.extend(Scrap(card) for card in dict.fromkeys(in_play) if _scrappable(card))
    if state.may_scrap:
        actions.extend(ScrapHandCard(card) for card in dict.fromkeys(player.hand))
        actions.extend(ScrapDiscardCard(card) for card in dict.fromkeys(player.discard))
    market = state.trade_row + ((EXPLORER,) if state.explorers else ())
    actions.extend(
        Acquire(card) for card in dict.fromkeys(market) if card.cost <= state.trade
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

    Two so far: a pending discard (`PlayerState.to_discard`), one `Discard`
    per distinct card in hand (`EndTurn` caps the count at the hand's size,
    so while it is positive the hand isn't empty); and an owed scrap from
    hand (`GameState.must_scrap`, Machine Base), one `ScrapHandCard` per
    distinct card in hand (owed only with a card in hand, and nothing else
    can happen until it is made). Later effects that interrupt the turn
    with a decision belong here too.
    """
    player = state.players[state.current]
    if player.to_discard:
        return tuple(Discard(card) for card in dict.fromkeys(player.hand))
    if state.must_scrap:
        return tuple(ScrapHandCard(card) for card in dict.fromkeys(player.hand))
    return None


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
            _ally_usable(card)
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
    destroy one of the opponent's bases."""
    return all(
        isinstance(
            e,
            Trade
            | Combat
            | Authority
            | OpponentDiscards
            | ScrapHandOrDiscard
            | DestroyBase,
        )
        for e in effects
    )


@functools.cache
def _pure_gain_ship(card: Card) -> bool:
    """Whether `card` is a ship whose primary effects are pure gain. Cached by
    card, which hashes by its name: cheaper than rescanning its effects."""
    return card.type is CardType.SHIP and _pure_gain(card.primary)


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
      from hand or discard pile, or destroy one of the opponent's bases,
      which they may do at any time in the turn: the opponent's bases only
      change when the player destroys them, and a `DestroyBase` with no
      base to target is lost either way),
      unless something could scrap it from hand this turn instead
      (`_hand_scrap_ahead`). A ship in hand and a ship in play both go to
      the discard pile at the end of the turn; in play it only adds options
      (ally abilities, scrapping).
    - `UseAlly` whose ally effects are pure gain, and `UseBase` whose primary
      effects are pure gain: the card stays where it is, and the gain can't
      be lost by taking it early.

    Everything else is a decision. `Draw` never is automatic: drawing can
    reshuffle the discard pile, so drawing before or after an `Acquire` (the
    acquired card goes to the discard pile) changes the deck. Playing a base
    is a decision too: a base left in hand goes to the discard pile, so it
    changes what is reshuffled. `Scrap` loses the card, and `Acquire`,
    attacks and `EndTurn` are the turn's real choices. `Discard`,
    `ScrapHandCard` and `ScrapDiscardCard` are the player's choice of card,
    and `DestroyTarget` their choice of the opponent's base.

    This is sound only for the effects the engine implements so far, with
    one accepted gap: a draw may bring a card that scraps from hand, and a
    ship already played can't be scrapped by it. Every new effect or action
    must revisit it.
    """
    match action:
        case PlayCard(card):
            return _pure_gain_ship(card) and not _hand_scrap_ahead(state, card)
        case UseAlly(card):
            return _pure_gain_ally(card)
        case UseBase(card):
            return _pure_gain_primary(card)
        case (
            Scrap()
            | Acquire()
            | AttackPlayer()
            | AttackBase()
            | DestroyTarget()
            | Discard()
            | ScrapHandCard()
            | ScrapDiscardCard()
            | EndTurn()
        ):
            return False
        case _:
            assert_never(action)


def _after_discard_scrap(state: GameState) -> tuple[Action, ...]:
    """The legal actions once the player has scrapped from their discard
    pile (`GameState.discard_scraps`), until they next draw: automatic
    ones, those that draw more cards than the deck holds, so reshuffle the
    discard pile, more scraps from it of cards whose name sorts no earlier
    than the last one's, and `EndTurn`.

    The canonical timing of discard-pile scraps (docs/star-realms.md): they
    come last in the turn, or just before a draw that reshuffles the pile.
    Safe because a scrap from the discard pile commutes with every other
    action: nothing else reads the player's discard pile but a reshuffle.
    Acquired cards are added to it and stay there until the end of the
    turn, a destroyed base goes to the opponent's, and a draw from the deck
    alone leaves it be. So any turn that scraps from the discard pile and
    then does something else reaches the same end-of-turn outcome with the
    scrap moved later, up to the next reshuffling draw or the end of the
    turn; the scrap allowance (`may_scrap`) only grows until it is spent.
    Two scraps from the discard pile commute too, so they are made in
    order of card name. Hand scraps keep their timing: the heuristic scraps
    a Viper from hand before it would play it, and holding the scrap back
    keeps ships in hand waiting as decisions (`_hand_scrap_ahead`).

    Every new effect that draws, or reads the discard pile, must revisit it
    (`draws` counts `Draw` effects only).
    """
    player = state.players[state.current]
    deck = len(player.deck)
    actions: list[Action] = [
        a for a in _maybe_automatic(state) if automatic(state, a) or draws(a) > deck
    ]
    actions.extend(
        Scrap(card)
        for card in dict.fromkeys(player.in_play + player.bases)
        if _scrappable(card) and _scrap_draws(card) > deck
    )
    if state.may_scrap:
        last = state.discard_scraps[-1].name
        actions.extend(
            ScrapDiscardCard(card)
            for card in dict.fromkeys(player.discard)
            if card.name >= last
        )
    actions.append(EndTurn())
    return tuple(actions)


def _draw_count(effects: tuple[Effect, ...]) -> int:
    """How many cards `effects` draw."""
    return sum(e.cards for e in effects if isinstance(e, Draw))


@functools.cache
def _primary_draws(card: Card) -> int:
    """How many cards `card`'s primary ability draws (cached by card)."""
    return _draw_count(card.primary)


@functools.cache
def _ally_draws(card: Card) -> int:
    """How many cards `card`'s ally ability draws (cached)."""
    return _draw_count(card.ally)


@functools.cache
def _scrap_draws(card: Card) -> int:
    """How many cards `card`'s scrap ability draws (cached)."""
    return _draw_count(card.scrap)


def draws(action: Action) -> int:
    """How many cards taking `action` draws for the current player (before
    the end of the turn, which `EndTurn` doesn't count)."""
    match action:
        case PlayCard(card) | UseBase(card):
            return _primary_draws(card)
        case UseAlly(card):
            return _ally_draws(card)
        case Scrap(card):
            return _scrap_draws(card)
        case (
            Acquire()
            | AttackPlayer()
            | AttackBase()
            | DestroyTarget()
            | Discard()
            | ScrapHandCard()
            | ScrapDiscardCard()
            | EndTurn()
        ):
            return 0
        case _:
            assert_never(action)


def _scraps_hand(effects: tuple[Effect, ...]) -> bool:
    """Whether `effects` include scrapping a card from hand."""
    return any(isinstance(e, ScrapHandOrDiscard | ScrapFromHand) for e in effects)


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
    does. Conservative: it doesn't check that an ally ability is allied.
    Cards a later draw might bring aren't counted (see `automatic`)."""
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
                state = state.replace(players=_players(state, me, player))
            else:
                # Playing a base uses its primary ability for this turn.
                player = player.replace(hand=hand, bases=(*player.bases, card))
                state = state.replace(
                    players=_players(state, me, player),
                    bases_used=(*state.bases_used, card),
                )
            for effect in card.primary:
                state = _resolve(state, effect)
            return state
        case UseAlly(card):
            state = state.replace(allies_used=(*state.allies_used, card))
            for effect in card.ally:
                state = _resolve(state, effect)
            return state
        case UseBase(card):
            state = state.replace(bases_used=(*state.bases_used, card))
            for effect in card.primary:
                state = _resolve(state, effect)
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
                card,
            )
            for effect in card.scrap:
                state = _resolve(state, effect)
            return state
        case Acquire(card):
            player = player.replace(discard=(*player.discard, card))
            players = _players(state, me, player)
            trade = state.trade - card.cost
            if card == EXPLORER:
                return state.replace(
                    players=players, explorers=state.explorers - 1, trade=trade
                )
            i = state.trade_row.index(card)
            refill = state.trade_deck[:1]
            return state.replace(
                players=players,
                trade_row=state.trade_row[:i] + refill + state.trade_row[i + 1 :],
                trade_deck=state.trade_deck[1:],
                trade=trade,
            )
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
                    discard_scraps=(*state.discard_scraps, card),
                ),
                card,
            )
        case EndTurn():
            player = player.replace(
                hand=(),
                in_play=(),
                discard=player.discard + player.in_play + player.hand,
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
                discard_scraps=(),
                rng_seed=seed,
            )
        case _:
            assert_never(action)
