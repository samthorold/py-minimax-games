"""The rules: dealing, legal actions and applying them."""

import dataclasses
import random
from collections.abc import Callable
from typing import assert_never

from minimax_games.games.star_realms.action import (
    Acquire,
    Action,
    AttackBase,
    AttackPlayer,
    EndTurn,
    PlayCard,
    Scrap,
    UseAlly,
    UseBase,
)
from minimax_games.games.star_realms.card import Card, CardType
from minimax_games.games.star_realms.catalogue import (
    EXPLORER,
    EXPLORER_PILE,
    SLICE_1_TRADE_DECK,
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
    return dataclasses.replace(player, deck=deck, hand=hand, discard=discard), seed


def new_game(seed: int, trade_deck: tuple[Card, ...] = SLICE_1_TRADE_DECK) -> GameState:
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


def _with_player(state: GameState, index: int, player: PlayerState) -> GameState:
    players = list(state.players)
    players[index] = player
    first, second = players
    return dataclasses.replace(state, players=(first, second))


def _resolve(state: GameState, effect: Effect) -> GameState:
    match effect:
        case Trade(amount):
            return dataclasses.replace(state, trade=state.trade + amount)
        case Combat(amount):
            return dataclasses.replace(state, combat=state.combat + amount)
        case Authority(amount):
            player = state.players[state.current]
            player = dataclasses.replace(player, authority=player.authority + amount)
            return _with_player(state, state.current, player)
        case Draw(cards):
            player, seed = _draw(state.players[state.current], cards, state.rng_seed)
            state = _with_player(state, state.current, player)
            return dataclasses.replace(state, rng_seed=seed)
        case (
            AcquireShipFreeOnTop()
            | AllyForAllFactions()
            | Choice()
            | CopyShip()
            | DestroyBase()
            | DiscardThenDraw()
            | DrawIfBases()
            | DrawPerFactionCard()
            | NextShipOnTop()
            | OpponentDiscards()
            | ScrapFromHand()
            | ScrapHandOrDiscard()
            | ScrapThenDraw()
            | ScrapTradeRow()
            | ShipsGainCombat()
        ):
            # Only first-slice ships are dealt so far; later slices
            # implement the rest (docs/star-realms.md).
            msg = f"{effect} is not implemented yet"
            raise NotImplementedError(msg)
        case _:
            assert_never(effect)


def _implemented(effect: Effect) -> bool:
    return isinstance(effect, Trade | Combat | Authority | Draw)


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
    ability the engine implements; one `Acquire` per distinct affordable card
    in the Trade Row or on the Explorer pile; one `AttackBase` per distinct
    opponent's base whose defense the combat covers, only Outposts while the
    opponent has one; `AttackPlayer` while there is combat to spend and the
    opponent has no Outpost; and `EndTurn`. Empty once the game is over.
    """
    if winner(state) is not None:
        return ()
    player = state.players[state.current]
    actions: list[Action] = [PlayCard(card) for card in dict.fromkeys(player.hand)]
    in_play = player.in_play + player.bases
    actions.extend(
        UseAlly(card)
        for card in dict.fromkeys(in_play)
        if _ally_usable(card)
        and _allied(card, in_play)
        and in_play.count(card) > state.allies_used.count(card)
    )
    actions.extend(
        UseBase(card)
        for card in dict.fromkeys(player.bases)
        if _primary_usable(card)
        and player.bases.count(card) > state.bases_used.count(card)
    )
    actions.extend(Scrap(card) for card in dict.fromkeys(in_play) if _scrappable(card))
    market = state.trade_row + ((EXPLORER,) if state.explorers else ())
    actions.extend(
        Acquire(card) for card in dict.fromkeys(market) if card.cost <= state.trade
    )
    targets = state.players[1 - state.current].bases
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


def _pure_gain(effects: tuple[Effect, ...]) -> bool:
    """Whether `effects` only add trade, combat or Authority."""
    return all(isinstance(e, Trade | Combat | Authority) for e in effects)


def automatic(action: Action) -> bool:
    """Whether `action` is applied automatically, never offered as a decision.

    The rule for conservative automatic play (docs/star-realms.md): an action
    is automatic when taking it now can't make any later option worse.

    - `PlayCard` of a **ship** whose primary effects are pure gain (trade,
      combat, Authority). A ship in hand and a ship in play both go to the
      discard pile at the end of the turn; in play it only adds options
      (ally abilities, scrapping).
    - `UseAlly` whose ally effects are pure gain, and `UseBase` whose primary
      effects are pure gain: the card stays where it is, and the gain can't
      be lost by taking it early.

    Everything else is a decision. `Draw` never is automatic: drawing can
    reshuffle the discard pile, so drawing before or after an `Acquire` (the
    acquired card goes to the discard pile) changes the deck. Playing a base
    is a decision too: a base left in hand goes to the discard pile, so it
    changes what is reshuffled. `Scrap` loses the card, and `Acquire`,
    attacks and `EndTurn` are the turn's real choices.

    This is sound only for the effects the engine implements so far: every
    new effect or action (opponent discards, scrapping from hand, ...) must
    revisit it.
    """
    match action:
        case PlayCard(card):
            return card.type is CardType.SHIP and _pure_gain(card.primary)
        case UseAlly(card):
            return _pure_gain(card.ally)
        case UseBase(card):
            return _pure_gain(card.primary)
        case Scrap() | Acquire() | AttackPlayer() | AttackBase() | EndTurn():
            return False
        case _:
            assert_never(action)


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
    return next((a for a in legal_actions(state) if automatic(a)), None)


def advance(
    state: GameState, action: Action, on_action: OnAction | None = None
) -> GameState:
    """Apply `action`, a decision, then any automatic play it opens up.

    This is the move generator bots and search share: from a state `advance`
    returns, `legal_actions` holds decisions only. `apply` is the bare rules,
    without automatic play.
    """
    return auto_play(apply(state, action), on_action)


def apply(state: GameState, action: Action) -> GameState:
    """Return the state after the current player takes `action`.

    `state` itself is never changed. Raises `ValueError` if `action` isn't legal.
    """
    if action not in legal_actions(state):
        msg = f"{action} is not legal now"
        raise ValueError(msg)
    return _apply(state, action)


def _apply(state: GameState, action: Action) -> GameState:
    """`apply` for an action already known to be legal."""
    player = state.players[state.current]
    match action:
        case PlayCard(card):
            player = dataclasses.replace(player, hand=_without(player.hand, card))
            if card.type is CardType.SHIP:
                player = dataclasses.replace(player, in_play=(*player.in_play, card))
                state = _with_player(state, state.current, player)
            else:
                # Playing a base uses its primary ability for this turn.
                player = dataclasses.replace(player, bases=(*player.bases, card))
                state = _with_player(state, state.current, player)
                state = dataclasses.replace(state, bases_used=(*state.bases_used, card))
            for effect in card.primary:
                state = _resolve(state, effect)
            return state
        case UseAlly(card):
            state = dataclasses.replace(state, allies_used=(*state.allies_used, card))
            for effect in card.ally:
                state = _resolve(state, effect)
            return state
        case UseBase(card):
            state = dataclasses.replace(state, bases_used=(*state.bases_used, card))
            for effect in card.primary:
                state = _resolve(state, effect)
            return state
        case Scrap(card):
            if card.type is CardType.SHIP:
                player = dataclasses.replace(
                    player, in_play=_without(player.in_play, card)
                )
            else:
                player = dataclasses.replace(player, bases=_without(player.bases, card))
            state = _with_player(state, state.current, player)
            # Of identical copies, scrap one whose primary (for a base) and
            # ally abilities were used, so an unused one keeps its abilities.
            if card in state.bases_used:
                used = _without(state.bases_used, card)
                state = dataclasses.replace(state, bases_used=used)
            if card in state.allies_used:
                used = _without(state.allies_used, card)
                state = dataclasses.replace(state, allies_used=used)
            if card == EXPLORER:
                state = dataclasses.replace(state, explorers=state.explorers + 1)
            else:
                state = dataclasses.replace(state, scrap_heap=(*state.scrap_heap, card))
            for effect in card.scrap:
                state = _resolve(state, effect)
            return state
        case Acquire(card):
            if card == EXPLORER:
                state = dataclasses.replace(state, explorers=state.explorers - 1)
            else:
                i = state.trade_row.index(card)
                refill = state.trade_deck[:1]
                state = dataclasses.replace(
                    state,
                    trade_row=state.trade_row[:i] + refill + state.trade_row[i + 1 :],
                    trade_deck=state.trade_deck[1:],
                )
            player = dataclasses.replace(player, discard=(*player.discard, card))
            state = _with_player(state, state.current, player)
            return dataclasses.replace(state, trade=state.trade - card.cost)
        case AttackPlayer():
            opponent = state.players[1 - state.current]
            opponent = dataclasses.replace(
                opponent, authority=opponent.authority - state.combat
            )
            return dataclasses.replace(
                _with_player(state, 1 - state.current, opponent), combat=0
            )
        case AttackBase(card):
            opponent = state.players[1 - state.current]
            opponent = dataclasses.replace(
                opponent,
                bases=_without(opponent.bases, card),
                discard=(*opponent.discard, card),
            )
            state = _with_player(state, 1 - state.current, opponent)
            return dataclasses.replace(state, combat=state.combat - (card.defense or 0))
        case EndTurn():
            player = dataclasses.replace(
                player,
                hand=(),
                in_play=(),
                discard=player.discard + player.in_play + player.hand,
            )
            player, seed = _draw(player, HAND_SIZE, state.rng_seed)
            state = _with_player(state, state.current, player)
            return dataclasses.replace(
                state,
                current=1 - state.current,
                turn=state.turn + 1,
                trade=0,
                combat=0,
                allies_used=(),
                bases_used=(),
                rng_seed=seed,
            )
        case _:
            assert_never(action)
