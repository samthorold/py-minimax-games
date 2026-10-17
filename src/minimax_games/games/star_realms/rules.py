"""The rules: dealing, legal actions and applying them."""

import dataclasses
import random
from typing import assert_never

from minimax_games.games.star_realms.action import (
    Acquire,
    Action,
    AttackPlayer,
    EndTurn,
    PlayCard,
    Scrap,
)
from minimax_games.games.star_realms.card import Card
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


def winner(state: GameState) -> int | None:
    """The index of the player who has won, or `None` while the game goes on."""
    for i, player in enumerate(state.players):
        if player.authority <= 0:
            return 1 - i
    return None


def legal_actions(state: GameState) -> tuple[Action, ...]:
    """The actions open to the current player.

    One `PlayCard` per distinct card in hand, one `Scrap` per distinct card in
    play with a scrap ability the engine implements, one `Acquire` per distinct
    affordable card in the Trade Row or on the Explorer pile, `AttackPlayer`
    while there is combat to spend, and `EndTurn`. Empty once the game is over.
    """
    if winner(state) is not None:
        return ()
    player = state.players[state.current]
    actions: list[Action] = [PlayCard(card) for card in dict.fromkeys(player.hand)]
    actions.extend(
        Scrap(card) for card in dict.fromkeys(player.in_play) if _scrappable(card)
    )
    market = state.trade_row + ((EXPLORER,) if state.explorers else ())
    actions.extend(
        Acquire(card) for card in dict.fromkeys(market) if card.cost <= state.trade
    )
    if state.combat > 0:
        actions.append(AttackPlayer())
    actions.append(EndTurn())
    return tuple(actions)


def apply(state: GameState, action: Action) -> GameState:
    """Return the state after the current player takes `action`.

    `state` itself is never changed. Raises `ValueError` if `action` isn't legal.
    """
    if action not in legal_actions(state):
        msg = f"{action} is not legal now"
        raise ValueError(msg)
    player = state.players[state.current]
    match action:
        case PlayCard(card):
            i = player.hand.index(card)
            hand = player.hand[:i] + player.hand[i + 1 :]
            player = dataclasses.replace(
                player, hand=hand, in_play=(*player.in_play, card)
            )
            state = _with_player(state, state.current, player)
            for effect in card.primary:
                state = _resolve(state, effect)
            return state
        case Scrap(card):
            i = player.in_play.index(card)
            in_play = player.in_play[:i] + player.in_play[i + 1 :]
            player = dataclasses.replace(player, in_play=in_play)
            state = _with_player(state, state.current, player)
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
                rng_seed=seed,
            )
        case _:
            assert_never(action)
