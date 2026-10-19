"""The game log's text: one line per applied action, plus turn headers.

The match runner calls these only when a log sink is given, so a match
without a log formats nothing. Players are numbered from 1 in the text.
"""

from typing import assert_never

from minimax_games.games.star_realms.action import (
    Acquire,
    AcquireFreeShip,
    AcquireOnTop,
    Action,
    AttackBase,
    AttackPlayer,
    Choose,
    DestroyTarget,
    Discard,
    EndTurn,
    PlayCard,
    Recycle,
    Scrap,
    ScrapDiscardCard,
    ScrapHandCard,
    ScrapTradeRowCard,
    UseAlly,
    UseBase,
)
from minimax_games.games.star_realms.card import Card
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
from minimax_games.games.star_realms.state import GameState


def _player(index: int) -> str:
    return f"Player {index + 1}"


def _authorities(state: GameState) -> str:
    first, second = state.players
    return f"Authority: Player 1 {first.authority}, Player 2 {second.authority}"


def turn_header(state: GameState) -> str:
    """The line that opens a turn: whose it is and both players' Authority."""
    return f"Turn {state.turn}: {_player(state.current)} to play. {_authorities(state)}"


def result(state: GameState, winner: int) -> str:
    """The closing line: who won, on which turn, and the final Authority."""
    return f"{_player(winner)} wins on turn {state.turn}. {_authorities(state)}"


def _gain(effect: Effect, state: GameState, card: Card) -> str:
    """What `effect`, one of `card`'s, gives the current player of `state`,
    e.g. "3 trade", "1 card" or "1 discard by Player 2"."""
    match effect:
        case Trade(amount):
            return f"{amount} trade"
        case Combat(amount):
            return f"{amount} combat"
        case Authority(amount):
            return f"{amount} Authority"
        case Draw(cards):
            return f"{cards} card{'s' if cards != 1 else ''}"
        case OpponentDiscards(cards):
            plural = "s" if cards != 1 else ""
            return f"{cards} discard{plural} by {_player(1 - state.current)}"
        case ScrapHandOrDiscard(cards):
            return f"{cards} scrap{'s' if cards != 1 else ''} from hand or discard pile"
        case ScrapFromHand():
            return "1 scrap from hand"
        case DestroyBase():
            return f"1 base of {_player(1 - state.current)} to destroy"
        case ScrapTradeRow(cards):
            return f"{cards} scrap{'s' if cards != 1 else ''} from the Trade Row"
        case DrawPerFactionCard():
            faction = card.faction.value if card.faction is not None else "no"
            return f"1 card per {faction} card played"
        case DiscardThenDraw(cards):
            return f"up to {cards} cards discarded then drawn"
        case NextShipOnTop():
            return "1 ship acquired on top of the deck"
        case AcquireShipFreeOnTop():
            return "1 free ship on top of the deck"
        case Choice(options):
            return " or ".join(_gains(option, state, card) for option in options)
        case (
            AllyForAllFactions()
            | CopyShip()
            | DrawIfBases()
            | ScrapThenDraw()
            | ShipsGainCombat()
        ):
            # Not playable yet; later slices give these their own wording.
            return str(effect)
        case _:
            assert_never(effect)


def _gains(effects: tuple[Effect, ...], state: GameState, card: Card) -> str:
    """What `effects`, `card`'s, give, e.g. "3 trade, 1 card"."""
    return ", ".join(_gain(effect, state, card) for effect in effects)


def describe(action: Action, state: GameState) -> str:
    """Describe `action` as the current player takes it in `state` (before it)."""
    player = _player(state.current)
    match action:
        case PlayCard(card):
            return f"{player} plays {card.name}"
        case UseAlly(card):
            gains = _gains(card.ally, state, card)
            return f"{player} uses {card.name}'s ally ability for {gains}"
        case UseBase(card):
            return f"{player} uses {card.name} for {_gains(card.primary, state, card)}"
        case Scrap(card):
            return f"{player} scraps {card.name} for {_gains(card.scrap, state, card)}"
        case Acquire(card):
            return f"{player} acquires {card.name} (cost {card.cost})"
        case AcquireOnTop(card):
            return (
                f"{player} acquires {card.name} (cost {card.cost}) on top of the deck"
            )
        case AcquireFreeShip(card):
            return f"{player} acquires {card.name} for free on top of the deck"
        case AttackPlayer():
            opponent = 1 - state.current
            before = state.players[opponent].authority
            after = before - state.combat
            return (
                f"{player} attacks {_player(opponent)} for {state.combat}"
                f" (Authority {before} -> {after})"
            )
        case AttackBase(card):
            opponent = _player(1 - state.current)
            return f"{player} destroys {opponent}'s {card.name} ({card.defense} combat)"
        case DestroyTarget(card):
            opponent = _player(1 - state.current)
            return f"{player} destroys {opponent}'s {card.name} (no combat)"
        case Discard(card):
            return f"{player} discards {card.name}"
        case ScrapHandCard(card):
            return f"{player} scraps {card.name} from hand"
        case ScrapDiscardCard(card):
            return f"{player} scraps {card.name} from the discard pile"
        case ScrapTradeRowCard(card):
            scrapped = f"{player} scraps {card.name} from the Trade Row"
            if not state.trade_deck:
                return scrapped
            return f"{scrapped}; {state.trade_deck[0].name} replaces it"
        case Choose(card, option):
            return f"{player} chooses {_gains(option, state, card)} from {card.name}"
        case Recycle(cards):
            discarded = ", ".join(card.name for card in cards)
            drawn = f"{len(cards)} card{'s' if len(cards) != 1 else ''}"
            return f"{player} discards {discarded} and draws {drawn}"
        case EndTurn():
            return f"{player} ends the turn"
        case _:
            assert_never(action)
