"""The game log's text: one line per applied action, plus turn headers.

The match runner calls these only when a log sink is given, so a match
without a log formats nothing. Players are numbered from 1 in the text.
"""

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


def _gain(effect: Effect, state: GameState) -> str:
    """What `effect` gives the current player of `state`, e.g. "3 trade",
    "1 card" or "1 discard by Player 2"."""
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
            # Not playable yet; later slices give these their own wording.
            return str(effect)
        case _:
            assert_never(effect)


def describe(action: Action, state: GameState) -> str:
    """Describe `action` as the current player takes it in `state` (before it)."""
    player = _player(state.current)
    match action:
        case PlayCard(card):
            return f"{player} plays {card.name}"
        case UseAlly(card):
            gains = ", ".join(_gain(effect, state) for effect in card.ally)
            return f"{player} uses {card.name}'s ally ability for {gains}"
        case UseBase(card):
            gains = ", ".join(_gain(effect, state) for effect in card.primary)
            return f"{player} uses {card.name} for {gains}"
        case Scrap(card):
            gains = ", ".join(_gain(effect, state) for effect in card.scrap)
            return f"{player} scraps {card.name} for {gains}"
        case Acquire(card):
            return f"{player} acquires {card.name} (cost {card.cost})"
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
        case EndTurn():
            return f"{player} ends the turn"
        case _:
            assert_never(action)
