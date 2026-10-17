"""The game log's text: one line per applied action, plus turn headers.

The match runner calls these only when a log sink is given, so a match
without a log formats nothing. Players are numbered from 1 in the text.
"""

from typing import assert_never

from minimax_games.games.star_realms.action import (
    Action,
    AttackPlayer,
    EndTurn,
    PlayCard,
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


def describe(action: Action, state: GameState) -> str:
    """Describe `action` as the current player takes it in `state` (before it)."""
    player = _player(state.current)
    match action:
        case PlayCard(card):
            return f"{player} plays {card.name}"
        case AttackPlayer():
            opponent = 1 - state.current
            before = state.players[opponent].authority
            after = before - state.combat
            return (
                f"{player} attacks {_player(opponent)} for {state.combat}"
                f" (Authority {before} -> {after})"
            )
        case EndTurn():
            return f"{player} ends the turn"
        case _:
            assert_never(action)
