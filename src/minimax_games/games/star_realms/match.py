"""The match runner: deal, ask each bot in turn, apply, until someone wins."""

from collections.abc import Callable
from dataclasses import dataclass

from minimax_games.games.star_realms import rules, transcript
from minimax_games.games.star_realms.bots import Bot

MAX_TURNS = 1000

type Log = Callable[[str], None]
"""A game log sink: called with each line of the transcript, e.g. `print`."""


class MatchTooLong(RuntimeError):
    """A match passed `max_turns`: a sign of a rules or bot bug, never a draw."""


@dataclass(frozen=True, slots=True)
class MatchResult:
    winner: int
    turns: int


def play_match(
    bots: tuple[Bot, Bot],
    seed: int,
    max_turns: int = MAX_TURNS,
    log: Log | None = None,
) -> MatchResult:
    """Play one game from `seed`, `bots[0]` moving first.

    With `log`, each line of the game's transcript is passed to it as the game
    goes. Without it (the default), nothing is formatted.
    """
    state = rules.new_game(seed)
    turn = 0
    while (winner := rules.winner(state)) is None:
        if state.turn > max_turns:
            msg = f"no winner after {max_turns} turns (seed {seed})"
            raise MatchTooLong(msg)
        if log is not None and state.turn != turn:
            turn = state.turn
            log(transcript.turn_header(state))
        actions = rules.legal_actions(state)
        action = bots[state.current].choose(state, actions)
        applied = rules.apply(state, action)
        if log is not None:
            log(transcript.describe(action, state))
        state = applied
    if log is not None:
        log(transcript.result(state, winner))
    return MatchResult(winner=winner, turns=state.turn)
