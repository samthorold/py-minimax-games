"""The match runner: deal, ask each bot in turn, apply, until someone wins."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from minimax_games.games.star_realms import rules, transcript
from minimax_games.games.star_realms.action import Action
from minimax_games.games.star_realms.bots import Bot
from minimax_games.games.star_realms.state import GameState
from minimax_games.games.star_realms.view import observe

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

    Each bot is given only its own view of the game (`view.observe`), never
    the state.

    Automatic play (`rules.auto_play`) runs at the start and after each
    action, so bots are only ever offered decisions.

    With `log`, each line of the game's transcript is passed to it as the game
    goes, automatic actions marked "(auto)". Without it (the default),
    nothing is formatted.
    """
    turn = 0

    def record(state: GameState, action: Action, *, auto: bool = False) -> None:
        nonlocal turn
        assert log is not None
        if state.turn != turn:
            turn = state.turn
            log(transcript.turn_header(state))
        log(transcript.describe(action, state) + (" (auto)" if auto else ""))

    on_auto = None if log is None else partial(record, auto=True)
    state = rules.auto_play(rules.new_game(seed), on_auto)
    while (winner := rules.winner(state)) is None:
        if state.turn > max_turns:
            msg = f"no winner after {max_turns} turns (seed {seed})"
            raise MatchTooLong(msg)
        actions = rules.legal_actions(state)
        action = bots[state.current].choose(observe(state, state.current), actions)
        if log is not None:
            record(state, action)
        state = rules.advance(state, action, on_auto)
    if log is not None:
        log(transcript.result(state, winner))
    return MatchResult(winner=winner, turns=state.turn)
