"""The match runner: deal, ask each bot in turn, apply, until someone wins."""

from dataclasses import dataclass

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.bots import Bot

MAX_TURNS = 1000


class MatchTooLong(RuntimeError):
    """A match passed `max_turns`: a sign of a rules or bot bug, never a draw."""


@dataclass(frozen=True, slots=True)
class MatchResult:
    winner: int
    turns: int


def play_match(
    bots: tuple[Bot, Bot], seed: int, max_turns: int = MAX_TURNS
) -> MatchResult:
    """Play one game from `seed`, `bots[0]` moving first."""
    state = rules.new_game(seed)
    while (winner := rules.winner(state)) is None:
        if state.turn > max_turns:
            msg = f"no winner after {max_turns} turns (seed {seed})"
            raise MatchTooLong(msg)
        actions = rules.legal_actions(state)
        state = rules.apply(state, bots[state.current].choose(state, actions))
    return MatchResult(winner=winner, turns=state.turn)
