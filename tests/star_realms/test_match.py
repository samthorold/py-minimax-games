import pytest

from minimax_games.games.star_realms import rules, transcript
from minimax_games.games.star_realms.action import (
    Action,
    AttackPlayer,
    EndTurn,
    PlayCard,
)
from minimax_games.games.star_realms.bots import Bot, RandomBot
from minimax_games.games.star_realms.card import VIPER
from minimax_games.games.star_realms.match import MatchTooLong, play_match
from minimax_games.games.star_realms.state import GameState

SEEDS = range(50)


class Recorder:
    """Wraps a bot: records its actions, and checks `apply` leaves the state alone."""

    def __init__(self, bot: Bot, log: list[Action]) -> None:
        self.bot = bot
        self.log = log

    def choose(self, state: GameState, actions: tuple[Action, ...]) -> Action:
        action = self.bot.choose(state, actions)
        before = hash(state)
        snapshot = (state.players, state.rng_seed, state.trade, state.combat)
        rules.apply(state, action)
        assert hash(state) == before
        assert (state.players, state.rng_seed, state.trade, state.combat) == snapshot
        self.log.append(action)
        return action


class Passive:
    def choose(self, state: GameState, actions: tuple[Action, ...]) -> Action:
        return EndTurn()


def test_games_between_random_bots_end_with_a_winner():
    for seed in SEEDS:
        result = play_match((RandomBot(seed), RandomBot(seed + 1000)), seed)
        assert result.winner in (0, 1)
        assert result.turns > 0


def test_the_same_seed_gives_the_same_game():
    def record(seed: int) -> tuple[object, list[Action]]:
        log: list[Action] = []
        bots = (Recorder(RandomBot(seed), log), Recorder(RandomBot(seed + 1), log))
        return play_match(bots, seed), log

    assert record(7) == record(7)
    assert record(7) != record(8)


def test_a_match_that_never_ends_fails_loudly():
    with pytest.raises(MatchTooLong):
        play_match((Passive(), Passive()), seed=0, max_turns=20)


class Aggressive:
    """Plays the first card in hand, attacks whenever it can, then ends the turn."""

    def choose(self, state: GameState, actions: tuple[Action, ...]) -> Action:
        for action in actions:
            if isinstance(action, PlayCard):
                return action
        if AttackPlayer() in actions:
            return AttackPlayer()
        return EndTurn()


def test_the_log_shows_each_turns_plays_attacks_and_authority():
    seed = 3
    lines: list[str] = []
    play_match((Aggressive(), Aggressive()), seed, log=lines.append)

    hand = rules.new_game(seed).players[0].hand
    vipers = hand.count(VIPER)
    expected = [
        "Turn 1: Player 1 to play. Authority: Player 1 50, Player 2 50",
        *(f"Player 1 plays {card.name}" for card in hand),
        *(
            [f"Player 1 attacks Player 2 for {vipers} (Authority 50 -> {50 - vipers})"]
            if vipers
            else []
        ),
        "Player 1 ends the turn",
        f"Turn 2: Player 2 to play. Authority: Player 1 50, Player 2 {50 - vipers}",
    ]
    assert lines[: len(expected)] == expected


def test_the_log_ends_with_the_winner_and_final_authority():
    lines: list[str] = []
    result = play_match((RandomBot(1), RandomBot(2)), seed=5, log=lines.append)

    winner, loser = f"Player {result.winner + 1}", f"Player {2 - result.winner}"
    assert lines[-1].startswith(f"{winner} wins on turn {result.turns}. Authority:")
    assert lines[-2].startswith(f"{winner} attacks {loser} for ")


def test_the_same_seed_gives_the_same_transcript():
    def transcript(seed: int) -> list[str]:
        lines: list[str] = []
        play_match((RandomBot(seed), RandomBot(seed + 1)), seed, log=lines.append)
        return lines

    assert transcript(7) == transcript(7)
    assert transcript(7) != transcript(8)


def test_without_a_log_nothing_is_formatted(monkeypatch: pytest.MonkeyPatch):
    def fail(*args: object) -> str:
        raise AssertionError("formatted a log line with the log off")

    for formatter in ("turn_header", "describe", "result"):
        monkeypatch.setattr(transcript, formatter, fail)

    result = play_match((RandomBot(1), RandomBot(2)), seed=5)
    assert result.winner in (0, 1)
