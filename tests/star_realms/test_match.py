import pytest

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.action import Action, EndTurn
from minimax_games.games.star_realms.bots import Bot, RandomBot
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
