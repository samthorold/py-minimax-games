import dataclasses
import re
from collections import Counter
from collections.abc import Iterator

import pytest

from minimax_games.games.star_realms import rules, transcript
from minimax_games.games.star_realms.action import (
    Acquire,
    Action,
    AttackBase,
    AttackPlayer,
    Discard,
    EndTurn,
    PlayCard,
    Scrap,
    UseAlly,
    UseBase,
)
from minimax_games.games.star_realms.bots import (
    BenchmarkBot,
    Bot,
    GreedyBot,
    RandomBot,
)
from minimax_games.games.star_realms.catalogue import CATALOGUE, EXPLORER, VIPER
from minimax_games.games.star_realms.effect import Authority, Draw
from minimax_games.games.star_realms.match import MatchTooLong, play_match
from minimax_games.games.star_realms.sampler import SamplerBot
from minimax_games.games.star_realms.state import GameState
from minimax_games.games.star_realms.view import View, observe

SEEDS = range(50)


class Recorder:
    """Wraps a bot: records its actions."""

    def __init__(self, bot: Bot, log: list[Action]) -> None:
        self.bot = bot
        self.log = log

    def choose(self, view: View, actions: tuple[Action, ...]) -> Action:
        action = self.bot.choose(view, actions)
        self.log.append(action)
        return action


def moves(seed: int, bots: tuple[Bot, Bot]) -> Iterator[tuple[GameState, Action]]:
    """Each state of a game from `seed` with the action taken in it, the bot's
    or an automatic one, as `play_match` plays it: for checking properties of
    the rules."""
    automatic: list[tuple[GameState, Action]] = []

    def record(state: GameState, action: Action) -> None:
        automatic.append((state, action))

    state = rules.auto_play(rules.new_game(seed), record)
    while True:
        yield from automatic
        automatic.clear()
        if rules.winner(state) is not None:
            return
        view = observe(state, state.current)
        action = bots[state.current].choose(view, rules.legal_actions(state))
        yield state, action
        state = rules.advance(state, action, record)


def test_applying_an_action_leaves_the_state_alone():
    for state, action in moves(7, (RandomBot(7), RandomBot(8))):
        before = hash(state)
        snapshot = (state.players, state.rng_seed, state.trade, state.combat)
        rules.apply(state, action)
        assert hash(state) == before
        assert (state.players, state.rng_seed, state.trade, state.combat) == snapshot


class Passive:
    """Ends every turn, after a discard it must make."""

    def choose(self, view: View, actions: tuple[Action, ...]) -> Action:
        return EndTurn() if EndTurn() in actions else actions[0]


def test_games_between_random_bots_end_with_a_winner():
    # With the 30-card first slice, seed 13 stalled: once the market was
    # empty, the Authority automatic play gave outpaced random attacks. Since
    # opponent discards joined the deck, none of seeds 0-299 does.
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

    def choose(self, view: View, actions: tuple[Action, ...]) -> Action:
        for action in actions:
            if isinstance(action, PlayCard):
                return action
        if AttackPlayer() in actions:
            return AttackPlayer()
        return EndTurn() if EndTurn() in actions else actions[0]


def test_the_log_shows_each_turns_plays_attacks_and_authority():
    seed = 3
    lines: list[str] = []
    play_match((Aggressive(), Aggressive()), seed, log=lines.append)

    hand = rules.new_game(seed).players[0].hand
    vipers = hand.count(VIPER)
    expected = [
        "Turn 1: Player 1 to play. Authority: Player 1 50, Player 2 50",
        *(f"Player 1 plays {card.name} (auto)" for card in hand),
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


def all_cards(state: GameState) -> Counter[str]:
    """Every card in the game, by name, wherever it is."""
    piles = [
        state.trade_deck,
        state.trade_row,
        (EXPLORER,) * state.explorers,
        state.scrap_heap,
    ]
    for player in state.players:
        piles += [
            player.deck,
            player.hand,
            player.in_play,
            player.discard,
            player.bases,
        ]
    return Counter(card.name for pile in piles for card in pile)


def test_random_games_neither_create_nor_lose_cards_and_use_every_action():
    taken: Counter[str] = Counter()
    for seed in SEEDS:
        cards = all_cards(rules.new_game(seed))
        for state, action in moves(seed, (RandomBot(seed), RandomBot(seed + 1000))):
            assert all_cards(state) == cards
            assert all_cards(rules.apply(state, action)) == cards
            taken[type(action).__name__] += 1
    kinds = (
        Acquire,
        Scrap,
        UseAlly,
        UseBase,
        AttackBase,
        AttackPlayer,
        PlayCard,
        Discard,
    )
    for kind in kinds:
        assert taken[kind.__name__] > 0, kind


class Buyer:
    """Makes any discard it must, plays its whole hand, acquires whatever it
    can, attacks bases then the player, ends the turn."""

    def choose(self, view: View, actions: tuple[Action, ...]) -> Action:
        for kind in (Discard, PlayCard, Acquire, AttackBase, AttackPlayer):
            for action in actions:
                if isinstance(action, kind):
                    return action
        return EndTurn()


def test_the_log_shows_acquisitions_with_their_cost():
    lines: list[str] = []
    play_match((Buyer(), Buyer()), seed=1, log=lines.append)
    acquires = [line for line in lines if " acquires " in line]
    assert acquires
    for line in acquires:
        match = re.fullmatch(r"Player [12] acquires (.+) \(cost (\d+)\)", line)
        assert match is not None
        name, cost = match.groups()
        assert CATALOGUE[name].cost == int(cost)


def test_the_log_shows_a_scrap_and_what_it_gave():
    state = rules.new_game(seed=1)
    ram, frigate = CATALOGUE["Ram"], CATALOGUE["Imperial Frigate"]
    assert transcript.describe(Scrap(EXPLORER), state) == (
        "Player 1 scraps Explorer for 2 combat"
    )
    assert transcript.describe(Scrap(ram), state) == "Player 1 scraps Ram for 3 trade"
    assert transcript.describe(Scrap(frigate), state) == (
        "Player 1 scraps Imperial Frigate for 1 card"
    )
    blob = dataclasses.replace(ram, scrap=(Authority(4), Draw(2)))
    assert transcript.describe(Scrap(blob), dataclasses.replace(state, current=1)) == (
        "Player 2 scraps Ram for 4 Authority, 2 cards"
    )


def test_the_log_shows_an_ally_ability_and_what_it_gave():
    state = rules.new_game(seed=1)
    cutter, mothership = CATALOGUE["Cutter"], CATALOGUE["Mothership"]
    assert transcript.describe(UseAlly(cutter), state) == (
        "Player 1 uses Cutter's ally ability for 4 combat"
    )
    assert transcript.describe(UseAlly(mothership), state) == (
        "Player 1 uses Mothership's ally ability for 1 card"
    )


def test_the_log_shows_a_base_used_and_a_base_destroyed():
    state = rules.new_game(seed=1)
    wheel, station = CATALOGUE["Blob Wheel"], CATALOGUE["Space Station"]
    assert transcript.describe(UseBase(wheel), state) == (
        "Player 1 uses Blob Wheel for 1 combat"
    )
    assert transcript.describe(AttackBase(wheel), state) == (
        "Player 1 destroys Player 2's Blob Wheel (5 combat)"
    )
    assert transcript.describe(
        AttackBase(station), dataclasses.replace(state, current=1)
    ) == ("Player 2 destroys Player 1's Space Station (4 combat)")


def test_the_log_shows_opponent_discards_given_and_made():
    state = rules.new_game(seed=1)
    survey, redoubt = CATALOGUE["Survey Ship"], CATALOGUE["Royal Redoubt"]
    assert transcript.describe(Scrap(survey), state) == (
        "Player 1 scraps Survey Ship for 1 discard by Player 2"
    )
    assert transcript.describe(
        UseAlly(redoubt), dataclasses.replace(state, current=1)
    ) == ("Player 2 uses Royal Redoubt's ally ability for 1 discard by Player 1")
    assert transcript.describe(
        Discard(VIPER), dataclasses.replace(state, current=1)
    ) == ("Player 2 discards Viper")


class OnlyViews:
    """Wraps a bot: checks every observation it is given is its own view."""

    def __init__(self, bot: Bot, player: int) -> None:
        self.bot = bot
        self.player = player
        self.calls = 0

    def choose(self, view: View, actions: tuple[Action, ...]) -> Action:
        assert not isinstance(view, GameState)
        assert isinstance(view, View)
        assert view.observer == view.current == self.player
        self.calls += 1
        return self.bot.choose(view, actions)


class NoAutomatic:
    """Wraps a bot: checks it is never offered an automatic action."""

    def __init__(self, bot: Bot) -> None:
        self.bot = bot

    def choose(self, view: View, actions: tuple[Action, ...]) -> Action:
        assert not any(rules.automatic(a) for a in actions), actions
        return self.bot.choose(view, actions)


def test_bots_are_never_offered_an_automatic_action():
    for seed in range(10):
        bots = (NoAutomatic(RandomBot(seed)), NoAutomatic(RandomBot(seed + 1)))
        play_match(bots, seed)


def test_the_log_marks_automatic_actions():
    lines: list[str] = []
    play_match((RandomBot(1), RandomBot(2)), seed=5, log=lines.append)
    auto = [line for line in lines if line.endswith(" (auto)")]
    assert any(" plays Scout (auto)" in line for line in auto)
    assert not any(" acquires " in line or " attacks " in line for line in auto)


def test_bots_are_only_ever_given_their_own_view():
    for seed in range(5):
        first = GreedyBot() if seed % 2 else RandomBot(seed)
        bots = (OnlyViews(first, 0), OnlyViews(RandomBot(seed + 1), 1))
        play_match(bots, seed)
        assert bots[0].calls > 0
        assert bots[1].calls > 0


def test_the_sampler_is_only_ever_given_its_own_view():
    # It searches sampled worlds, never the state: it is no benchmark.
    assert not isinstance(SamplerBot(0), BenchmarkBot)
    for seed in range(2):
        sampler = SamplerBot(seed, worlds=2, turns=1, budget=60)
        bots = (OnlyViews(sampler, 0), OnlyViews(GreedyBot(), 1))
        with pytest.raises(MatchTooLong):
            play_match(bots, seed, max_turns=4)
        assert bots[0].calls > 0


class SeesTheState:
    """A benchmark-only bot: checks it is given the true state, then plays
    greedily."""

    def __init__(self) -> None:
        self.calls = 0

    def choose_from_state(
        self, state: GameState, actions: tuple[Action, ...]
    ) -> Action:
        assert isinstance(state, GameState)
        assert rules.legal_actions(state) == actions
        self.calls += 1
        return GreedyBot().choose(observe(state, state.current), actions)


def test_only_a_benchmark_bot_is_given_the_true_state():
    for seed in range(4):
        cheat, ordinary = SeesTheState(), OnlyViews(RandomBot(seed), seed % 2)
        bots = (ordinary, cheat) if seed % 2 == 0 else (cheat, ordinary)
        play_match(bots, seed)
        assert cheat.calls > 0
        assert ordinary.calls > 0
