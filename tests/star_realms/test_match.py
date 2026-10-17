import dataclasses
import re
from collections import Counter

import pytest

from minimax_games.games.star_realms import rules, transcript
from minimax_games.games.star_realms.action import (
    Acquire,
    Action,
    AttackBase,
    AttackPlayer,
    EndTurn,
    PlayCard,
    Scrap,
    UseAlly,
    UseBase,
)
from minimax_games.games.star_realms.bots import Bot, RandomBot
from minimax_games.games.star_realms.catalogue import CATALOGUE, EXPLORER, VIPER
from minimax_games.games.star_realms.effect import Authority, Draw
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


class Conserving:
    """Wraps a bot: checks no card is created or lost by any action, and counts
    the kinds of action it takes."""

    def __init__(self, bot: Bot, cards: Counter[str]) -> None:
        self.bot = bot
        self.cards = cards
        self.taken: Counter[str] = Counter()

    def choose(self, state: GameState, actions: tuple[Action, ...]) -> Action:
        assert all_cards(state) == self.cards
        action = self.bot.choose(state, actions)
        assert all_cards(rules.apply(state, action)) == self.cards
        self.taken[type(action).__name__] += 1
        return action


def test_random_games_neither_create_nor_lose_cards_and_use_every_action():
    taken: Counter[str] = Counter()
    for seed in SEEDS:
        cards = all_cards(rules.new_game(seed))
        bots = (
            Conserving(RandomBot(seed), cards),
            Conserving(RandomBot(seed + 1000), cards),
        )
        play_match((bots[0], bots[1]), seed)
        taken += bots[0].taken + bots[1].taken
    kinds = (Acquire, Scrap, UseAlly, UseBase, AttackBase, AttackPlayer, PlayCard)
    for kind in kinds:
        assert taken[kind.__name__] > 0, kind


class Buyer:
    """Plays its whole hand, acquires whatever it can, attacks bases then the
    player, ends the turn."""

    def choose(self, state: GameState, actions: tuple[Action, ...]) -> Action:
        for kind in (PlayCard, Acquire, AttackBase, AttackPlayer):
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
