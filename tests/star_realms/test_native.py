"""Differential tests: the Rust engine (`minimax_games._native.star_realms`)
against the Python engine, the reference, on full-deck games.

Seeds deal the same game in both (the Rust engine reproduces CPython's
`random`), so every state, every list of legal actions and both players'
views must be equal after every action, in greedy-vs-greedy and seeded
random games, and so must the worlds sampled from a view with the same
seed (docs/star-realms.md, "Compiled engine"). Each test plays its seeds in
order and stops at the first divergence, naming the seed and the action.
"""

import random
from collections import Counter
from collections.abc import Callable
from typing import Any, get_args

import pytest

from minimax_games._native import star_realms as native
from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.action import Action, Choose
from minimax_games.games.star_realms.bots import greedy_move
from minimax_games.games.star_realms.card import Card
from minimax_games.games.star_realms.catalogue import CATALOGUE
from minimax_games.games.star_realms.effect import Choice
from minimax_games.games.star_realms.state import (
    NO_FACTIONS_PLAYED,
    GameState,
    PlayerState,
)
from minimax_games.games.star_realms.view import (
    PlayerView,
    View,
    observe,
    sample_world,
)

GREEDY_GAMES = 100
RANDOM_GAMES = 60
RANDOM_MAX_TURNS = 200


def _names(cards: tuple[Card, ...]) -> list[str]:
    return [card.name for card in cards]


def _player(player: PlayerState) -> dict[str, Any]:
    return {
        "authority": player.authority,
        "deck": _names(player.deck),
        "hand": _names(player.hand),
        "in_play": _names(player.in_play),
        "discard": _names(player.discard),
        "bases": _names(player.bases),
        "to_discard": player.to_discard,
        "known_top": player.known_top,
    }


# The turn state, the fields `GameState` and `View` share, piles first.
_TURN_PILES = ("trade_row", "scrap_heap", "allies_used", "bases_used", "choosing")
_TURN_COUNTS = (
    "explorers",
    "current",
    "turn",
    "trade",
    "combat",
    "may_scrap",
    "must_scrap",
    "may_destroy",
    "may_scrap_row",
    "recycling",
    "may_on_top",
    "free_ships",
    "scrap_to_draw",
    "copying",
)


def _turn(state: GameState | View) -> dict[str, Any]:
    turn: dict[str, Any] = {f: _names(getattr(state, f)) for f in _TURN_PILES}
    turn |= {f: getattr(state, f) for f in _TURN_COUNTS}
    turn["factions_played"] = list(state.factions_played)
    turn["trade_deck"] = _names(state.trade_deck)
    return turn


def snapshot(state: GameState) -> dict[str, Any]:
    """`state` as the Rust engine's `Game.state()` gives it."""
    return {
        "players": [_player(p) for p in state.players],
        "rng_seed": state.rng_seed,
        **_turn(state),
    }


def _player_view(player: PlayerView) -> dict[str, Any]:
    return {
        "authority": player.authority,
        "hand_size": player.hand_size,
        "deck_size": player.deck_size,
        "hand": _names(player.hand),
        "unseen": _names(player.unseen),
        "in_play": _names(player.in_play),
        "discard": _names(player.discard),
        "bases": _names(player.bases),
        "to_discard": player.to_discard,
        "deck_top": _names(player.deck_top),
    }


def view_snapshot(view: View) -> dict[str, Any]:
    """`view` as the Rust engine's `Game.observe()` gives it."""
    return {
        "observer": view.observer,
        "players": [_player_view(p) for p in view.players],
        **_turn(view),
    }


def _option(action: Choose) -> int:
    """`action`'s option's index among its card's options."""
    effects = action.card.primary + action.card.ally + action.card.scrap
    choice = next(e for e in effects if isinstance(e, Choice))
    return choice.options.index(action.option)


def native_action(action: Action) -> native.NativeAction:
    """`action` as the Rust engine names it: its kind, then its fields, a
    card as its name, cards as a tuple of names, an option as its index."""
    if isinstance(action, Choose):
        return "Choose", action.card.name, _option(action)
    fields = []
    for name in type(action).__match_args__:
        value = getattr(action, name)
        if isinstance(value, Card):
            fields.append(value.name)
        else:
            fields.append(tuple(card.name for card in value))
    return type(action).__name__, *fields


def _effect(effect: Any) -> tuple[str, list[Any]]:
    """`effect` as the Rust engine's `cards()` gives it."""
    if isinstance(effect, Choice):
        options = [[_effect(e) for e in option] for option in effect.options]
        return "Choice", [options]
    return type(effect).__name__, [getattr(effect, f) for f in effect.__match_args__]


def test_the_rust_cards_are_the_catalogue_s():
    assert [
        {
            "name": card.name,
            "cost": card.cost,
            "faction": None if card.faction is None else card.faction.value,
            "type": card.type.value,
            "defense": card.defense,
            "primary": [_effect(e) for e in card.primary],
            "ally": [_effect(e) for e in card.ally],
            "scrap": [_effect(e) for e in card.scrap],
            "copies": card.copies,
        }
        for card in CATALOGUE.values()
    ] == native.cards()


type Chooser = Callable[[GameState, tuple[Action, ...], native.Game], Action]


def _play(seed: int, choose: Chooser, max_turns: int) -> tuple[GameState, int]:
    """Play a game from `seed` in both engines, comparing state, legal
    actions, winner, both views, and a world sampled from the current
    player's view, after every action; `choose(state, actions)` picks each
    decision from the Python engine's legal actions (and may check the Rust
    `game`'s choice). Returns the final
    state and the number of decisions; fails at the first divergence,
    naming the seed and the action."""
    state = rules.auto_play(rules.new_game(seed))
    game = native.Game(seed)
    taken: list[Action] = []
    while True:
        where = f"seed {seed}, after action {len(taken)}"
        if taken:
            where += f" {native_action(taken[-1])}, turn {state.turn}"
        assert game.state() == snapshot(state), f"state differs: {where}"
        actions = rules.legal_actions(state)
        assert game.legal_actions() == [native_action(a) for a in actions], (
            f"legal actions differ: {where}"
        )
        assert game.winner() == rules.winner(state), f"winner differs: {where}"
        for observer in (0, 1):
            view = observe(state, observer)
            assert game.observe(observer) == view_snapshot(view), (
                f"view {observer} differs: {where}"
            )
        sample_seed = seed * 100_000 + len(taken)
        view = observe(state, state.current)
        world = sample_world(view, random.Random(sample_seed))
        sampled = game.sample_world(state.current, sample_seed)
        assert sampled.state() == snapshot(world), f"sampled world differs: {where}"
        if not actions or state.turn > max_turns:
            return state, len(taken)
        action = choose(state, actions, game)
        state = rules.advance(state, action)
        game.advance(native_action(action))
        taken.append(action)
        _TAKEN[type(action).__name__] += 1


_TAKEN: Counter[str] = Counter()
"""How often each kind of action was taken across the games played."""


def _greedy(state: GameState, actions: tuple[Action, ...], game: native.Game) -> Action:
    action = greedy_move(state.must_scrap, actions)
    assert game.greedy_move() == native_action(action), f"greedy differs: {action}"
    return action


def test_greedy_games_match_the_python_engine():
    for seed in range(GREEDY_GAMES):
        state, decisions = _play(seed, _greedy, max_turns=1000)
        result = (rules.winner(state), state.turn, decisions)
        assert native.play_greedy(seed) == result, f"seed {seed}"


def _random(seed: int) -> Chooser:
    """A seeded random bot."""
    rng = random.Random(seed)
    return lambda _, actions, __: rng.choice(actions)


def test_random_games_match_the_python_engine_and_take_every_kind_of_action():
    _TAKEN.clear()
    for seed in range(RANDOM_GAMES):
        _play(seed, _random(seed), RANDOM_MAX_TURNS)
    kinds = {variant.__name__ for variant in get_args(Action.__value__)}
    assert kinds - _TAKEN.keys() == set()


def test_a_sampled_world_has_the_view():
    game = native.Game(3)
    for _ in range(40):
        for observer in (0, 1):
            world = game.sample_world(observer, 11)
            assert world.observe(observer) == game.observe(observer)
        game.advance(game.greedy_move())


def test_an_illegal_action_is_refused():
    game = native.Game(0)
    with pytest.raises(ValueError, match="not legal"):
        game.advance(("Acquire", "Fleet HQ"))
    with pytest.raises(ValueError, match="unknown card"):
        game.advance(("Acquire", "Nebula"))
    with pytest.raises(ValueError, match="unknown action"):
        game.advance(("EndTurn", "Scout"))
    with pytest.raises(ValueError, match="observer"):
        game.observe(2)


def test_the_initial_factions_played_are_empty():
    assert native.Game(0).state()["factions_played"] == list(NO_FACTIONS_PLAYED)
