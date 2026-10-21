"""Differential tests: the Rust engine (`minimax_games._native.star_realms`)
against the Python engine, the reference, on first-slice games.

Seeds deal the same game in both (the Rust engine reproduces CPython's
`random`), so every state and every list of legal actions must be equal,
after every action, in greedy-vs-greedy and seeded random games
(docs/star-realms.md, "Compiled engine").
"""

import random
from typing import Any

import pytest

from minimax_games._native import star_realms as native
from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.action import Action
from minimax_games.games.star_realms.bots import greedy_move
from minimax_games.games.star_realms.card import Card
from minimax_games.games.star_realms.catalogue import CATALOGUE, playable_deck
from minimax_games.games.star_realms.effect import Authority, Combat, Draw, Trade
from minimax_games.games.star_realms.state import (
    NO_FACTIONS_PLAYED,
    GameState,
    PlayerState,
)

FIRST_SLICE = playable_deck(frozenset({Trade, Combat, Authority, Draw}))

GREEDY_GAMES = 200
RANDOM_GAMES = 100
RANDOM_MAX_TURNS = 300


def _names(cards: tuple[Card, ...]) -> list[str]:
    return [card.name for card in cards]


def _player(player: PlayerState) -> dict[str, Any]:
    # Fields the first slice never sets: the Rust engine leaves them out.
    assert (player.to_discard, player.known_top) == (0, 0)
    return {
        "authority": player.authority,
        "deck": _names(player.deck),
        "hand": _names(player.hand),
        "in_play": _names(player.in_play),
        "discard": _names(player.discard),
        "bases": _names(player.bases),
    }


def snapshot(state: GameState) -> dict[str, Any]:
    """`state` as the Rust engine's `Game.state()` gives it."""
    unmodelled = (
        state.may_scrap,
        state.must_scrap,
        state.may_destroy,
        state.may_scrap_row,
        state.choosing,
        state.recycling,
        state.may_on_top,
        state.free_ships,
        state.scrap_to_draw,
        state.copying,
    )
    assert unmodelled == (0, 0, 0, 0, (), 0, 0, 0, 0, 0)
    return {
        "players": [_player(p) for p in state.players],
        "rng_seed": state.rng_seed,
        "trade_deck": _names(state.trade_deck),
        "trade_row": _names(state.trade_row),
        "explorers": state.explorers,
        "scrap_heap": _names(state.scrap_heap),
        "current": state.current,
        "turn": state.turn,
        "trade": state.trade,
        "combat": state.combat,
        "allies_used": _names(state.allies_used),
        "bases_used": _names(state.bases_used),
        "factions_played": list(state.factions_played),
    }


def native_action(action: Action) -> native.NativeAction:
    """`action` as the Rust engine names it: its kind and its card's name."""
    card = getattr(action, "card", None)
    return type(action).__name__, None if card is None else card.name


def _effects(effects: tuple[Any, ...]) -> list[tuple[str, int]]:
    return [
        (type(e).__name__, e.cards if isinstance(e, Draw) else e.amount)
        for e in effects
    ]


def test_the_rust_cards_are_the_catalogue_s_first_slice_and_starting_cards():
    expected = [
        card
        for card in CATALOGUE.values()
        if card.faction is None or card in FIRST_SLICE
    ]
    assert [
        {
            "name": card.name,
            "cost": card.cost,
            "faction": None if card.faction is None else card.faction.value,
            "type": card.type.value,
            "defense": card.defense,
            "primary": _effects(card.primary),
            "ally": _effects(card.ally),
            "scrap": _effects(card.scrap),
            "copies": card.copies,
        }
        for card in expected
    ] == native.cards()


def _play(seed: int, choose, max_turns: int) -> GameState:
    """Play a first-slice game from `seed` in both engines, comparing state
    and legal actions after every action; `choose(state, actions)` picks
    each decision from the Python engine's legal actions."""
    state = rules.auto_play(rules.new_game(seed, FIRST_SLICE))
    game = native.Game(seed)
    while True:
        assert game.state() == snapshot(state), f"seed {seed}, turn {state.turn}"
        actions = rules.legal_actions(state)
        assert game.legal_actions() == [native_action(a) for a in actions], (
            f"seed {seed}, turn {state.turn}"
        )
        assert game.winner() == rules.winner(state)
        if not actions or state.turn > max_turns:
            return state
        action = choose(state, actions)
        state = rules.advance(state, action)
        game.advance(*native_action(action))


@pytest.mark.parametrize("seed", range(GREEDY_GAMES))
def test_greedy_games_match_the_python_engine(seed: int):
    decisions = 0

    def greedy(state: GameState, actions: tuple[Action, ...]) -> Action:
        nonlocal decisions
        decisions += 1
        return greedy_move(state.must_scrap, actions)

    state = _play(seed, greedy, max_turns=1000)
    assert native.play_greedy(seed) == (rules.winner(state), state.turn, decisions)


def test_the_rust_greedy_bot_chooses_as_the_python_one():
    for seed in range(20):
        state = rules.auto_play(rules.new_game(seed, FIRST_SLICE))
        game = native.Game(seed)
        while (actions := rules.legal_actions(state)) and state.turn <= 1000:
            action = greedy_move(state.must_scrap, actions)
            assert game.greedy_move() == native_action(action)
            state = rules.advance(state, action)
            game.advance(*native_action(action))


@pytest.mark.parametrize("seed", range(RANDOM_GAMES))
def test_random_games_match_the_python_engine(seed: int):
    rng = random.Random(seed)
    _play(seed, lambda _, actions: rng.choice(actions), max_turns=RANDOM_MAX_TURNS)


def test_an_illegal_action_is_refused():
    game = native.Game(0)
    with pytest.raises(ValueError, match="not legal"):
        game.advance("Acquire", "Mothership")
    with pytest.raises(ValueError, match="unknown card"):
        game.advance("Acquire", "Stealth Needle")
    with pytest.raises(ValueError, match="unknown action"):
        game.advance("EndTurn", "Scout")


def test_the_initial_factions_played_are_empty():
    assert native.Game(0).state()["factions_played"] == list(NO_FACTIONS_PLAYED)
