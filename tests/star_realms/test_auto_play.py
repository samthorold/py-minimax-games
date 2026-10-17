import dataclasses
from collections.abc import Callable

import pytest

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.action import Action, EndTurn, PlayCard, UseAlly
from minimax_games.games.star_realms.card import Card
from minimax_games.games.star_realms.catalogue import CATALOGUE, SCOUT, VIPER
from minimax_games.games.star_realms.state import GameState, PlayerState


def with_hand(state: GameState, *hand: Card) -> GameState:
    first = dataclasses.replace(state.players[0], hand=hand)
    return dataclasses.replace(state, players=(first, state.players[1]))


def test_auto_play_plays_pure_gain_ships_and_leaves_no_automatic_action():
    state = with_hand(rules.new_game(seed=1), SCOUT, VIPER, SCOUT)
    after = rules.auto_play(state)
    assert after.players[0].hand == ()
    assert sorted(c.name for c in after.players[0].in_play) == [
        "Scout",
        "Scout",
        "Viper",
    ]
    assert after.trade == 2
    assert after.combat == 1
    assert not any(rules.automatic(a) for a in rules.legal_actions(after))
    assert PlayCard(SCOUT) not in rules.legal_actions(after)


CUTTER = CATALOGUE["Cutter"]  # Trade Federation; ally: 4 combat
SHUTTLE = CATALOGUE["Federation Shuttle"]  # ally: 4 Authority
BLOB_FIGHTER = CATALOGUE["Blob Fighter"]  # Blob; ally: draw
TRADE_POD = CATALOGUE["Trade Pod"]  # Blob; ally: 2 combat
CORVETTE = CATALOGUE["Corvette"]  # primary: 1 combat, draw


def test_pure_gain_ally_abilities_are_used_automatically_but_draws_are_not():
    state = with_hand(rules.new_game(seed=1), CUTTER, SHUTTLE, BLOB_FIGHTER, TRADE_POD)
    after = rules.auto_play(state)
    assert after.players[0].hand == ()
    assert sorted(c.name for c in after.allies_used) == [
        "Cutter",
        "Federation Shuttle",
        "Trade Pod",
    ]
    assert rules.legal_actions(after)[0] == UseAlly(BLOB_FIGHTER)
    assert not any(rules.automatic(a) for a in rules.legal_actions(after))


def test_a_ship_that_draws_is_left_to_the_player():
    state = with_hand(rules.new_game(seed=1), CORVETTE, SCOUT)
    after = rules.auto_play(state)
    assert after.players[0].hand == (CORVETTE,)
    assert PlayCard(CORVETTE) in rules.legal_actions(after)
    assert not rules.automatic(PlayCard(CORVETTE))


WAR_WORLD = CATALOGUE["War World"]  # Star Empire outpost: 3 combat; ally 4 combat
BLOB_WHEEL = CATALOGUE["Blob Wheel"]  # Blob base: 1 combat


def with_bases(state: GameState, *bases: Card) -> GameState:
    first = dataclasses.replace(state.players[0], bases=bases)
    return dataclasses.replace(state, players=(first, state.players[1]))


def test_bases_in_play_are_used_automatically_but_playing_a_base_is_a_decision():
    state = with_hand(rules.new_game(seed=1), BLOB_WHEEL)
    state = with_bases(state, WAR_WORLD, WAR_WORLD)
    after = rules.auto_play(state)
    assert after.bases_used == (WAR_WORLD, WAR_WORLD)
    assert after.allies_used == (WAR_WORLD, WAR_WORLD)
    assert after.combat == 3 + 3 + 4 + 4
    assert after.players[0].hand == (BLOB_WHEEL,)
    assert not rules.automatic(PlayCard(BLOB_WHEEL))


# Outcome preservation: on small hand-built states, brute-force every way to
# play the turn with and without automatic play, and compare the outcomes.

type Step = Callable[[GameState, Action], GameState]
type Outcome = tuple[tuple[object, ...], int, int]


def outcome(state: GameState) -> Outcome:
    """What a turn leaves behind, as the turn is about to end, and both
    players' Authority (the player's, then the opponent's), kept apart for
    `dominated`. Left out: the trade and combat (lost at the end of the turn),
    the abilities used, and the order of the hand, ships in play and discard
    pile, which all go to the discard pile and are shuffled before they are
    drawn from."""
    me = state.players[state.current]
    opponent = state.players[1 - state.current]
    rest = (
        me.deck,
        names(me.hand + me.in_play + me.discard),
        names(me.bases),
        dataclasses.replace(opponent, authority=0),
        state.trade_row,
        state.trade_deck,
        state.explorers,
        names(state.scrap_heap),
        state.rng_seed,
    )
    return rest, me.authority, opponent.authority


def names(cards: tuple[Card, ...]) -> tuple[str, ...]:
    return tuple(sorted(card.name for card in cards))


def ordered(cards: tuple[Card, ...]) -> tuple[Card, ...]:
    return tuple(sorted(cards, key=lambda card: card.name))


def canonical(state: GameState) -> GameState:
    """`state` with the piles whose order can't matter for the rest of the turn
    sorted, so the search meets each position once. (The discard pile's order
    matters: a draw may shuffle it.)"""
    me = state.players[state.current]
    me = dataclasses.replace(
        me, hand=ordered(me.hand), in_play=ordered(me.in_play), bases=ordered(me.bases)
    )
    players = list(state.players)
    players[state.current] = me
    first, second = players
    return dataclasses.replace(
        state,
        players=(first, second),
        allies_used=ordered(state.allies_used),
        bases_used=ordered(state.bases_used),
    )


def outcomes(state: GameState, step: Step) -> set[Outcome]:
    """The outcome of every way to play out the current turn from `state`."""
    found: set[Outcome] = set()
    seen: set[GameState] = set()
    stack = [state]
    while stack:
        state = canonical(stack.pop())
        if state in seen:
            continue
        seen.add(state)
        actions = rules.legal_actions(state)
        if EndTurn() in actions or not actions:
            found.add(outcome(state))
        stack.extend(step(state, a) for a in actions if a != EndTurn())
    return found


def dominated(manual: set[Outcome], auto: set[Outcome]) -> bool:
    """Whether automatic play reaches only outcomes manual play reaches, and,
    for every manual outcome, one that is the same but leaves the player at
    least as much Authority and the opponent at most as much."""
    return auto <= manual and all(
        any(
            rest == other and mine >= my_authority and theirs <= their_authority
            for other, mine, theirs in auto
        )
        for rest, my_authority, their_authority in manual
    )


def automated(state: GameState, action: Action) -> GameState:
    return rules.auto_play(rules.apply(state, action))


RAM = CATALOGUE["Ram"]  # Blob; ally 2 combat; scrap 3 trade
SPACE_STATION = CATALOGUE["Space Station"]  # Star Empire; scrap 4 trade


def small(
    hand: tuple[Card, ...],
    deck: tuple[Card, ...] = (),
    discard: tuple[Card, ...] = (),
    bases: tuple[Card, ...] = (),
    trade_row: tuple[Card, ...] = (CUTTER,),
) -> GameState:
    state = rules.new_game(seed=1)
    first = PlayerState(50, deck=deck, hand=hand, discard=discard, bases=bases)
    second = dataclasses.replace(state.players[1], bases=(BLOB_WHEEL,))
    return dataclasses.replace(
        state,
        players=(first, second),
        trade_row=trade_row,
        trade_deck=(TRADE_POD,),
        explorers=1,
    )


SMALL_STATES = {
    # Ships, around a draw that reshuffles the discard pile.
    "ships": small(hand=(SCOUT, SCOUT, VIPER, CORVETTE), discard=(VIPER,)),
    # Ally abilities, with a draw ally and a scrap that breaks the alliance.
    "allies": small(hand=(RAM, TRADE_POD, BLOB_FIGHTER), deck=(SCOUT,)),
    # A base, with a reshuffling draw, an ally and scrapping the base.
    "bases": small(hand=(CORVETTE,), discard=(SCOUT,), bases=(SPACE_STATION,)),
}


@pytest.mark.parametrize("name", SMALL_STATES)
def test_automatic_play_keeps_every_end_of_turn_outcome(name: str):
    state = SMALL_STATES[name]
    manual = outcomes(state, rules.apply)
    auto = outcomes(rules.auto_play(state), automated)
    assert rules.auto_play(state) != state
    assert len(auto) < len(manual)
    assert dominated(manual, auto)


def test_the_outcome_check_catches_automating_a_draw(monkeypatch: pytest.MonkeyPatch):
    state = SMALL_STATES["ships"]
    manual = outcomes(state, rules.apply)
    automatic = rules.automatic
    monkeypatch.setattr(
        rules, "automatic", lambda a: a == PlayCard(CORVETTE) or automatic(a)
    )
    auto = outcomes(rules.auto_play(state), automated)
    assert not dominated(manual, auto)
