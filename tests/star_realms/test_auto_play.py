import dataclasses
import functools
import random
from collections import Counter
from collections.abc import Callable

import pytest

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.action import (
    Acquire,
    Action,
    EndTurn,
    PlayCard,
    ScrapHandCard,
    UseAlly,
)
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
    assert not any(rules.automatic(after, a) for a in rules.legal_actions(after))
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
    assert not any(rules.automatic(after, a) for a in rules.legal_actions(after))


def test_a_ship_that_draws_is_left_to_the_player():
    state = with_hand(rules.new_game(seed=1), CORVETTE, SCOUT)
    after = rules.auto_play(state)
    assert after.players[0].hand == (CORVETTE,)
    assert PlayCard(CORVETTE) in rules.legal_actions(after)
    assert not rules.automatic(after, PlayCard(CORVETTE))


IMPERIAL_FIGHTER = CATALOGUE["Imperial Fighter"]  # 2 combat, opponent discards 1
ROYAL_REDOUBT = CATALOGUE["Royal Redoubt"]  # Star Empire outpost; ally: discards 1


def test_making_the_opponent_discard_is_a_gain_played_automatically():
    state = with_hand(rules.new_game(seed=1), IMPERIAL_FIGHTER, CORVETTE)
    state = with_bases(state, ROYAL_REDOUBT)
    after = rules.auto_play(state)
    assert after.players[0].hand == (CORVETTE,)
    assert set(after.allies_used) == {IMPERIAL_FIGHTER, ROYAL_REDOUBT}
    assert after.players[1].to_discard == 2
    # The Fighter's 2 and its ally ability's 2, and the Redoubt's 3.
    assert after.combat == 2 + 2 + 3


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
    assert not rules.automatic(after, PlayCard(BLOB_WHEEL))


TRADE_BOT = CATALOGUE["Trade Bot"]  # Machine Cult: 1 trade, may scrap hand/discard
BATTLE_MECH = CATALOGUE["Battle Mech"]  # 4 combat, may scrap; ally: draw
JUNKYARD = CATALOGUE["Junkyard"]  # outpost: may scrap from hand or discard pile
MACHINE_BASE = CATALOGUE["Machine Base"]  # outpost: draw, then scrap from hand


def test_scrap_abilities_are_granted_automatically_but_the_scraps_are_decisions():
    state = with_hand(rules.new_game(seed=1), BATTLE_MECH, CORVETTE)
    after = rules.auto_play(state)
    assert after.players[0].hand == (CORVETTE,)
    assert after.may_scrap == 1
    assert after.combat == 4
    state = with_bases(with_hand(rules.new_game(seed=1), CORVETTE), JUNKYARD)
    after = rules.auto_play(state)
    assert after.bases_used == (JUNKYARD,)
    assert after.may_scrap == 1
    assert not any(rules.automatic(after, a) for a in rules.legal_actions(after))


MISSILE_MECH = CATALOGUE["Missile Mech"]  # 6 combat, may destroy target base
COMMAND_SHIP = CATALOGUE["Command Ship"]  # ally: may destroy target base
THE_HIVE = CATALOGUE["The Hive"]  # Blob base


def test_destroy_abilities_are_granted_automatically_but_the_target_is_a_decision():
    state = with_hand(rules.new_game(seed=1), MISSILE_MECH, CORVETTE)
    second = dataclasses.replace(state.players[1], bases=(THE_HIVE, WAR_WORLD))
    state = dataclasses.replace(state, players=(state.players[0], second))
    after = rules.auto_play(state)
    assert after.players[0].hand == (CORVETTE,)
    assert after.may_destroy == 1
    assert after.combat == 6
    assert not any(rules.automatic(after, a) for a in rules.legal_actions(after))
    first = dataclasses.replace(after.players[0], in_play=(COMMAND_SHIP, CUTTER))
    after = rules.auto_play(dataclasses.replace(after, players=(first, second)))
    assert COMMAND_SHIP in after.allies_used
    assert after.may_destroy == 2


BATTLE_POD = CATALOGUE["Battle Pod"]  # Blob: 4 combat, may scrap from Trade Row
BLOB_DESTROYER = CATALOGUE["Blob Destroyer"]  # ally: destroy base, scrap row


def test_row_scraps_are_granted_automatically_but_the_card_is_a_decision():
    state = with_hand(rules.new_game(seed=1), BATTLE_POD, BLOB_DESTROYER)
    after = rules.auto_play(state)
    assert after.players[0].hand == ()
    assert BLOB_DESTROYER in after.allies_used
    assert after.may_scrap_row == 2
    assert after.combat == 12  # with Battle Pod's ally ability
    assert not any(rules.automatic(after, a) for a in rules.legal_actions(after))


def test_ships_wait_in_hand_while_something_could_scrap_them_from_hand():
    state = with_bases(with_hand(rules.new_game(seed=1), SCOUT, VIPER), JUNKYARD)
    after = rules.auto_play(state)
    assert after.may_scrap == 1
    assert after.players[0].hand == (SCOUT, VIPER)
    after = rules.advance(after, ScrapHandCard(VIPER))
    assert after.players[0].hand == ()
    assert after.trade == 1
    # So do they while another card in hand, played, could scrap them.
    for scrapper in (TRADE_BOT, MACHINE_BASE):
        state = with_hand(rules.new_game(seed=1), SCOUT, scrapper)
        assert SCOUT in rules.auto_play(state).players[0].hand
    # A card that scraps from hand doesn't keep itself waiting.
    state = with_hand(rules.new_game(seed=1), TRADE_BOT)
    assert rules.auto_play(state).players[0].hand == ()


# Outcome preservation: on small hand-built states, brute-force every way to
# play the turn with and without automatic play, and compare the outcomes.

type Step = Callable[[GameState, Action], GameState]
type Outcome = tuple[tuple[object, ...], int, int, int]


def outcome(state: GameState) -> Outcome:
    """What a turn leaves behind, as the turn is about to end, and both
    players' Authority (the player's, then the opponent's) and the discards
    the opponent has pending, kept apart for `dominated`. Left out: the trade
    and combat (lost at the end of the turn), the abilities used, and the
    order of the hand, ships in play and discard pile, which all go to the
    discard pile and are shuffled before they are drawn from."""
    me = state.players[state.current]
    opponent = state.players[1 - state.current]
    rest = (
        me.deck,
        names(me.hand + me.in_play + me.discard),
        names(me.bases),
        dataclasses.replace(opponent, authority=0, to_discard=0),
        state.trade_row,
        state.trade_deck,
        state.explorers,
        names(state.scrap_heap),
        state.rng_seed,
    )
    return rest, me.authority, opponent.authority, opponent.to_discard


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
    least as much Authority, the opponent at most as much and at least as
    many discards to make."""
    return auto <= manual and all(
        any(
            rest == other
            and mine >= my_authority
            and theirs <= their_authority
            and discards >= their_discards
            for other, mine, theirs, discards in auto
        )
        for rest, my_authority, their_authority, their_discards in manual
    )


def automated(state: GameState, action: Action) -> GameState:
    return rules.auto_play(rules.apply(state, action))


RAM = CATALOGUE["Ram"]  # Blob; ally 2 combat; scrap 3 trade
SPACE_STATION = CATALOGUE["Space Station"]  # Star Empire; scrap 4 trade
SURVEY_SHIP = CATALOGUE["Survey Ship"]  # Star Empire; draw; scrap: discards 1


def small(
    hand: tuple[Card, ...],
    deck: tuple[Card, ...] = (),
    discard: tuple[Card, ...] = (),
    bases: tuple[Card, ...] = (),
    trade_row: tuple[Card, ...] = (CUTTER,),
    in_play: tuple[Card, ...] = (),
) -> GameState:
    state = rules.new_game(seed=1)
    first = PlayerState(
        50, deck=deck, hand=hand, discard=discard, bases=bases, in_play=in_play
    )
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
    # Opponent discards: a ship's, a base's ally ability and a scrap ability's.
    "discards": small(
        hand=(IMPERIAL_FIGHTER, SURVEY_SHIP, SCOUT),
        deck=(VIPER,),
        bases=(ROYAL_REDOUBT,),
    ),
    # Scrapping from hand or discard pile: a ship's optional scrap, of a ship
    # in hand or a card in the discard pile, around a buy and a draw, beside
    # allies in play giving Authority.
    "scraps": small(
        hand=(TRADE_BOT, SCOUT, CORVETTE),
        deck=(VIPER,),
        discard=(VIPER,),
        in_play=(SHUTTLE, CUTTER),
    ),
    # Machine Base's draw then forced scrap from hand, beside the allies.
    "machine base": small(
        hand=(SCOUT, VIPER),
        deck=(VIPER, SCOUT),
        bases=(MACHINE_BASE,),
        in_play=(SHUTTLE, CUTTER),
    ),
    # Destroying a base: a ship's ability and an ally's, against the base
    # combat could destroy too, around a draw.
    "destroy": small(
        hand=(MISSILE_MECH, CORVETTE),
        deck=(SCOUT,),
        in_play=(COMMAND_SHIP, CUTTER),
    ),
    # Scrapping from the Trade Row: a ship's ability, beside its ally, around
    # buys from the row (before or after its refill) and a draw.
    "trade row": small(
        hand=(BATTLE_POD, TRADE_POD, CORVETTE),
        deck=(SCOUT,),
        trade_row=(CUTTER, BLOB_FIGHTER),
    ),
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
        rules,
        "automatic",
        lambda s, a: a == PlayCard(CORVETTE) or automatic(s, a),
    )
    auto = outcomes(rules.auto_play(state), automated)
    assert not dominated(manual, auto)


@pytest.mark.parametrize("name", ["scraps", "machine base"])
def test_the_outcome_check_catches_playing_ships_that_could_be_scrapped_from_hand(
    name: str, monkeypatch: pytest.MonkeyPatch
):
    state = SMALL_STATES[name]
    manual = outcomes(state, rules.apply)
    monkeypatch.setattr(rules, "_hand_scrap_ahead", lambda state, card: False)
    auto = outcomes(rules.auto_play(state), automated)
    assert not dominated(manual, auto)


# The automatic-play scan: auto_play must take the same automatic actions, in
# the same order, as its definition, which `old_auto_play` spells out (scan
# all of `legal_actions` for the first automatic action, apply it, repeat).


def old_auto_play(state: GameState) -> list[Action]:
    taken = []
    while action := next(
        (a for a in rules.legal_actions(state) if rules.automatic(state, a)), None
    ):
        taken.append(action)
        state = rules.apply(state, action)
    return taken


def auto_played(state: GameState) -> tuple[list[Action], GameState]:
    """The automatic actions `rules.auto_play` takes from `state`, and the
    state it returns."""
    taken: list[Action] = []
    return taken, rules.auto_play(state, lambda _, action: taken.append(action))


def unplayed_states(seed: int, moves: int = 250) -> list[GameState]:
    """The states of a random game from `seed` played with the bare rules,
    without automatic play, so automatic actions pile up. It buys whenever it
    can, so that later turns have bases and allies to use (long enough that
    some survive the random destroy-base choices)."""
    rng = random.Random(seed)
    state = rules.new_game(seed)
    states = [state]
    for _ in range(moves):
        actions = rules.legal_actions(state)
        if not actions:
            break
        buys = [a for a in actions if isinstance(a, Acquire)]
        state = rules.apply(state, rng.choice(buys or actions))
        states.append(state)
    return states


def test_auto_play_takes_the_automatic_actions_in_legal_actions_order():
    kinds: Counter[str] = Counter()
    for seed in range(25):
        for state in unplayed_states(seed):
            taken, after = auto_played(state)
            expected = old_auto_play(state)
            assert taken == expected
            assert after == functools.reduce(rules.apply, expected, state)
            kinds.update(type(a).__name__ for a in taken)
    # The states exercise every kind of automatic action.
    assert min(kinds[k] for k in ("PlayCard", "UseAlly", "UseBase")) >= 20


def test_auto_play_does_nothing_once_the_game_is_over():
    state = with_hand(rules.new_game(seed=1), SCOUT, VIPER)
    loser = dataclasses.replace(state.players[1], authority=0)
    state = dataclasses.replace(state, players=(state.players[0], loser))
    assert rules.auto_play(state) == state
