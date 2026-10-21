import random

import numpy as np

from minimax_games.games.star_realms import features, rules
from minimax_games.games.star_realms.action import EndTurn
from minimax_games.games.star_realms.catalogue import CATALOGUE
from minimax_games.games.star_realms.fit import boundaries
from minimax_games.games.star_realms.state import GameState


def boundary_states(seeds: range) -> list[GameState]:
    """Turn-boundary states of seeded heuristic-vs-heuristic games."""
    return [state for seed in seeds for state in boundaries(seed).states]


def test_a_row_has_one_value_per_feature_name():
    state = rules.new_game(0)
    row = features.encode(state)
    assert row.shape == (len(features.NAMES),)
    assert len(set(features.NAMES)) == len(features.NAMES)
    assert row.dtype == np.float64


def test_shuffling_a_deck_or_discard_pile_does_not_change_the_row():
    rng = random.Random(0)
    for state in boundary_states(range(3)):
        row = features.encode(state)
        for i, player in enumerate(state.players):
            deck, discard = list(player.deck), list(player.discard)
            rng.shuffle(deck)
            rng.shuffle(discard)
            shuffled = player.replace(deck=tuple(deck), discard=tuple(discard))
            players = (
                (shuffled, state.players[1]) if i == 0 else (state.players[0], shuffled)
            )
            assert np.array_equal(features.encode(state.replace(players=players)), row)


def test_which_cards_were_drawn_does_not_change_the_row():
    for state in boundary_states(range(3)):
        row = features.encode(state)
        players = tuple(
            p.replace(deck=p.hand + p.deck, hand=(), known_top=0) for p in state.players
        )
        assert players[0] != state.players[0] or players[1] != state.players[1]
        assert np.array_equal(
            features.encode(state.replace(players=(players[0], players[1]))), row
        )


def test_the_row_is_from_the_side_of_the_player_who_just_ended_their_turn():
    state = rules.auto_play(rules.new_game(0))
    ended = rules.apply(state, EndTurn())
    me, them = ended.players[0], ended.players[1].replace(authority=37)
    row = features.encode(ended.replace(players=(me, them)))
    named = dict(zip(features.NAMES, row, strict=True))
    assert named["ended: authority"] == 50
    assert named["next: authority"] == 37
    assert named["ended: pool Scout"] == 8
    assert named["turn"] == 2


def test_bases_in_play_are_counted_apart_from_the_pool():
    state = rules.new_game(0)
    outpost = CATALOGUE["Trading Post"]
    me = state.players[0].replace(bases=(outpost,))
    row = features.encode(state.replace(players=(me, state.players[1]), current=1))
    named = dict(zip(features.NAMES, row, strict=True))
    assert named["ended: base Trading Post"] == 1
    assert named["ended: pool Trading Post"] == 0
    assert named["ended: base defense"] == outpost.defense


def test_the_trade_row_is_counted():
    state = rules.new_game(0)
    row = dict(zip(features.NAMES, features.encode(state), strict=True))
    for card in state.trade_row:
        assert row[f"row {card.name}"] == state.trade_row.count(card)
    assert sum(v for k, v in row.items() if k.startswith("row ")) == 5


def test_encode_all_stacks_rows():
    states = boundary_states(range(1))[:4]
    matrix = features.encode_all(states)
    assert matrix.shape == (4, len(features.NAMES))
    for row, state in zip(matrix, states, strict=True):
        assert np.array_equal(row, features.encode(state))


def _extra(state: GameState) -> dict[str, float]:
    row = features.extra(features.encode(state)[None, :])[0]
    return dict(zip(features.EXTRA, row, strict=True))


def test_extra_features_are_computed_from_the_counts_one_column_per_name():
    rows = features.encode_all(boundary_states(range(2)))
    extra = features.extra(rows)
    assert extra.shape == (len(rows), len(features.EXTRA))
    assert len(set(features.EXTRA)) == len(features.EXTRA)
    assert not set(features.EXTRA) & set(features.NAMES)
    # a float32 dataset's rows give the same columns
    np.testing.assert_allclose(features.extra(rows.astype(np.float32)), extra)


def test_combat_per_hand_and_the_kill_threat_of_a_starting_deck():
    state = rules.auto_play(rules.new_game(0))
    ended = rules.apply(state, EndTurn())
    extra = _extra(ended)
    # 8 Scouts and 2 Vipers: 5 cards of 10 hold 1 combat on average
    assert extra["next: combat per hand"] == 1.0
    assert extra["next: kill threat"] == 1.0 - 50
    assert extra["ended: kill threat"] == 1.0 - 50
    assert extra["next: Blob density"] == 0.0
    assert extra["next: Blob pairs"] == 0.0


def test_faction_counts_and_bases_add_to_combat_and_threat():
    state = rules.auto_play(rules.new_game(0))
    ended = rules.apply(state, EndTurn())
    fighter = CATALOGUE["Blob Fighter"]  # 3 combat
    base = CATALOGUE["The Hive"]  # a Blob base: 3 combat
    nxt = ended.players[ended.current]
    nxt = nxt.replace(discard=(*nxt.discard, fighter, fighter), bases=(base,))
    other = ended.players[1 - ended.current].replace(bases=(CATALOGUE["Trading Post"],))
    players = [other, other]
    players[ended.current] = nxt
    extra = _extra(ended.replace(players=(players[0], players[1])))
    # 12 cards: 2 Vipers and 2 Fighters, 8 combat in all, 5 drawn, and the Hive
    assert extra["next: combat per hand"] == 5 * 8 / 12 + 3
    assert extra["next: kill threat"] == 5 * 8 / 12 + 3 - (50 + 4)
    assert extra["next: Blob density"] == 2 / 12
    assert extra["next: Blob pairs"] == 10 * 2 * 1 / (12 * 11)
    assert extra["next: Blob bases"] == 1
    assert extra["ended: Trade Federation bases"] == 1
    assert extra["next: Star Empire density"] == 0.0
