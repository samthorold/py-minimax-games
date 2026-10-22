"""Probes of a judgement (`probe`; docs/star-realms.md, How its understanding
is judged): positions that differ from real ones in one fact, and what a
judgement makes of the difference."""

import numpy as np
import pytest

from minimax_games._native import star_realms as native
from minimax_games.games.star_realms import features, fit, probe, selfplay
from minimax_games.games.star_realms.catalogue import CATALOGUE
from minimax_games.games.star_realms.state import GameState
from minimax_games.games.star_realms.value import load


def _moved(state: GameState, name: str) -> GameState:
    """`state` with one copy of `name` moved from the Trade Deck into the
    deck of the player who just ended: a valid position by construction,
    since every card is still in exactly one place."""
    i = next(i for i, card in enumerate(state.trade_deck) if card.name == name)
    card = state.trade_deck[i]
    ended = 1 - state.current
    player = state.players[ended]
    players = list(state.players)
    players[ended] = player.replace(deck=(*player.deck, card))
    return state.replace(
        players=(players[0], players[1]),
        trade_deck=state.trade_deck[:i] + state.trade_deck[i + 1 :],
    )


def test_adding_a_card_is_the_position_with_a_copy_moved_from_the_trade_deck():
    game = fit.boundaries(3)
    checked = 0
    for state in game.states[::5]:
        for name in {card.name for card in state.trade_deck}:
            row = features.encode(state)
            assert np.array_equal(
                probe.add_card(row, name), features.encode(_moved(state, name))
            )
            checked += 1
    assert checked > 100


def test_the_copies_available_are_those_in_the_trade_deck_and_scrap_heap():
    # the row can't tell the Trade Deck from the scrap heap: both are out of
    # sight, so a copy in either counts as one a player could have bought
    game = fit.boundaries(4)
    for state in game.states[::3]:
        row = features.encode(state)
        for name in features.CARDS:
            hidden = sum(c.name == name for c in (*state.trade_deck, *state.scrap_heap))
            if name in ("Scout", "Viper", "Explorer"):
                continue
            assert probe.available(row, name) == hidden


def test_a_card_with_no_copy_left_can_t_be_added():
    row = features.encode(fit.boundaries(3).states[10])
    name = "Battle Pod"  # 2 copies
    full = row.copy()
    full[features.NAMES.index(f"next: pool {name}")] += probe.available(row, name)
    assert probe.available(full, name) == 0
    with pytest.raises(ValueError, match="no copy"):
        probe.add_card(full, name)


@pytest.fixture(scope="module")
def rows() -> np.ndarray:
    """Real positions: every turn's end of a few heuristic games."""
    return features.encode_all(
        state for seed in range(3) for state in fit.boundaries(seed).states
    )


def _pool(name: str) -> int:
    return features.NAMES.index(f"ended: pool {name}")


def _linear(weights: dict[str, float]) -> probe.Judgement:
    """A judgement linear in the counts of the ended player's cards, each
    card of a function weighing that function's weight."""
    w = np.zeros(len(features.NAMES))
    for function, weight in weights.items():
        for name in probe.FUNCTIONS[function]:
            w[_pool(name)] = weight

    def judge(x: np.ndarray) -> np.ndarray:
        return 0.3 + np.asarray(x) @ w

    return judge


def test_a_linear_judgement_gains_as_much_from_every_card_of_a_function(
    rows: np.ndarray,
):
    for function in ("draw", "scrap", "discard"):
        found = probe.synergy(_linear({function: 0.01}), rows, seed=1)
        gains = found.gains[function]
        assert gains.shape == (len(rows), 3)
        assert np.allclose(gains, 0.01)
        assert np.allclose(found.gains["none"], 0)


def test_a_judgement_with_same_function_synergy_gains_more_from_each_card(
    rows: np.ndarray,
):
    # the chance grows with the square of the draw cards owned: the k-th one
    # added to n already owned adds 0.001 (2(n + k) - 1), 0.002 more each time
    draws = [_pool(name) for name in probe.FUNCTIONS["draw"]]

    def judge(x: np.ndarray) -> np.ndarray:
        return 0.3 + 0.001 * np.asarray(x)[:, draws].sum(axis=1) ** 2

    found = probe.synergy(judge, rows, seed=2)
    gains = found.gains["draw"]
    owned = rows[:, draws].sum(axis=1)
    assert np.allclose(gains[:, 0], 0.001 * (2 * owned + 1))
    assert np.allclose(np.diff(gains, axis=1), 0.002)
    assert np.allclose(found.gains["none"], 0)


def test_on_the_logit_scale_a_logistic_judgement_shows_no_synergy(rows: np.ndarray):
    # a chance that saturates gains less from each card near 0 or 1, though
    # its score is linear: the logit scale takes the saturation out
    linear = _linear({"draw": 0.4})

    def judge(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(linear(x) - 0.3)))

    found = probe.synergy(judge, rows, seed=3)
    assert not np.allclose(np.diff(found.gains["draw"], axis=1), 0)
    assert np.allclose(found.logit_gains["draw"], 0.4)


def test_every_position_the_synergy_probe_judges_is_valid_and_differs_in_one_fact(
    rows: np.ndarray,
):
    judged: list[np.ndarray] = []

    def judge(x: np.ndarray) -> np.ndarray:
        judged.append(np.array(x))
        return np.full(len(x), 0.5)

    probe.synergy(judge, rows, samples=2)
    assert len(judged) == len(probe.FUNCTIONS)
    pools = [_pool(name) for name in features.CARDS]
    real = {row.tobytes() for row in rows.astype(np.float64)}
    for function, x in zip(probe.FUNCTIONS, judged, strict=True):
        steps = x.reshape(-1, probe.DEPTH + 1, x.shape[1]).astype(np.float64)
        # each sequence starts at a real position, and each step after it
        # adds one card of the function to the ended player's deck
        assert all(start.tobytes() in real for start in steps[:, 0])
        added = np.diff(steps, axis=1)
        others = np.ones(x.shape[1], dtype=bool)
        others[pools] = False
        assert (added[:, :, others] == 0).all()
        assert (added.sum(axis=2) == 1).all()
        functional = [_pool(name) for name in probe.FUNCTIONS[function]]
        assert (added[:, :, functional].sum(axis=2) == 1).all()
        for row in x[:: max(1, len(x) // 300)]:
            for name in probe.FUNCTIONS[function]:
                assert probe.available(row, name) >= 0


def test_the_summary_gives_each_card_s_gain_and_its_spread_by_phase(
    rows: np.ndarray,
):
    draws = [_pool(name) for name in probe.FUNCTIONS["draw"]]

    def judge(x: np.ndarray) -> np.ndarray:
        return 0.3 + 0.001 * np.asarray(x)[:, draws].sum(axis=1) ** 2

    found = probe.synergy(judge, rows, seed=2)
    summary = probe.summary(found)
    turn = rows[:, features.NAMES.index("turn")]
    owned = rows[:, draws].sum(axis=1)
    for phase, (first, last) in probe.PHASES.items():
        at = (turn >= first) & (turn <= last)
        line = next(s for s in summary if s.phase == phase and s.function == "draw")
        assert line.positions == at.sum() > 0
        expected = 0.001 * (2 * owned[at] + 1)
        assert np.isclose(line.mean[0], expected.mean())
        assert np.isclose(line.sd[0], expected.std())
        assert np.allclose(line.mean[1:] - line.mean[:-1], 0.002)
        assert np.allclose(line.step, 0.002)
        assert np.allclose(line.step_se, 0)
        assert line.more == 1.0
    every = next(s for s in summary if s.phase == "all" and s.function == "none")
    assert every.positions == len(rows)
    assert np.allclose(every.mean, 0)
    assert every.more == 0.0


@pytest.mark.filterwarnings("error")
def test_a_position_with_no_copies_left_to_add_is_left_out(rows: np.ndarray):
    taken = rows[:2].copy()
    for name in probe.FUNCTIONS["discard"]:
        taken[0, features.NAMES.index(f"next: pool {name}")] += probe.available(
            taken[0], name
        )
    found = probe.synergy(_linear({"discard": 0.01}), taken)
    assert np.isnan(found.gains["discard"][0]).all()
    assert np.allclose(found.gains["discard"][1], 0.01)


def _per_unit(trade: float, combat: float) -> probe.Judgement:
    """A judgement linear in the trade and the combat the ended player's
    cards give when played: `trade` per unit of trade, `combat` per unit of
    combat."""
    w = np.zeros(len(features.NAMES))
    for name in features.CARDS:
        card = CATALOGUE[name]
        w[_pool(name)] = trade * probe.output(card, "trade") + combat * probe.output(
            card, "combat"
        )

    def judge(x: np.ndarray) -> np.ndarray:
        return 0.3 + np.asarray(x) @ w

    return judge


def test_a_judgement_linear_in_trade_and_combat_gains_its_weight_per_unit(
    rows: np.ndarray,
):
    found = probe.investment(_per_unit(0.004, 0.001), rows)
    assert found.pairs
    for economy, combat in found.pairs:
        trade = probe.output(CATALOGUE[economy], "trade")
        hits = probe.output(CATALOGUE[combat], "combat")
        made = ~np.isnan(found.chance.card_gap[economy, combat])
        assert made.sum() > len(rows) / 2
        assert np.allclose(found.chance.trade[economy][made], 0.004)
        assert np.allclose(found.chance.combat[combat][made], 0.001)
        assert np.allclose(
            found.chance.card_gap[economy, combat][made], 0.004 * trade - 0.001 * hits
        )


def test_the_comparable_cards_are_ships_of_one_cost_giving_only_trade_or_combat():
    assert probe.PAIRS == (
        ("Federation Shuttle", "Blob Fighter"),
        ("Freighter", "Blob Destroyer"),
    )
    assert probe.output(CATALOGUE["Patrol Mech"], "trade") == 3
    assert probe.output(CATALOGUE["Patrol Mech"], "combat") == 5
    assert probe.output(CATALOGUE["Command Ship"], "combat") == 5


def test_every_position_the_investment_probe_judges_is_valid_and_differs_in_one_card(
    rows: np.ndarray,
):
    judged: list[np.ndarray] = []

    def judge(x: np.ndarray) -> np.ndarray:
        judged.append(np.array(x))
        return np.full(len(x), 0.5)

    probe.investment(judge, rows)
    real = {row.tobytes() for row in rows.astype(np.float64)}
    allowed = {
        _pool(name) for pair in probe.PAIRS for name in pair
    }  # the pair's cards, in the ended player's pool
    for x in judged:
        for row in x:
            if row.tobytes() in real:
                continue
            changed = np.flatnonzero(np.abs(x[: len(rows)] - row).sum(axis=1) == 1)
            assert len(changed), "a judged row is one card away from a real one"
            column = int(np.flatnonzero(row - x[changed[0]])[0])
            assert column in allowed


def test_on_the_logit_scale_a_logistic_judgement_gains_its_weight_per_unit(
    rows: np.ndarray,
):
    linear = _per_unit(0.3, 0.1)

    def judge(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(linear(x) - 3)))

    found = probe.investment(judge, rows)
    for economy, combat in found.pairs:
        made = ~np.isnan(found.chance.card_gap[economy, combat])
        assert not np.allclose(found.chance.trade[economy][made], 0.3)
        assert np.allclose(found.logit.trade[economy][made], 0.3)
        assert np.allclose(found.logit.combat[combat][made], 0.1)


def test_the_investment_summary_gives_trade_against_combat_by_phase(
    rows: np.ndarray,
):
    # trade weighs 0.004 a unit up to turn 8 and 0.001 after; combat 0.002
    early, late = _per_unit(0.004, 0.002), _per_unit(0.001, 0.002)
    turn = rows[:, features.NAMES.index("turn")]

    def judge(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        at = x[:, features.NAMES.index("turn")] <= 8
        return np.where(at, early(x), late(x))

    found = probe.investment(judge, rows)
    summary = probe.investment_summary(found)
    for phase, (first, last) in probe.PHASES.items():
        weight = 0.004 if phase == "early" else 0.001
        for economy, combat in probe.PAIRS:
            line = next(
                s for s in summary if s.phase == phase and s.pair == (economy, combat)
            )
            made = ~np.isnan(found.chance.card_gap[economy, combat])
            at = (turn >= first) & (turn <= last) & made
            assert line.positions == at.sum() > 0
            assert np.isclose(line.trade, weight)
            assert np.isclose(line.combat, 0.002)
            assert np.isclose(line.trade_sd, 0)
            assert np.isclose(line.unit_gap, weight - 0.002)
            assert np.isclose(line.unit_gap_se, 0)
            trade = probe.output(CATALOGUE[economy], "trade")
            hits = probe.output(CATALOGUE[combat], "combat")
            assert np.isclose(line.card_gap, weight * trade - 0.002 * hits)
            assert line.economy_more == float(weight * trade > 0.002 * hits)
    every = [s for s in summary if s.phase == "all"]
    assert len(every) == len(probe.PAIRS)
    logit = probe.investment_summary(found, logit=True)
    assert len(logit) == len(summary)


@pytest.mark.filterwarnings("error")
def test_a_pair_with_no_copy_left_of_either_card_leaves_the_position_out(
    rows: np.ndarray,
):
    economy, combat = probe.PAIRS[0]
    taken = rows[:3].copy()
    for i, name in ((0, economy), (1, combat)):
        taken[i, features.NAMES.index(f"next: pool {name}")] += probe.available(
            taken[i], name
        )
    found = probe.investment(_per_unit(0.004, 0.001), taken)
    gap = found.chance.card_gap[economy, combat]
    assert np.isnan(gap[:2]).all()
    assert np.isnan(found.chance.trade[economy][:2]).all()
    assert np.isnan(found.chance.combat[combat][:2]).all()
    assert not np.isnan(gap[2])


def test_a_buy_is_economy_or_combat_by_what_its_card_s_primary_ability_gives():
    kinds = {name: probe.kind(CATALOGUE[name]) for name in probe.BUYABLE}
    assert "Scout" not in kinds
    assert "Viper" not in kinds
    assert kinds["Explorer"] == "economy"
    assert kinds["Freighter"] == "economy"
    assert kinds["Trading Post"] == "economy"  # Authority or trade
    assert kinds["Survey Ship"] == "economy"  # trade, and a draw
    assert kinds["Blob Fighter"] == "combat"
    assert kinds["Defense Center"] == "combat"  # Authority or combat
    assert kinds["Fleet HQ"] == "combat"  # every ship gains combat
    assert kinds["Patrol Mech"] == "mixed"  # trade or combat
    assert kinds["Battle Station"] == "neither"
    assert kinds["Stealth Needle"] == "neither"
    assert set(kinds.values()) == {"economy", "combat", "mixed", "neither"}


def test_buys_are_attributed_to_the_turn_and_seat_they_were_made_in():
    games = [
        [
            ("Acquire", "Explorer"),
            ("AttackPlayer",),
            ("EndTurn",),
            ("EndTurn",),
            ("PlayCard", "Freighter"),
            ("AcquireOnTop", "Blob Fighter"),
            ("Acquire", "Patrol Mech"),
            ("EndTurn",),
        ],
        [("EndTurn",), ("Acquire", "Battle Station"), ("AcquireFreeShip", "Ram")],
    ]
    found = probe.buys(games)
    assert found.card == (
        "Explorer",
        "Blob Fighter",
        "Patrol Mech",
        "Battle Station",
    )
    assert found.game.tolist() == [0, 0, 0, 1]
    assert found.turn.tolist() == [1, 3, 3, 2]
    assert found.seat.tolist() == [0, 0, 0, 1]
    assert found.kind == ("economy", "combat", "mixed", "neither")


def test_buy_shares_count_each_kind_by_band_of_turns():
    end = ("EndTurn",)
    game = [
        ("Acquire", "Explorer"),
        ("Acquire", "Blob Fighter"),
        end,
        ("Acquire", "Freighter"),
        end,
        ("Acquire", "Ram"),
        end,
        ("Acquire", "Patrol Mech"),
        ("Acquire", "Battle Station"),
    ]
    shares = probe.buy_shares(probe.buys([game]), bands=((1, 2), (3, 10)))
    assert [s.band for s in shares] == [(1, 10), (1, 2), (3, 10)]
    whole, first, second = shares
    assert first.buys == 3
    assert first.share == {
        "economy": 2 / 3,
        "combat": 1 / 3,
        "mixed": 0.0,
        "neither": 0.0,
    }
    assert second.buys == 3
    assert second.share["combat"] == 1 / 3
    assert second.share["mixed"] == 1 / 3
    assert whole.buys == 6
    assert whole.share["economy"] == 2 / 6
    # the Explorer, always in the market, is economy, and counted apart too
    assert first.explorer == 1 / 3
    assert second.explorer == 0.0
    empty = probe.buy_shares(probe.buys([game]), bands=((20, 30),))[1]
    assert empty.buys == 0
    assert np.isnan(empty.share["economy"])


def test_the_champion_s_recorded_games_are_the_games_its_positions_come_from():
    champion = selfplay.Planner(load("pf01"), worlds=1, budget=40)
    seeds = range(700, 703)
    games = probe.played(champion, seeds, threads=2)
    assert games.actions is not None
    assert len(games.actions) == len(seeds)
    assert np.array_equal(games.x, probe.positions(champion, seeds, threads=2))


def test_the_buys_of_real_games_are_where_replaying_them_puts_them():
    seeds = range(500, 503)
    played = selfplay.play([(s, "heuristic", "heuristic") for s in seeds], record=True)
    assert played.actions is not None
    found = probe.buys(played.actions)
    expected = []
    for i, (seed, actions) in enumerate(zip(seeds, played.actions, strict=True)):
        game = native.Game(seed)
        for action in actions:
            if action[0] in ("Acquire", "AcquireOnTop"):
                state = game.state()
                expected.append((i, state["turn"], state["current"], action[1]))
            game.advance(action)
    assert len(expected) > 30
    assert (
        list(
            zip(
                found.game.tolist(),
                found.turn.tolist(),
                found.seat.tolist(),
                found.card,
                strict=True,
            )
        )
        == expected
    )
