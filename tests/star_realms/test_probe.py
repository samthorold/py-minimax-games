"""Probes of a judgement (`probe`; docs/star-realms.md, How its understanding
is judged): positions that differ from real ones in one fact, and what a
judgement makes of the difference."""

import numpy as np
import pytest

from minimax_games.games.star_realms import features, fit, probe
from minimax_games.games.star_realms.state import GameState


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
