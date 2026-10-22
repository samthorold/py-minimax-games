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


def _by_faction(weights: dict[str, float]) -> probe.Judgement:
    """A judgement linear in the counts of the ended player's cards, each
    card of a faction weighing that faction's weight."""
    w = np.zeros(len(features.NAMES))
    for faction, weight in weights.items():
        for name in probe.FACTIONS[faction]:
            w[_pool(name)] = weight

    def judge(x: np.ndarray) -> np.ndarray:
        return 0.3 + np.asarray(x) @ w

    return judge


def _column(name: str) -> int:
    return features.NAMES.index(name)


_TRADE_DECK = [name for name in features.CARDS if name in probe.TRADE_DECK]
_WORTH = {name: 0.001 * (i + 1) for i, name in enumerate(_TRADE_DECK)}
"""A distinct worth for each Trade Deck card, so the one the opponent values
most and least in each position is known."""


def _denying(share: float) -> probe.Judgement:
    """A judgement in which each card the next player owns costs the player
    who just ended its worth, and each card in the Trade Row `share` of it:
    the row is seen as bearing on the next buyer by `share`."""
    w = np.zeros(len(features.NAMES))
    for name, worth in _WORTH.items():
        w[_column(f"next: pool {name}")] = -worth
        w[_column(f"row {name}")] = -share * worth

    def judge(x: np.ndarray) -> np.ndarray:
        return 0.5 + np.asarray(x) @ w

    return judge


def test_a_linear_judgement_gains_a_faction_s_weight_from_its_card_in_any_deck(
    rows: np.ndarray,
):
    weights = {"Blob": 0.01, "Trade Federation": 0.02, "Star Empire": 0.03}
    found = probe.faction_pairs(_by_faction(weights), rows, seed=1)
    for faction in probe.FACTIONS:
        assert found.gains[faction].shape == (len(rows),)
        assert np.allclose(found.gains[faction], weights.get(faction, 0))


def _deck(cards: dict[str, int], bases: dict[str, int] | None = None) -> np.ndarray:
    """A row whose ended player owns `cards` (each faction's first ships)
    and has `bases` of each faction in play; the next player owns one of
    every Trade Deck card, which never counts."""
    row = np.zeros(len(features.NAMES))
    for name in features.CARDS:
        if name in probe.FACTIONS["Machine Cult"]:
            row[features.NAMES.index(f"next: pool {name}")] = 1
    for faction, n in cards.items():
        ships = [s for s in probe.FACTIONS[faction] if s not in features.BASES]
        row[_pool(ships[0])] += n
    for faction, n in (bases or {}).items():
        base = next(b for b in probe.FACTIONS[faction] if b in features.BASES)
        row[features.NAMES.index(f"ended: base {base}")] += n
    return row


def test_a_deck_is_mostly_one_faction_when_it_holds_a_strict_majority_of_enough_cards():
    blob, trade, star, cult = probe.FACTIONS
    rows = np.array(
        [
            _deck({blob: 3, trade: 1}),
            _deck({blob: 2, trade: 2}),  # a tie is no majority
            _deck({star: 2, trade: 1}),  # too few faction cards to tell
            _deck({cult: 2, trade: 1}, bases={cult: 1}),  # a base in play counts
            _deck({}),
            _deck({trade: 5, blob: 2, star: 2}),
        ]
    )
    assert probe.LEAST == 4
    assert probe.mostly(rows).tolist() == [0, -1, -1, 3, -1, 1]


def _owned(rows: np.ndarray, faction: str) -> np.ndarray:
    return rows[:, [_pool(name) for name in probe.FACTIONS[faction]]].sum(axis=1)


def _blob_and_trade(x: np.ndarray) -> np.ndarray:
    """The chance grows with the product of Blob and Trade Federation cards
    owned: a Trade Federation card adds 0.001 per Blob card, and the other
    way round, and no other card adds anything."""
    x = np.asarray(x)
    return 0.3 + 0.001 * _owned(x, "Blob") * _owned(x, "Trade Federation")


@pytest.fixture(scope="module")
def many(rows: np.ndarray) -> np.ndarray:
    """Valid positions with decks mostly of every faction: each real
    position, and from it, per faction, the position with cards of that
    faction added (`add_card`) until the deck is mostly of it, where enough
    copies are left."""
    made = [rows]
    for i, names in enumerate(probe.FACTIONS.values()):
        for row in rows:
            while probe.mostly(row[None])[0] != i:
                name = max(names, key=lambda name: probe.available(row, name))
                if probe.available(row, name) < 1:
                    break
                row = probe.add_card(row, name)
            else:
                made.append(row[None])
    return np.concatenate(made)


def test_the_pair_table_gives_each_faction_s_gain_in_decks_mostly_of_each(
    many: np.ndarray,
):
    found = probe.faction_pairs(_blob_and_trade, many, seed=4)
    deck = probe.mostly(many)
    table = probe.pairs(found)
    assert len(table) == 16
    for cell in table:
        # positions with no card of the faction left to add are left out
        at = deck == list(probe.FACTIONS).index(cell.deck)
        at &= ~np.isnan(found.gains[cell.card])
        assert cell.positions == at.sum() > 1
        partner = {"Blob": "Trade Federation", "Trade Federation": "Blob"}
        if cell.card in partner:
            expected = 0.001 * _owned(many[at], partner[cell.card])
        else:
            expected = np.zeros(at.sum())
        assert np.isclose(cell.mean, expected.mean())
        assert np.isclose(cell.sd, expected.std())
        assert np.isclose(cell.se, expected.std(ddof=1) / np.sqrt(at.sum()))


def _interactions(table: list[probe.Pair]) -> np.ndarray:
    return np.array([cell.interaction for cell in table]).reshape(4, 4)


def test_a_pair_s_interaction_is_what_it_adds_beyond_its_deck_and_its_card(
    many: np.ndarray,
):
    # a card's own worth (its column) and a deck's need of any card (its
    # row) aren't the pair's: the interaction takes both out
    weights = {"Blob": 0.01, "Trade Federation": 0.02, "Star Empire": 0.03}
    linear = probe.pairs(probe.faction_pairs(_by_faction(weights), many))
    assert np.allclose(_interactions(linear), 0)

    table = probe.pairs(probe.faction_pairs(_blob_and_trade, many, seed=5))
    found = _interactions(table)
    assert np.allclose(found.sum(axis=0), 0)
    assert np.allclose(found.sum(axis=1), 0)
    means = np.array([cell.mean for cell in table]).reshape(4, 4)
    expected = means - means.mean(axis=1, keepdims=True) - means.mean(axis=0)
    assert np.allclose(found, expected + means.mean())
    top = sorted(table, key=lambda cell: cell.interaction)[-2:]
    assert {(cell.deck, cell.card) for cell in top} == {
        ("Blob", "Trade Federation"),
        ("Trade Federation", "Blob"),
    }


def test_a_pair_s_interaction_has_the_standard_error_of_its_decks_positions(
    many: np.ndarray,
):
    weights = {"Blob": 0.01, "Trade Federation": 0.02, "Star Empire": 0.03}
    linear = probe.pairs(probe.faction_pairs(_by_faction(weights), many))
    assert np.allclose([cell.interaction_se for cell in linear], 0)

    # the interaction is, per row of decks, a mean over that row's own
    # positions of each card's gain less the mean of the four cards' gains,
    # and the rows' positions are disjoint, so their variances add
    found = probe.faction_pairs(_blob_and_trade, many, seed=5)
    left = ~np.isnan(np.stack(list(found.gains.values()), axis=1)).any(axis=1)
    gains = np.zeros((len(many), 4))
    gains[:, 0] = 0.001 * _owned(many, "Trade Federation")
    gains[:, 1] = 0.001 * _owned(many, "Blob")
    centred = gains - gains.mean(axis=1, keepdims=True)
    for cell in probe.pairs(found):
        i = list(probe.FACTIONS).index(cell.deck)
        j = list(probe.FACTIONS).index(cell.card)
        variance = 0.0
        for k in range(4):
            h = centred[(found.deck == k) & left, j]
            weight = (k == i) - 1 / 4
            variance += weight**2 * h.var(ddof=1) / len(h)
        assert np.isclose(cell.interaction_se, np.sqrt(variance))


def test_the_denial_probe_finds_how_much_of_the_opponent_s_gain_the_row_shows(
    rows: np.ndarray,
):
    found = probe.denial(_denying(0.25), rows)
    out_of_sight = np.array(
        [[probe.available(row, name) >= 1 for name in _TRADE_DECK] for row in rows]
    )
    worths = np.array([_WORTH[name] for name in _TRADE_DECK])
    most = np.where(out_of_sight, worths, -np.inf).max(axis=1)
    least = np.where(out_of_sight, worths, np.inf).min(axis=1)
    assert np.allclose(found.valued, most - least)
    assert np.allclose(found.effect, -0.25 * (most - least))
    assert np.allclose(probe.denial(_denying(0), rows).effect, 0)


def test_every_position_the_denial_probe_judges_is_valid_and_differs_in_one_card(
    rows: np.ndarray,
):
    judged: list[np.ndarray] = []

    def judge(x: np.ndarray) -> np.ndarray:
        judged.append(np.array(x, dtype=np.float64))
        return np.full(len(x), 0.5)

    probe.denial(judge, rows)
    x = np.concatenate(judged)
    real = rows.astype(np.float64)
    nexts = [_column(f"next: pool {name}") for name in _TRADE_DECK]
    shown = [_column(f"row {name}") for name in _TRADE_DECK]
    swaps = 0
    for changed in x[:: max(1, len(x) // 2000)]:
        # each is a real position with one copy out of sight added to the
        # next player's pool, or one Trade Row card swapped for one out of
        # sight: the row still shows 5 cards and no count is negative
        diff = changed - real
        near = np.flatnonzero(np.abs(diff).sum(axis=1) <= 2)
        assert len(near)
        d = diff[near[0]]
        assert d[nexts].sum() == 1 or (d[shown].sum() == 0 and d.sum() == 0)
        swaps += d[nexts].sum() == 0
        assert changed[shown].sum() == real[near[0]][shown].sum()
        for name in _TRADE_DECK:
            assert probe.available(changed, name) >= 0
    assert swaps > 0


def test_the_denial_summary_gives_the_effect_and_its_spread_by_phase(
    rows: np.ndarray,
):
    found = probe.denial(_denying(0.25), rows)
    summary = probe.denial_summary(found)
    turn = rows[:, features.NAMES.index("turn")]
    assert [line.phase for line in summary] == ["all", *probe.PHASES]
    for line in summary:
        first, last = probe.PHASES.get(line.phase, (0, 10**9))
        at = (turn >= first) & (turn <= last)
        assert line.positions == at.sum() > 0
        assert np.isclose(line.mean, found.effect[at].mean())
        assert np.isclose(line.sd, found.effect[at].std())
        assert np.isclose(line.se, found.effect[at].std(ddof=1) / np.sqrt(at.sum()))
        assert np.isclose(line.valued, found.valued[at].mean())
        assert np.isclose(line.seen, 0.25)
        assert line.below == 1.0
    logit = probe.denial_summary(found, logit=True)[0]
    assert np.isclose(logit.mean, np.nanmean(found.logit_effect))


@pytest.fixture(scope="module")
def recorded() -> selfplay.SelfPlay:
    """Heuristic games played in Rust with every decision recorded."""
    seeds = range(probe.BASE_SEED, probe.BASE_SEED + 6)
    return selfplay.play(
        selfplay.alternating("heuristic", "heuristic", seeds), record=True
    )


_TAKES = ("Acquire", "AcquireOnTop", "AcquireFreeShip")


def test_each_trade_row_buy_and_scrap_is_found_at_the_end_of_its_turn(
    recorded: selfplay.SelfPlay,
):
    found = probe.choices(recorded)
    assert recorded.actions is not None
    expected = []
    for g, (seed, actions) in enumerate(
        zip(range(probe.BASE_SEED, probe.BASE_SEED + 6), recorded.actions, strict=True)
    ):
        if recorded.winners[g] is None:
            continue
        rows = np.flatnonzero(recorded.game == g)
        game = native.Game(seed)
        turn = 0
        for action in actions:
            kind = action[0]
            if kind in (*_TAKES, "ScrapTradeRowCard") and action[1] in probe.TRADE_DECK:
                state = game.state()
                assert action[1] in state["trade_row"]
                if turn < len(rows):
                    row = rows[turn]
                    assert recorded.seat[row] == state["current"]
                    expected.append((row, action[1], kind == "ScrapTradeRowCard"))
            turn += kind == "EndTurn"
            game.advance(action)
        assert turn == len(rows)
    got = [
        (row, probe.TRADE_DECK[card], scrap)
        for row, card, scrap in zip(found.row, found.card, found.scrap, strict=True)
    ]
    assert got == expected
    assert found.scrap.any()
    assert (~found.scrap).sum() > 50
    # a card bought is in the buyer's draw pool when their turn ends
    for row, name, scrap in got:
        if not scrap:
            assert recorded.x[row, _column(f"ended: pool {name}")] >= 1


def test_a_choice_s_worth_to_its_chooser_and_to_the_opponent_are_judged(
    recorded: selfplay.SelfPlay,
):
    # each card the player who just ended owns adds its worth to their
    # chance, and each the next player owns takes away the worth of the
    # card at the other end of the catalogue: half the cards are worth more
    # to the opponent, and choosing one of those is denial
    own = dict(zip(_TRADE_DECK, _WORTH.values(), strict=True))
    theirs = dict(zip(_TRADE_DECK, reversed(_WORTH.values()), strict=True))
    w = np.zeros(len(features.NAMES))
    for name in _TRADE_DECK:
        w[_column(f"ended: pool {name}")] = own[name]
        w[_column(f"next: pool {name}")] = -theirs[name]

    def judge(x: np.ndarray) -> np.ndarray:
        return 0.5 + np.asarray(x) @ w

    found = probe.choices(recorded)
    worth = probe.worth(judge, recorded.x, found)
    names = [probe.TRADE_DECK[c] for c in found.card]
    assert np.allclose(worth.own, [own[name] for name in names])
    assert np.allclose(worth.kept, [theirs[name] for name in names])
    assert np.array_equal(worth.denial, [own[n] < theirs[n] for n in names])
    assert 0 < worth.denial.mean() < 1


def test_the_denial_share_is_given_for_buys_and_scraps_by_phase():
    chosen = probe.Choices(
        row=np.arange(6),
        card=np.zeros(6, dtype=int),
        scrap=np.array([False, False, False, True, False, True]),
    )
    worth = probe.Worth(
        own=np.array([0.1, 0.1, 0.3, 0.1, np.nan, 0.2]),
        kept=np.array([0.2, 0.0, 0.0, 0.2, np.nan, 0.1]),
        denial=np.array([True, False, False, True, False, False]),
        turn=np.array([1, 2, 20, 3, 4, 30]),
    )
    shares = {(s.phase, s.kind): s for s in probe.denial_shares(worth, chosen)}
    every = shares["all", "buy"]
    assert every.choices == 3  # the one with no judgement is left out
    assert np.isclose(every.share, 1 / 3)
    assert np.isclose(every.own, (0.1 + 0.1 + 0.3) / 3)
    assert np.isclose(every.kept, 0.2 / 3)
    assert shares["early", "buy"].choices == 2
    assert shares["early", "scrap"].share == 1.0
    assert shares["late", "scrap"].share == 0.0
    assert shares["middle", "scrap"].choices == 0


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
    allowed = {_pool(name) for pair in probe.PAIRS for name in pair}
    for x in judged:
        for row in x:
            if row.tobytes() in real:
                continue
            # one card of a pair more in the ended player's pool than a real row
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
    games = probe.games(champion, seeds, threads=2, record=True)
    assert games.actions is not None
    assert len(games.actions) == len(seeds)
    assert np.array_equal(games.x, probe.positions(champion, seeds, threads=2))


def test_the_buys_of_real_games_are_where_replaying_them_puts_them():
    seeds = range(500, 503)
    played = probe.games("heuristic", seeds, record=True)
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
