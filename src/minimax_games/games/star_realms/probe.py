"""Probes of a judgement (docs/star-realms.md, How its understanding is
judged): real positions changed in one fact, and how the judged chance of
winning responds.

A position is its row of features (`features.encode`), which is all a
judgement sees, so a probe changes the row. A changed row is still a valid
position: a card added to a player's deck is a copy taken from those out of
sight (`available`), so every card is still in one place.

The synergy probe (docs/star-realms/strategy.md, Synergy) adds cards of one
function to the deck of the player who just ended, one at a time, and
reports what each adds to the judged chance: if cards of a function are
worth more together, the second and third add more than the first.

The investment probe (docs/star-realms/strategy.md, The race and the
investment) weighs trade against combat: in each position it adds, apart,
each card of a pair of comparable cards (`PAIRS`: ships of one cost, one
giving only trade, the other only combat) and reports what each adds per
unit of what it gives, and which of the two the judgement would rather have.
Beside it, `buys` reads what players bought from recorded games, each buy
an economy or a combat one by its card's `kind`.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from minimax_games._native import star_realms as native
from minimax_games.games.star_realms import features
from minimax_games.games.star_realms.card import Card, CardType
from minimax_games.games.star_realms.catalogue import CATALOGUE
from minimax_games.games.star_realms.effect import (
    Choice,
    Combat,
    Effect,
    ShipsGainCombat,
    Trade,
)
from minimax_games.games.star_realms.selfplay import (
    Planner,
    SelfPlay,
    alternating,
    play,
)

type Judgement = Callable[[np.ndarray], np.ndarray]
"""A judgement of rows of positions (`features.NAMES`): the chance the
player who just ended goes on to win from each (`value.Model.predict`)."""

_SIDES = ("ended", "next")
_COLUMN = {name: i for i, name in enumerate(features.NAMES)}


def _columns(name: str) -> list[int]:
    """The columns of a row that count a copy of `name` someone holds or the
    Trade Row shows."""
    places = [f"{side}: pool {name}" for side in _SIDES]
    if name in features.BASES:
        places += [f"{side}: base {name}" for side in _SIDES]
    places.append(f"row {name}")
    return [_COLUMN[place] for place in places]


_SEEN = {
    name: _columns(name) for name, card in CATALOGUE.items() if card.faction is not None
}
"""Each Trade Deck card's columns that count the copies in sight."""


def available(row: np.ndarray, name: str) -> int:
    """How many copies of Trade Deck card `name` neither player holds nor
    the Trade Row shows: those in the Trade Deck or the scrap heap, which the
    row can't tell apart, since both are out of sight."""
    if name not in _SEEN:
        msg = f"{name} isn't a Trade Deck card"
        raise ValueError(msg)
    return CATALOGUE[name].copies - int(row[_SEEN[name]].sum())


def add_card(row: np.ndarray, name: str) -> np.ndarray:
    """`row` with one more copy of Trade Deck card `name` in the draw pool
    of the player who just ended their turn, taken from those out of sight
    (`available`)."""
    if available(row, name) < 1:
        msg = f"no copy of {name} is left to add"
        raise ValueError(msg)
    changed = np.array(row, dtype=np.float64, copy=True)
    changed[_COLUMN[f"ended: pool {name}"]] += 1
    return changed


def _functions() -> dict[str, tuple[str, ...]]:
    tables = features.extra_tables(more=True)["more"]
    trade_deck = [name for name in features.CARDS if CATALOGUE[name].faction]
    index = {name: i for i, name in enumerate(features.CARDS)}

    def doing(*kinds: str) -> tuple[str, ...]:
        return tuple(
            name for name in trade_deck if any(tables[k][index[name]] for k in kinds)
        )

    every = set(doing("draw", "scrap", "discard"))
    return {
        "draw": doing("draw"),
        "scrap": doing("scrap"),
        "discard": doing("discard"),
        "none": tuple(name for name in trade_deck if name not in every),
    }


FUNCTIONS: dict[str, tuple[str, ...]] = _functions()
"""The Trade Deck cards that draw, that scrap and that make the opponent
discard, in any of their abilities (as `features.MORE` counts them; a card
can be in more than one), and as a control, those that do none of these."""

DEPTH = 3
"""How many cards of a function the synergy probe adds: the first, second
and third."""


@dataclass(frozen=True, slots=True, eq=False)
class Synergy:
    """What each card of a function added to the judged chance, per
    position."""

    gains: dict[str, np.ndarray]
    """Per function, shape `(positions, DEPTH)`: the mean over the samples
    of what the first, second and third card added; NaN where no sample
    could add that many (no copies left)."""
    logit_gains: dict[str, np.ndarray]
    """The same on the logit scale, `log(p / (1 - p))`, where a judgement
    whose score is linear before a sigmoid gains equally from every card:
    in chances, a card adds less where the chance is already near 0 or 1,
    which reads as no synergy where the score has none."""
    turn: np.ndarray
    """Each position's turn (`features.NAMES`' "turn"), for its phase."""


_EPSILON = 1e-12
"""How near 0 or 1 a chance is clipped before taking its logit."""


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPSILON, 1 - _EPSILON)
    return np.log(p / (1 - p))


def _mean_over_samples(found: np.ndarray) -> np.ndarray:
    """The mean over axis 1 of the samples that were made (not NaN), and
    NaN where none was: a sample adds all `DEPTH` cards or none, so the
    first, second and third card are measured on the same positions."""
    made = ~np.isnan(found)
    count = made.sum(axis=1)
    total = np.where(made, found, 0).sum(axis=1)
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def _add_drawn(
    x: np.ndarray, names: tuple[str, ...], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Rows `x`, each with one card of `names` added as `add_card` adds it,
    drawn uniformly from those with a copy left (`available`, for every row
    at once), and whether each row had one left (a row with none is
    returned unchanged)."""
    seen = np.zeros((len(names), x.shape[1]))
    for i, name in enumerate(names):
        seen[i, _SEEN[name]] = 1
    copies = np.array([CATALOGUE[name].copies for name in names])
    pool = np.array([_COLUMN[f"ended: pool {name}"] for name in names])
    keys = rng.random((len(x), len(names)))
    keys[copies - x @ seen.T < 1] = -1
    made = keys.max(axis=1) >= 0
    added = x.copy()
    added[np.flatnonzero(made), pool[keys.argmax(axis=1)[made]]] += 1
    return added, made


def synergy(
    judgement: Judgement, rows: np.ndarray, seed: int = 0, samples: int = 4
) -> Synergy:
    """For each function in `FUNCTIONS` and each position in `rows`, add
    `DEPTH` cards of the function to the deck of the player who just ended,
    `samples` times with cards drawn at random (an rng seeded with `seed`),
    and judge each step. Plays no games."""
    rng = np.random.default_rng(seed)
    x = np.repeat(np.asarray(rows, dtype=np.float64), samples, axis=0)
    gains, logit_gains = {}, {}
    for function, names in FUNCTIONS.items():
        steps = [x]
        made = np.ones(len(x), dtype=bool)
        for _ in range(DEPTH):
            added, left = _add_drawn(steps[-1], names, rng)
            steps.append(added)
            made &= left
        # judged together, each sample's steps one after another
        batch = np.stack(steps, axis=1)[made].reshape(-1, x.shape[1])
        found = np.full((len(x), DEPTH), np.nan)
        logits = found.copy()
        if len(batch):
            chances = np.asarray(judgement(batch)).reshape(-1, DEPTH + 1)
            found[made] = np.diff(chances, axis=1)
            logits[made] = np.diff(_logit(chances), axis=1)
        shape = (len(rows), samples, DEPTH)
        gains[function] = _mean_over_samples(found.reshape(shape))
        logit_gains[function] = _mean_over_samples(logits.reshape(shape))
    turn = np.asarray(rows)[:, _COLUMN["turn"]]
    return Synergy(gains=gains, logit_gains=logit_gains, turn=turn)


PHASES: dict[str, tuple[int, int]] = {
    "early": (1, 8),
    "middle": (9, 16),
    "late": (17, 10**9),
}
"""Game phases by turn, first and last inclusive (a turn is one player's;
in the champion's games about a quarter of positions are by turn 7, half by
13 and three quarters by 19)."""


@dataclass(frozen=True, slots=True)
class Line:
    """One function's gains over one phase's positions."""

    phase: str
    function: str
    positions: int
    """The positions some sample could add every card to."""
    mean: np.ndarray
    """The first, second and third card's mean gain over the positions."""
    sd: np.ndarray
    """Their standard deviations across the positions."""
    step: np.ndarray
    """The mean of what each card added beyond the card before it (second
    less first, third less second), per position: above 0 is synergy."""
    step_se: np.ndarray
    """The standard errors of `step`'s means."""
    more: float
    """The share of positions where the second card added more than the
    first."""


def summary(found: Synergy, logit: bool = False) -> list[Line]:
    """Each function's gains, over all positions and then by phase, in
    chances or with `logit` on the logit scale."""
    gains = found.logit_gains if logit else found.gains
    lines = []
    for phase, at in _phases(found.turn):
        for function, g in gains.items():
            g = g[at & ~np.isnan(g[:, 0])]
            n = len(g)
            steps = np.diff(g, axis=1)
            lines.append(
                Line(
                    phase=phase,
                    function=function,
                    positions=n,
                    mean=g.mean(axis=0) if n else np.full(DEPTH, np.nan),
                    sd=g.std(axis=0) if n else np.full(DEPTH, np.nan),
                    step=steps.mean(axis=0) if n else np.full(DEPTH - 1, np.nan),
                    step_se=(
                        steps.std(axis=0, ddof=1) / np.sqrt(n)
                        if n > 1
                        else np.full(DEPTH - 1, np.nan)
                    ),
                    more=float((steps[:, 0] > 0).mean()) if n else float("nan"),
                )
            )
    return lines


type Output = Literal["trade", "combat"]


def _primary(card: Card) -> Iterable[Effect]:
    """`card`'s primary effects, and each option's of a choice."""
    for e in card.primary:
        if isinstance(e, Choice):
            for option in e.options:
                yield from option
        else:
            yield e


def output(card: Card, kind: Output) -> int:
    """The most trade or combat `card`'s primary ability gives when played
    (the larger option of a choice)."""
    effect = Trade if kind == "trade" else Combat
    unconditional = sum(e.amount for e in card.primary if isinstance(e, effect))
    options = [
        sum(e.amount for e in option if isinstance(e, effect))
        for e in card.primary
        if isinstance(e, Choice)
        for option in e.options
    ]
    return unconditional + max(options, default=0)


def _pure(card: Card) -> Output | None:
    """ "trade" or "combat" if `card` is a ship whose primary ability gives
    that and nothing else, else None."""
    if card.type is not CardType.SHIP or card.faction is None:
        return None
    if all(isinstance(e, Trade) for e in card.primary):
        return "trade"
    if all(isinstance(e, Combat) for e in card.primary):
        return "combat"
    return None


def _pairs() -> tuple[tuple[str, str], ...]:
    pure = {name: _pure(card) for name, card in CATALOGUE.items()}
    return tuple(
        (economy, combat)
        for economy, kind in pure.items()
        if kind == "trade"
        for combat, other in pure.items()
        if other == "combat" and CATALOGUE[combat].cost == CATALOGUE[economy].cost
    )


PAIRS: tuple[tuple[str, str], ...] = _pairs()
"""The comparable cards of the investment probe: Trade Deck ships of the
same cost whose primary ability gives only trade (the first) or only combat
(the second), so a pair differs in little but what it gives."""


@dataclass(frozen=True, slots=True, eq=False)
class Gains:
    """What one card of each pair added to the judgement, per position, on
    one scale: NaN where either card of the pair had no copy left."""

    trade: dict[str, np.ndarray]
    """Per economy card, what it added per unit of its trade."""
    combat: dict[str, np.ndarray]
    """Per combat card, what it added per unit of its combat."""
    card_gap: dict[tuple[str, str], np.ndarray]
    """Per pair, what the economy card added less what the combat card
    added: which of two cards of the same cost the judgement would rather
    the player had bought."""


@dataclass(frozen=True, slots=True, eq=False)
class Investment:
    """The investment probe's gains, in judged chance and in logit (as
    `Synergy`'s), per position."""

    pairs: tuple[tuple[str, str], ...]
    chance: Gains
    logit: Gains
    turn: np.ndarray
    """Each position's turn, for its phase."""


def _left(x: np.ndarray, name: str) -> np.ndarray:
    """`available` for every row of `x` at once."""
    return CATALOGUE[name].copies - x[:, _SEEN[name]].sum(axis=1)


def investment(
    judgement: Judgement, rows: np.ndarray, pairs: tuple[tuple[str, str], ...] = PAIRS
) -> Investment:
    """For each pair of comparable cards and each position in `rows`, add
    the economy card, and separately the combat card, to the deck of the
    player who just ended, and judge both. Plays no games."""
    x = np.asarray(rows, dtype=np.float64)
    scales: dict[str, Gains] = {
        scale: Gains(trade={}, combat={}, card_gap={}) for scale in ("chance", "logit")
    }
    for economy, other in pairs:
        made = (_left(x, economy) >= 1) & (_left(x, other) >= 1)
        added = []
        for name in (economy, other):
            changed = x.copy()
            changed[:, _COLUMN[f"ended: pool {name}"]] += 1
            added.append(changed)
        chances = np.asarray(judgement(np.concatenate([x, *added]))).reshape(3, -1)
        for scale, judged in (("chance", chances), ("logit", _logit(chances))):
            gains = np.where(made, judged[1:] - judged[0], np.nan)
            found = scales[scale]
            found.trade[economy] = gains[0] / output(CATALOGUE[economy], "trade")
            found.combat[other] = gains[1] / output(CATALOGUE[other], "combat")
            found.card_gap[economy, other] = gains[0] - gains[1]
    return Investment(
        pairs=pairs,
        chance=scales["chance"],
        logit=scales["logit"],
        turn=x[:, _COLUMN["turn"]],
    )


@dataclass(frozen=True, slots=True)
class PairLine:
    """One pair's investment gains over one phase's positions."""

    phase: str
    pair: tuple[str, str]
    positions: int
    """The positions both cards of the pair could be added to."""
    trade: float
    """The economy card's mean gain per unit of trade."""
    trade_sd: float
    """Its standard deviation across the positions."""
    combat: float
    """The combat card's mean gain per unit of combat."""
    combat_sd: float
    unit_gap: float
    """The mean of a unit of trade's gain less a unit of combat's, per
    position: above 0, the judgement weighs trade above combat."""
    unit_gap_se: float
    """The standard error of `unit_gap`."""
    card_gap: float
    """The mean of the economy card's gain less the combat card's."""
    card_gap_se: float
    economy_more: float
    """The share of positions where the economy card added more than the
    combat card of the same cost."""


def _phases(turn: np.ndarray) -> Iterable[tuple[str, np.ndarray]]:
    """Each phase of `PHASES`, after all positions ("all"), with which of
    the positions of `turn` are in it."""
    for phase, (first, last) in {"all": (0, 10**9), **PHASES}.items():
        yield phase, (turn >= first) & (turn <= last)


def _mean_se(values: np.ndarray) -> tuple[float, float]:
    n = len(values)
    mean = float(values.mean()) if n else float("nan")
    se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return mean, se


def investment_summary(found: Investment, logit: bool = False) -> list[PairLine]:
    """Each pair's gains, over all positions and then by phase, in chances
    or with `logit` on the logit scale."""
    gains = found.logit if logit else found.chance
    lines = []
    for phase, at in _phases(found.turn):
        for economy, combat in found.pairs:
            card_gap = gains.card_gap[economy, combat]
            keep = at & ~np.isnan(card_gap)
            trade, hits = gains.trade[economy][keep], gains.combat[combat][keep]
            n = int(keep.sum())
            unit_gap, unit_gap_se = _mean_se(trade - hits)
            gap, gap_se = _mean_se(card_gap[keep])
            lines.append(
                PairLine(
                    phase=phase,
                    pair=(economy, combat),
                    positions=n,
                    trade=float(trade.mean()) if n else float("nan"),
                    trade_sd=float(trade.std()) if n else float("nan"),
                    combat=float(hits.mean()) if n else float("nan"),
                    combat_sd=float(hits.std()) if n else float("nan"),
                    unit_gap=unit_gap,
                    unit_gap_se=unit_gap_se,
                    card_gap=gap,
                    card_gap_se=gap_se,
                    economy_more=(
                        float((card_gap[keep] > 0).mean()) if n else float("nan")
                    ),
                )
            )
    return lines


type Kind = Literal["economy", "combat", "mixed", "neither"]

BUYABLE: tuple[str, ...] = tuple(
    name for name, card in CATALOGUE.items() if card.faction or name == "Explorer"
)
"""The cards a player can buy: the Trade Deck's and the Explorer."""


def kind(card: Card) -> Kind:
    """What buying `card` invests in, by what its primary ability gives
    every time it is played (through either option of a choice): "economy"
    if trade and no combat, "combat" if combat (or combat for every ship,
    Fleet HQ's) and no trade, "mixed" if either (Patrol Mech's choice), and
    "neither" if neither (cards that only draw, scrap, copy or defend).
    Ally and scrap abilities don't count: they give only sometimes."""
    effects = list(_primary(card))
    trade = any(isinstance(e, Trade) for e in effects)
    combat = any(isinstance(e, Combat | ShipsGainCombat) for e in effects)
    if trade and combat:
        return "mixed"
    if trade:
        return "economy"
    return "combat" if combat else "neither"


_BUYS = ("Acquire", "AcquireOnTop")
"""The decisions that buy a card (`Acquire` and `AcquireOnTop`); a free
ship (`AcquireFreeShip`) costs no trade, so it isn't one."""


@dataclass(frozen=True, slots=True, eq=False)
class Buys:
    """Every card bought in some games, in order."""

    game: np.ndarray
    """Each buy's game, as its index among the games."""
    turn: np.ndarray
    """The turn it was bought in, as the game numbers them: 1 for the first
    player's first, both players' counted."""
    seat: np.ndarray
    """The seat that bought it (0 moves first)."""
    card: tuple[str, ...]
    kind: tuple[Kind, ...]
    """Each card's `kind`."""


def buys(games: Iterable[Sequence[native.NativeAction]]) -> Buys:
    """The buys in each game's decisions (`selfplay.play(..., record=True)`'s
    `actions`)."""
    game, turn, card = [], [], []
    for i, actions in enumerate(games):
        now = 1
        for action in actions:
            if action[0] == "EndTurn":
                now += 1
            elif action[0] in _BUYS:
                game.append(i)
                turn.append(now)
                card.append(action[1])
    turns = np.array(turn, dtype=np.int64)
    return Buys(
        game=np.array(game, dtype=np.int64),
        turn=turns,
        seat=(turns - 1) % 2,
        card=tuple(card),
        kind=tuple(kind(CATALOGUE[name]) for name in card),
    )


BANDS: tuple[tuple[int, int], ...] = (
    *((first, first + 3) for first in range(1, 25, 4)),
    (25, 10**9),
)
"""Bands of turns for buy shares, first and last inclusive: four turns
(two of each player's) at a time, to 24, then the rest."""

KINDS: tuple[Kind, ...] = ("economy", "combat", "mixed", "neither")


@dataclass(frozen=True, slots=True)
class ShareLine:
    """The buys over one band of turns."""

    band: tuple[int, int]
    buys: int
    share: dict[Kind, float]
    """Each kind's share of the buys (NaN if there were none)."""
    explorer: float
    """The Explorers' share (part of "economy"): the one card always for
    sale, so a buy that says least about the Trade Row."""


def buy_shares(
    found: Buys, bands: tuple[tuple[int, int], ...] = BANDS
) -> list[ShareLine]:
    """Each kind's share of the buys in `found`, over every turn the bands
    cover and then by band."""
    kinds = np.array(found.kind, dtype=object)
    explorers = np.array([card == "Explorer" for card in found.card], dtype=bool)
    lines = []
    whole = (bands[0][0], bands[-1][1])
    for first, last in (whole, *bands):
        at = (found.turn >= first) & (found.turn <= last)
        n = int(at.sum())
        share = {
            k: float((kinds[at] == k).mean()) if n else float("nan") for k in KINDS
        }
        explorer = float(explorers[at].mean()) if n else float("nan")
        lines.append(
            ShareLine(band=(first, last), buys=n, share=share, explorer=explorer)
        )
    return lines


BASE_SEED = 12_000_000
"""The first seed of the games probe positions come from: clear of every
policy-iteration range (`iterate.Settings`, below 4,000,000 for up to 100
rounds) and of every earlier tournament, fit and offline match (all below
10,000,000), so no position is from a game a judgement was trained on."""


def positions(champion: Planner, seeds: Iterable[int], threads: int = 1) -> np.ndarray:
    """Real positions: every turn's end in the champion's games against
    itself from `seeds`, played in Rust (`selfplay.play`)."""
    return play(alternating(champion, champion, seeds), threads=threads).x


def played(champion: Planner, seeds: Iterable[int], threads: int = 1) -> SelfPlay:
    """The champion's games against itself from `seeds`, as `positions`
    plays them, with every decision recorded (which changes no game)."""
    return play(alternating(champion, champion, seeds), threads=threads, record=True)


def investment_report(found: Investment, logit: bool = False) -> list[str]:
    """`investment_summary` as a table: per phase and pair, the positions,
    the economy card's gain per unit of trade and the combat card's per
    unit of combat (mean, sd across positions), a unit of trade less a unit
    of combat and the economy card less the combat card (mean, se), and the
    share of positions where the economy card added more."""
    scale = "logit" if logit else "chance"
    lines = [
        f"Judged gain ({scale}) per unit of trade and of combat (mean, sd across"
        " positions); trade less combat per unit, and economy card less combat"
        " card (mean, se):",
        f"{'phase':<7} {'economy / combat card':<34} {'positions':>9}"
        f" {'trade':>17} {'combat':>17} {'unit gap':>18} {'card gap':>18}"
        f" {'econ>comb':>9}",
    ]
    for line in investment_summary(found, logit=logit):
        pair = " / ".join(line.pair)
        lines.append(
            f"{line.phase:<7} {pair:<34} {line.positions:>9}"
            f" {line.trade:>+8.4f} ({line.trade_sd:.4f})"
            f" {line.combat:>+8.4f} ({line.combat_sd:.4f})"
            f" {line.unit_gap:>+9.4f} ({line.unit_gap_se:.4f})"
            f" {line.card_gap:>+9.4f} ({line.card_gap_se:.4f})"
            f" {line.economy_more:>9.1%}"
        )
    return lines


def _band(band: tuple[int, int]) -> str:
    first, last = band
    return f"{first}+" if last >= 10**9 else f"{first}-{last}"


def buys_report(players: dict[str, Buys]) -> list[str]:
    """Each player's `buy_shares`, side by side by band of turns: the buys,
    and the economy, combat, mixed and neither shares of them."""
    shares = {name: buy_shares(found) for name, found in players.items()}
    header = f"{'turns':<7}"
    for name in players:
        header += f" | {name:<9} {'buys':>6}" + "".join(f" {k:>8}" for k in KINDS)
        header += f" {'explorer':>8}"
    lines = [
        "Buys by turn (both players' turns counted; explorer is part of economy):",
        header,
    ]
    for i, band in enumerate([(0, 0), *BANDS]):
        line = f"{'all' if i == 0 else _band(band):<7}"
        for name in players:
            s = shares[name][i]
            line += f" | {'':<9} {s.buys:>6}" + "".join(
                f" {s.share[k]:>8.1%}" for k in KINDS
            )
            line += f" {s.explorer:>8.1%}"
        lines.append(line)
    return lines


def report(found: Synergy, logit: bool = False) -> list[str]:
    """`summary` as a table: per phase and function, the positions, each
    card's mean gain and standard deviation across positions, what the
    second and third added beyond the card before (with standard errors),
    and the share of positions where the second added more than the first."""
    scale = "logit" if logit else "chance"
    lines = [
        f"Judged gain ({scale}) of the 1st, 2nd and 3rd card of each function"
        " (mean, sd across positions); 2nd-1st and 3rd-2nd (mean, se):",
        f"{'phase':<7} {'function':<8} {'positions':>9}"
        f" {'1st':>16} {'2nd':>16} {'3rd':>16}"
        f" {'2nd-1st':>17} {'3rd-2nd':>17} {'2nd>1st':>8}",
    ]
    for line in summary(found, logit=logit):
        cards = " ".join(
            f"{m:>+7.4f} ({s:.4f})" for m, s in zip(line.mean, line.sd, strict=True)
        )
        steps = " ".join(
            f"{m:>+8.4f} ({s:.4f})"
            for m, s in zip(line.step, line.step_se, strict=True)
        )
        lines.append(
            f"{line.phase:<7} {line.function:<8} {line.positions:>9}"
            f" {cards} {steps} {line.more:>8.1%}"
        )
    return lines
