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

The faction-pair probe (the same section) adds one card of each faction to
the same deck, and reports, for decks mostly of each faction (`mostly`),
what a card of each faction adds: a 4-by-4 table, with each pair's
interaction, what it adds beyond its deck's and its card's, to show whether
the judgement singles out particular pairs.

The denial probe (strategy.md, Denial) asks whether a card the next player
would value, in the Trade Row for them to buy, makes the position worse for
the player who just ended. Which card they would value is the judgement's
own answer: the one whose copy in the next player's draw pool costs the
player who just ended most, against the one that costs least. Swapping each
into the Trade Row in place of the same card (`denial`) keeps the row at
its five cards and every card in one place. In games, a card a player buys
or scraps from the Trade Row is denial when the judgement finds it worth
less in their own deck than kept from the opponent (`worth`).
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from minimax_games.games.star_realms import features
from minimax_games.games.star_realms.card import Faction
from minimax_games.games.star_realms.catalogue import CATALOGUE
from minimax_games.games.star_realms.selfplay import (
    Planner,
    Player,
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

TRADE_DECK: tuple[str, ...] = tuple(_SEEN)
"""The Trade Deck's distinct cards, in catalogue order: those a Trade Row
can show."""


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


def _out_of_sight(x: np.ndarray, names: tuple[str, ...] = TRADE_DECK) -> np.ndarray:
    """Per row and card of `names`, how many copies are out of sight
    (`available`, for every row at once)."""
    seen = np.zeros((len(names), x.shape[1]))
    for i, name in enumerate(names):
        seen[i, _SEEN[name]] = 1
    copies = np.array([CATALOGUE[name].copies for name in names])
    return copies - x @ seen.T


def _add_drawn(
    x: np.ndarray, names: tuple[str, ...], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Rows `x`, each with one card of `names` added as `add_card` adds it,
    drawn uniformly from those with a copy left (`available`, for every row
    at once), and whether each row had one left (a row with none is
    returned unchanged)."""
    pool = np.array([_COLUMN[f"ended: pool {name}"] for name in names])
    keys = rng.random((len(x), len(names)))
    keys[_out_of_sight(x, names) < 1] = -1
    made = keys.max(axis=1) >= 0
    added = x.copy()
    added[np.flatnonzero(made), pool[keys.argmax(axis=1)[made]]] += 1
    return added, made


def _added_one_by_one(
    judgement: Judgement,
    rows: np.ndarray,
    names: tuple[str, ...],
    depth: int,
    samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """What each of `depth` cards of `names`, added one at a time to the
    deck of the player who just ended (each drawn at random from those with
    a copy left, `samples` times per position), added to the judged chance,
    and the same on the logit scale: each of shape `(positions, depth)`,
    the mean over the samples, NaN where no sample could add every card."""
    x = np.repeat(np.asarray(rows, dtype=np.float64), samples, axis=0)
    steps = [x]
    made = np.ones(len(x), dtype=bool)
    for _ in range(depth):
        added, left = _add_drawn(steps[-1], names, rng)
        steps.append(added)
        made &= left
    # judged together, each sample's steps one after another
    batch = np.stack(steps, axis=1)[made].reshape(-1, x.shape[1])
    found = np.full((len(x), depth), np.nan)
    logits = found.copy()
    if len(batch):
        chances = np.asarray(judgement(batch)).reshape(-1, depth + 1)
        found[made] = np.diff(chances, axis=1)
        logits[made] = np.diff(_logit(chances), axis=1)
    shape = (len(rows), samples, depth)
    return (
        _mean_over_samples(found.reshape(shape)),
        _mean_over_samples(logits.reshape(shape)),
    )


def synergy(
    judgement: Judgement, rows: np.ndarray, seed: int = 0, samples: int = 4
) -> Synergy:
    """For each function in `FUNCTIONS` and each position in `rows`, add
    `DEPTH` cards of the function to the deck of the player who just ended,
    `samples` times with cards drawn at random (an rng seeded with `seed`),
    and judge each step. Plays no games."""
    rng = np.random.default_rng(seed)
    gains, logit_gains = {}, {}
    for function, names in FUNCTIONS.items():
        gains[function], logit_gains[function] = _added_one_by_one(
            judgement, rows, names, DEPTH, samples, rng
        )
    turn = np.asarray(rows)[:, _COLUMN["turn"]]
    return Synergy(gains=gains, logit_gains=logit_gains, turn=turn)


FACTIONS: dict[str, tuple[str, ...]] = {
    faction.value: tuple(
        name for name, card in CATALOGUE.items() if card.faction is faction
    )
    for faction in Faction
}
"""Each faction's Trade Deck cards, by the faction's name."""

LEAST = 4
"""How many faction cards a deck needs before it can be mostly one faction:
with fewer, one purchase decides the majority (half of the champion's
positions have 6 or more, a quarter 3 or fewer)."""


def _owned(faction: str) -> list[int]:
    """The columns counting the ended player's copies of `faction`'s cards:
    in their draw pool and, for bases, in play."""
    return [
        i
        for name in FACTIONS[faction]
        for i in (
            _COLUMN[f"ended: pool {name}"],
            *([_COLUMN[f"ended: base {name}"]] if name in features.BASES else []),
        )
    ]


def mostly(rows: np.ndarray) -> np.ndarray:
    """Per row, the index in `FACTIONS` of the faction the deck of the
    player who just ended is mostly of: one that holds a strict majority of
    its faction cards (pool and bases in play; Scouts, Vipers and Explorers
    have none), when it has at least `LEAST` of them; -1 for any other."""
    x = np.asarray(rows)
    counts = np.stack([x[:, _owned(f)].sum(axis=1) for f in FACTIONS], axis=1)
    total = counts.sum(axis=1)
    major = 2 * counts.max(axis=1) > total
    return np.where(major & (total >= LEAST), counts.argmax(axis=1), -1)


@dataclass(frozen=True, slots=True, eq=False)
class FactionPairs:
    """What one card of each faction added to the judged chance, per
    position."""

    gains: dict[str, np.ndarray]
    """Per faction, shape `(positions,)`: the mean over the samples of what
    one of its cards added; NaN where none was left to add."""
    logit_gains: dict[str, np.ndarray]
    """The same on the logit scale (`Synergy.logit_gains`)."""
    deck: np.ndarray
    """Each position's faction the ended player's deck is mostly of
    (`mostly`)."""
    turn: np.ndarray
    """Each position's turn."""


def faction_pairs(
    judgement: Judgement, rows: np.ndarray, seed: int = 0, samples: int = 4
) -> FactionPairs:
    """For each faction in `FACTIONS` and each position in `rows`, add one
    of its cards to the deck of the player who just ended, `samples` times
    drawn at random (an rng seeded with `seed`), and judge it. Plays no
    games."""
    rng = np.random.default_rng(seed)
    gains, logit_gains = {}, {}
    for faction, names in FACTIONS.items():
        found, logits = _added_one_by_one(judgement, rows, names, 1, samples, rng)
        gains[faction], logit_gains[faction] = found[:, 0], logits[:, 0]
    turn = np.asarray(rows)[:, _COLUMN["turn"]]
    return FactionPairs(
        gains=gains, logit_gains=logit_gains, deck=mostly(rows), turn=turn
    )


def _owned_by_next(judgement: Judgement, x: np.ndarray) -> np.ndarray:
    """Per row and `TRADE_DECK` card, the judged chance of the player who
    just ended with one more copy of the card in the next player's draw
    pool, taken from those out of sight; NaN where none is left."""
    left = _out_of_sight(x) >= 1
    chances = np.full(left.shape, np.nan)
    for j, name in enumerate(TRADE_DECK):
        at = np.flatnonzero(left[:, j])
        if len(at):
            changed = x[at].copy()
            changed[:, _COLUMN[f"next: pool {name}"]] += 1
            chances[at, j] = judgement(changed)
    return chances


@dataclass(frozen=True, slots=True, eq=False)
class Denial:
    """What a Trade Row card the next player would value does to the judged
    chance of the player who just ended, per position."""

    effect: np.ndarray
    """The judged chance with the card the next player values most in the
    Trade Row less that with the card they value least, in place of the same
    row card, averaged over the row's distinct cards; below 0 is denial
    seen. NaN where fewer than two cards are out of sight."""
    logit_effect: np.ndarray
    """The same on the logit scale."""
    valued: np.ndarray
    """How much more the next player owning the most valued card costs the
    player who just ended than owning the least valued one: the scale
    `effect` is read against (`-effect / valued` is the share of it the row
    shows)."""
    logit_valued: np.ndarray
    """The same on the logit scale."""
    most: np.ndarray
    """The most valued card, as its index in `TRADE_DECK` (-1 where NaN)."""
    turn: np.ndarray
    """Each position's turn, for its phase."""


def denial(judgement: Judgement, rows: np.ndarray) -> Denial:
    """For each position in `rows`, which Trade Deck card the next player
    would value most and least, by the judged cost to the player who just
    ended of the next player owning it (a copy out of sight added to their
    draw pool), and the judged chance with each of the two in the Trade Row
    in place of the same row card (the replaced card goes out of sight).
    Plays no games."""
    x = np.asarray(rows, dtype=np.float64)
    owned = _owned_by_next(judgement, x)
    choices = (~np.isnan(owned)).sum(axis=1)
    ok = np.flatnonzero(choices >= 2)
    most = np.full(len(x), -1)
    least = np.full(len(x), -1)
    # the most valued is the one whose owning lowers the chance most
    most[ok] = np.nanargmin(owned[ok], axis=1)
    least[ok] = np.nanargmax(owned[ok], axis=1)
    effect = np.full(len(x), np.nan)
    logit_effect = effect.copy()
    valued = effect.copy()
    logit_valued = effect.copy()
    lo, hi = owned[ok, least[ok]], owned[ok, most[ok]]
    valued[ok] = lo - hi
    logit_valued[ok] = _logit(lo) - _logit(hi)
    row = np.array([_COLUMN[f"row {name}"] for name in TRADE_DECK])
    shown = x[:, row] > 0
    totals = np.zeros((2, len(x)))
    slots = np.zeros(len(x))
    for j in range(len(TRADE_DECK)):
        at = ok[shown[ok, j]]
        if not len(at):
            continue
        swapped = []
        for card in (most[at], least[at]):
            changed = x[at].copy()
            changed[:, row[j]] -= 1
            changed[np.arange(len(at)), row[card]] += 1
            swapped.append(np.asarray(judgement(changed), dtype=np.float64))
        totals[0, at] += swapped[0] - swapped[1]
        totals[1, at] += _logit(swapped[0]) - _logit(swapped[1])
        slots[at] += 1
    filled = slots > 0
    effect[filled] = totals[0, filled] / slots[filled]
    logit_effect[filled] = totals[1, filled] / slots[filled]
    return Denial(
        effect=effect,
        logit_effect=logit_effect,
        valued=valued,
        logit_valued=logit_valued,
        most=most,
        turn=x[:, _COLUMN["turn"]],
    )


_TAKES = frozenset({"Acquire", "AcquireOnTop", "AcquireFreeShip"})
_INDEX = {name: i for i, name in enumerate(TRADE_DECK)}


@dataclass(frozen=True, slots=True, eq=False)
class Choices:
    """The Trade Row cards players took or scrapped in recorded games, each
    with the position the turn it was chosen in ended in."""

    row: np.ndarray
    """Each choice's row in the games' rows (`SelfPlay.x`)."""
    card: np.ndarray
    """The card, as its index in `TRADE_DECK`."""
    scrap: np.ndarray
    """Whether it was scrapped from the Trade Row (else acquired: bought,
    or taken free by Blob Carrier's ally ability)."""


def choices(played: SelfPlay) -> Choices:
    """Every Trade Row card acquired or scrapped in `played`'s finished
    games (played with `record`), except in a game's last turn, which ends
    without a position. Explorers aren't Trade Row cards, so they aren't
    counted."""
    if played.actions is None:
        msg = "the games weren't recorded"
        raise ValueError(msg)
    found: list[tuple[int, int, bool]] = []
    for g, actions in enumerate(played.actions):
        if played.winners[g] is None:
            continue
        rows = np.flatnonzero(played.game == g)
        turn = 0
        for action in actions:
            kind = action[0]
            scrap = kind == "ScrapTradeRowCard"
            chose = (kind in _TAKES or scrap) and action[1] in _INDEX
            if chose and turn < len(rows):
                found.append((int(rows[turn]), _INDEX[action[1]], scrap))
            turn += kind == "EndTurn"
    row, card, scrap = zip(*found, strict=True) if found else ((), (), ())
    return Choices(
        row=np.array(row, dtype=int),
        card=np.array(card, dtype=int),
        scrap=np.array(scrap, dtype=bool),
    )


@dataclass(frozen=True, slots=True, eq=False)
class Worth:
    """Each choice's judged worth to the player who chose it and to their
    opponent (`worth`)."""

    own: np.ndarray
    """The judged chance it adds to its chooser owning it."""
    kept: np.ndarray
    """The judged chance its chooser keeps by the opponent not owning it."""
    denial: np.ndarray
    """Whether `own` is less than `kept`: the card was worth more kept from
    the opponent than in the chooser's own deck."""
    turn: np.ndarray
    """The turn it was chosen in, for its phase."""


def worth(judgement: Judgement, rows: np.ndarray, chosen: Choices) -> Worth:
    """Judge each of `chosen` in the position its turn ended in, `rows[row]`,
    without the card: for a card acquired, that position with the copy taken
    out of the chooser's draw pool and put out of sight; for a card scrapped,
    the position itself, its copy already in the scrap heap. `own` is the
    chance the chooser's owning the copy adds to that position, and `kept`
    what the next player's owning it would take away. NaN (and not denial)
    where the copy isn't there to move (a card acquired and scrapped in the
    same turn)."""
    x = np.asarray(rows, dtype=np.float64)[chosen.row]
    names = [TRADE_DECK[c] for c in chosen.card]
    mine = np.array([_COLUMN[f"ended: pool {name}"] for name in names], dtype=int)
    theirs = np.array([_COLUMN[f"next: pool {name}"] for name in names], dtype=int)
    at = np.arange(len(x))
    without = x.copy()
    without[at[~chosen.scrap], mine[~chosen.scrap]] -= 1
    left = _out_of_sight(without)[at, chosen.card] >= 1
    ok = left & (without[at, mine] >= 0)
    owned, given = without.copy(), without.copy()
    owned[at, mine] += 1
    given[at, theirs] += 1
    own, kept = np.full(len(x), np.nan), np.full(len(x), np.nan)
    if ok.any():
        base = np.asarray(judgement(without[ok]), dtype=np.float64)
        own[ok] = np.asarray(judgement(owned[ok])) - base
        kept[ok] = base - np.asarray(judgement(given[ok]))
    return Worth(
        own=own,
        kept=kept,
        denial=ok & (own < kept),
        turn=x[:, _COLUMN["turn"]],
    )


@dataclass(frozen=True, slots=True)
class ShareLine:
    """How many of one phase's buys or Trade Row scraps were denial."""

    phase: str
    kind: str
    """"buy" (any acquisition from the Trade Row) or "scrap"."""
    choices: int
    """The choices judged (`Worth.own` not NaN)."""
    share: float
    """The share of them that were denial (`Worth.denial`)."""
    own: float
    """Their mean `Worth.own`."""
    kept: float
    """Their mean `Worth.kept`."""


def denial_shares(found: Worth, chosen: Choices) -> list[ShareLine]:
    """The share of buys and of Trade Row scraps that were denial, over all
    turns and then by phase."""
    lines = []
    judged = ~np.isnan(found.own)
    for phase, at in _phases(found.turn):
        for kind, scrap in (("buy", False), ("scrap", True)):
            these = at & judged & (chosen.scrap == scrap)
            n = int(these.sum())
            nan = float("nan")
            lines.append(
                ShareLine(
                    phase=phase,
                    kind=kind,
                    choices=n,
                    share=float(found.denial[these].mean()) if n else nan,
                    own=float(found.own[these].mean()) if n else nan,
                    kept=float(found.kept[these].mean()) if n else nan,
                )
            )
    return lines


@dataclass(frozen=True, slots=True)
class Pair:
    """What a card of one faction adds to decks mostly of another (or the
    same) faction."""

    deck: str
    card: str
    positions: int
    """The positions whose deck is mostly `deck` and a card of `card` was
    left to add."""
    mean: float
    sd: float
    """The standard deviation across the positions."""
    se: float
    """The standard error of the mean."""
    interaction: float
    """What the pair adds beyond its deck's and its card's: `mean` less the
    mean of its row (what any card adds to such decks) and of its column
    (what such a card adds to any deck), plus the table's mean. Above 0,
    the judgement singles the pair out; a judgement that adds to a card's
    worth regardless of the deck has none."""
    interaction_se: float
    """Its standard error. Per row of decks the interaction is a mean over
    that row's positions of the card's gain less the four cards' mean gain,
    and rows hold disjoint positions, so their variances add (over the
    positions where every faction had a card left to add; positions of the
    same game counted as independent)."""


def _se(g: np.ndarray) -> float:
    return float(g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else float("nan")


def pairs(found: FactionPairs, logit: bool = False) -> list[Pair]:
    """For every ordered pair of factions, the gain from one card of the
    second in decks mostly of the first, in chances or with `logit` on the
    logit scale: the 4-by-4 table, row by row."""
    gains = np.stack(list((found.logit_gains if logit else found.gains).values()), 1)
    k = len(FACTIONS)
    rows = [found.deck == i for i in range(k)]
    cells = [[g[at & ~np.isnan(g)] for g in gains.T] for at in rows]
    means = np.array([[g.mean() if len(g) else np.nan for g in r] for r in cells])
    interaction = (
        means
        - means.mean(axis=1, keepdims=True)
        - means.mean(axis=0, keepdims=True)
        + means.mean()
    )
    every = gains[~np.isnan(gains).any(axis=1)]
    every_deck = found.deck[~np.isnan(gains).any(axis=1)]
    centred = every - every.mean(axis=1, keepdims=True)
    # per row of decks and card, the variance of the row's mean of `centred`
    row_var = np.array(
        [
            centred[every_deck == i].var(axis=0, ddof=1) / (every_deck == i).sum()
            if (every_deck == i).sum() > 1
            else np.full(k, np.nan)
            for i in range(k)
        ]
    )
    weights = (np.eye(k) - 1 / k) ** 2  # [deck row i, row k]
    interaction_se = np.sqrt(weights @ row_var)
    names = list(FACTIONS)
    return [
        Pair(
            deck=names[i],
            card=names[j],
            positions=len(cells[i][j]),
            mean=float(means[i, j]),
            sd=float(cells[i][j].std()) if len(cells[i][j]) else float("nan"),
            se=_se(cells[i][j]),
            interaction=float(interaction[i, j]),
            interaction_se=float(interaction_se[i, j]),
        )
        for i in range(k)
        for j in range(k)
    ]


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


def _phases(turn: np.ndarray) -> Iterable[tuple[str, np.ndarray]]:
    """Each phase's name and which positions are in it, all of them first."""
    for phase, (first, last) in {"all": (0, 10**9), **PHASES}.items():
        yield phase, (turn >= first) & (turn <= last)


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


@dataclass(frozen=True, slots=True)
class DenialLine:
    """The denial probe's effect over one phase's positions."""

    phase: str
    positions: int
    """The positions with two or more cards out of sight to compare."""
    mean: float
    """The mean effect of the most valued card in the Trade Row against the
    least valued (`Denial.effect`)."""
    sd: float
    """Its standard deviation across positions."""
    se: float
    """The standard error of the mean."""
    valued: float
    """The mean of how much more the next player owning the most valued
    card costs than owning the least valued (`Denial.valued`)."""
    seen: float
    """`-mean / valued`: the share of what the card would be worth to the
    next player that the judgement sees in its being in the Trade Row for
    them (0 if it sees none, 1 if as much as if they owned it already)."""
    below: float
    """The share of positions where the effect is below 0."""


def denial_summary(found: Denial, logit: bool = False) -> list[DenialLine]:
    """The denial probe's effect over all positions and then by phase, in
    chances or with `logit` on the logit scale."""
    effects = found.logit_effect if logit else found.effect
    values = found.logit_valued if logit else found.valued
    lines = []
    for phase, at in _phases(found.turn):
        at = at & ~np.isnan(effects)
        e, v = effects[at], values[at]
        n = len(e)
        mean = float(e.mean()) if n else float("nan")
        valued = float(v.mean()) if n else float("nan")
        lines.append(
            DenialLine(
                phase=phase,
                positions=n,
                mean=mean,
                sd=float(e.std()) if n else float("nan"),
                se=_se(e),
                valued=valued,
                seen=-mean / valued if n else float("nan"),
                below=float((e < 0).mean()) if n else float("nan"),
            )
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
    return games(champion, seeds, threads=threads).x


def games(
    player: Player, seeds: Iterable[int], threads: int = 1, record: bool = False
) -> SelfPlay:
    """`player`'s games against itself from `seeds`, played in Rust
    (`selfplay.play`), with every decision kept if `record`."""
    return play(alternating(player, player, seeds), threads=threads, record=record)


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


def pairs_report(found: FactionPairs, logit: bool = False) -> list[str]:
    """`pairs` as two 4-by-4 tables, decks by row and added cards by column:
    each pair's mean gain, its standard deviation across positions and its
    standard error, then each pair's interaction."""
    scale = "logit" if logit else "chance"
    table = pairs(found, logit=logit)
    width = max(len(f) for f in FACTIONS) + 2
    head = f"{'deck mostly of':<{width + 8}}" + "".join(
        f"{card:>26}" for card in FACTIONS
    )
    lines = [
        f"Judged gain ({scale}) of one card of each faction (columns) in decks"
        " mostly of each (rows): mean (sd across positions) ±se, positions:",
        head,
    ]
    for deck in FACTIONS:
        row = [cell for cell in table if cell.deck == deck]
        n = min(cell.positions for cell in row)
        lines.append(
            f"{deck:<{width}}{n:>8}"
            + "".join(f" {c.mean:>+8.4f} ({c.sd:.4f}) ±{c.se:.4f}" for c in row)
        )
    lines += [
        f"Each pair's interaction ({scale}) ±se: its gain less its row's and"
        " column's means, plus the table's; above 0, the pair is singled out:",
        head,
    ]
    for deck in FACTIONS:
        row = [cell for cell in table if cell.deck == deck]
        lines.append(
            f"{deck:<{width + 8}}"
            + "".join(
                f"{f'{c.interaction:+.4f} ±{c.interaction_se:.4f}':>26}" for c in row
            )
        )
    return lines


def denial_report(found: Denial, logit: bool = False) -> list[str]:
    """`denial_summary` as a table: per phase, the positions, the effect of
    the most valued card in the Trade Row against the least (mean, sd across
    positions, se), how much more the most valued is worth to the next
    player owning it, the share of that the row shows, and the share of
    positions where the effect is below 0."""
    scale = "logit" if logit else "chance"
    lines = [
        f"Judged effect ({scale}) on the player who just ended of the card the"
        " next player values most in the Trade Row, against the one they value"
        " least:",
        f"{'phase':<7} {'positions':>9} {'effect':>9} {'sd':>7} {'se':>7}"
        f" {'valued':>8} {'seen':>7} {'below 0':>8}",
    ]
    for line in denial_summary(found, logit=logit):
        lines.append(
            f"{line.phase:<7} {line.positions:>9} {line.mean:>+9.4f}"
            f" {line.sd:>7.4f} {line.se:>7.4f} {line.valued:>8.4f}"
            f" {line.seen:>7.1%} {line.below:>8.1%}"
        )
    return lines


def deck_counts(found: FactionPairs) -> str:
    """How many positions' decks are mostly of each faction, and of none."""
    counts = ", ".join(
        f"{faction} {int((found.deck == i).sum())}"
        for i, faction in enumerate(FACTIONS)
    )
    return (
        f"Decks mostly of one faction (at least {LEAST} faction cards, a strict"
        f" majority): {counts}; mostly of no one faction"
        f" {int((found.deck == -1).sum())}"
    )


def share_report(players: dict[str, list[ShareLine]]) -> list[str]:
    """`denial_shares` of each player in `players` as one table: per phase
    and kind of choice, the choices, the share that were denial and their
    mean judged worth to the chooser and kept from the opponent."""
    lines = [
        "Buys and Trade Row scraps worth less in the chooser's deck than kept"
        " from the opponent (denial), by the judgement:",
        f"{'player':<10} {'phase':<7} {'kind':<5} {'choices':>7} {'denial':>7}"
        f" {'own':>8} {'kept':>8}",
    ]
    for player, shares in players.items():
        for line in shares:
            lines.append(
                f"{player:<10} {line.phase:<7} {line.kind:<5} {line.choices:>7}"
                f" {line.share:>7.1%} {line.own:>+8.4f} {line.kept:>+8.4f}"
            )
    return lines
