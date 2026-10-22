"""The whole-turn planner: chooses moves by scoring ends of turn
(docs/star-realms.md, Search: the whole-turn planner).

A legitimate player: an ordinary `bots.Bot` that sees only its view. To
plan, it samples worlds from the view (`view.sample_world`) and, in each,
lists the distinct ends of the turn reachable from the current state and
scores each with an evaluator (`evaluate` by default; any `Evaluator` that
scores turn-boundary states plugs in). An end of turn is the state as
`EndTurn` leaves it, before the next player's automatic play, or the state
after a play that wins.

A candidate is a line of decisions up to the next chance event (a play that
draws or reveals a card, `reveals`) or the end of the turn. Nothing hidden is
revealed along it, so it is the same line in every world, and the part of
the turn before the first chance event is enumerated once, in the first
world, for all of them. A candidate is worth the mean over worlds of the best
end of turn reachable through it, and the bot plays the best candidate's line
to its end, planning again only after the chance event that ends it.

States are merged up to the order of every pile but the decks (`merge_key`,
equal exactly when `turn_sizes.unordered` is), ends of turn and states within
the turn alike. Nothing in the rules reads a pile's order but a reshuffle of
the discard pile, and `rules._shuffle` permutes positions by draws that
depend only on the seed and the pile's size, never on its cards, and leaves
the same next seed. Each sampled world has a fresh seed, so the deck a
reshuffle makes has the same distribution over the worlds whatever order the
discard pile was in: the player, who can't see the seed, can't choose a
shuffle by choosing an order, so the orders are one decision. Keeping them
apart would let a world pick the order that happens to shuffle best in it,
which no player can do. The value is order-invariant too (`evaluate`, and
`features` for learned ones).

Two backends list the ends: this module's Python enumeration, the
reference, which scores each end as it finds it, and the Rust one
(`minimax_games._native.star_realms.plan_ends`), which lists them all
first, the same ends in the same order, and returns their feature rows as
one matrix, which a learned value (`RowEvaluator`) scores in one numpy call.
The plans are the same with either.
"""

import logging
import math
import random
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, replace
from operator import attrgetter
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from minimax_games._native import star_realms as native
from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.action import Action, EndTurn
from minimax_games.games.star_realms.bots import greedy_move
from minimax_games.games.star_realms.card import Card
from minimax_games.games.star_realms.evaluate import LINEAR, Evaluator
from minimax_games.games.star_realms.state import GameState, PlayerState
from minimax_games.games.star_realms.view import PlayerView, View, observe, sample_world

logger = logging.getLogger(__name__)

BUDGET = 20_000
"""The default number of states per plan, shared by the worlds: it lists
every end of 88% of turns, for about 1 s on CPython
(docs/star-realms-results.md, Milestone 6: full-deck turn sizes)."""

WORLDS = 4
"""The default number of worlds per plan, as the sampler's."""

type Backend = Literal["python", "rust"]
"""Which enumeration a plan uses: the Python reference, or the Rust port."""


@runtime_checkable
class RowEvaluator(Evaluator, Protocol):
    """An evaluator that also scores ends of turn from their feature rows
    (`features.encode`), as the Rust backend gives them: a learned value
    (`value.ValueEvaluator`)."""

    def score_rows(
        self,
        x: np.ndarray,
        winner: np.ndarray,
        current: np.ndarray,
        perspective: int,
    ) -> np.ndarray:
        """Each row's score from `perspective`'s side, as `__call__` scores
        its state: `x` the rows, `winner` each state's winner (-1 for none)
        and `current` its player to move."""
        ...


def reveals(before: GameState, after: GameState) -> bool:
    """Whether a play from `before` to `after`, within the mover's turn, is a
    chance event: it draws a card from the part of their deck no one knows
    (cards known to be on top don't count), shuffles their discard pile into
    their deck, or reveals a card from the Trade Deck."""
    was, now = before.players[before.current], after.players[before.current]
    return (
        after.rng_seed != before.rng_seed
        or len(after.trade_deck) != len(before.trade_deck)
        or len(now.deck) - now.known_top != len(was.deck) - was.known_top
    )


_name = attrgetter("name")


def _pile(cards: tuple[Card, ...]) -> tuple[Card, ...] | tuple[str, ...]:
    """A pile up to order: its cards' names sorted (a short pile as it is)."""
    return tuple(sorted(map(_name, cards))) if len(cards) > 1 else cards


def _player_key(p: PlayerState) -> Hashable:
    return (
        p.authority,
        p.deck,
        _pile(p.hand),
        _pile(p.in_play),
        _pile(p.discard),
        _pile(p.bases),
        p.to_discard,
        p.known_top,
    )


def merge_key(s: GameState) -> Hashable:
    """A key equal for two states exactly when they are equal up to the order
    of every pile but the decks (`turn_sizes.unordered`, a test checks), a
    quarter of its cost: the planner keys every state it reaches with it.
    It lists every field, so a new field of `GameState` or `PlayerState`
    belongs here too (a test counts them)."""
    first, second = s.players
    return (
        _player_key(first),
        _player_key(second),
        s.rng_seed,
        s.trade_deck,
        _pile(s.trade_row),
        s.explorers,
        _pile(s.scrap_heap),
        s.current,
        s.turn,
        s.trade,
        s.combat,
        _pile(s.allies_used),
        _pile(s.bases_used),
        s.may_scrap,
        s.must_scrap,
        s.may_destroy,
        s.may_scrap_row,
        s.choosing,
        s.recycling,
        s.factions_played,
        s.may_on_top,
        s.free_ships,
        s.scrap_to_draw,
        s.copying,
    )


def _ordered(state: GameState) -> list[Action]:
    """The legal actions in `state`, the greedy policy's move first."""
    actions = rules.legal_actions(state)
    first = greedy_move(state.must_scrap, actions)
    return [first, *(action for action in actions if action != first)]


def _end(state: GameState, action: Action) -> GameState | None:
    """The end of turn `action` reaches from `state`, or None if the turn
    goes on: `EndTurn` without the next player's automatic play, or a play
    that wins."""
    if isinstance(action, EndTurn):
        return rules.apply_legal(state, action)
    child = rules.advance_legal(state, action)
    return child if rules.winner(child) is not None else None


class _OverBudget(Exception):
    """Raised when a plan's enumeration would pass its share of the budget."""


@dataclass(frozen=True, slots=True)
class Plan:
    """A plan: deterministic for the same worlds, budget and evaluator."""

    line: tuple[Action, ...]
    """The decisions to play, from the first world's current state, up to and
    including the next chance event or the end of the turn."""
    value: float | None
    """The line's value: the mean over worlds of the best end of turn found
    through it, from the mover's side (None if no end was found)."""
    states: int
    """Distinct states visited, the shared ones once: at most the budget."""
    complete: bool
    """True if every world listed every end of the turn."""


class _World:
    """One world's enumeration after the shared part: its replay of the
    shared states, the value of each state listed, and the states it may
    still visit."""

    def __init__(
        self, root: GameState, limit: int, evaluator: Evaluator, perspective: int
    ) -> None:
        self.states: dict[int, GameState] = {0: root}
        self.values: dict[Hashable, float] = {}
        self.limit = limit
        self.evaluator = evaluator
        self.perspective = perspective
        self.found: float | None = None
        """The best end found since the current candidate's search began."""

    def replay(self, index: int, shared: Sequence[tuple[int, Action]]) -> GameState:
        """Shared state `index` in this world, replayed along its line."""
        state = self.states.get(index)
        if state is None:
            parent, action = shared[index]
            state = rules.advance_legal(self.replay(parent, shared), action)
            self.states[index] = state
        return state

    def _seen(self, key: Hashable) -> float | None:
        value = self.values.get(key)
        if value is None and len(self.values) >= self.limit:
            raise _OverBudget
        return value

    def _note(self, value: float) -> float:
        if self.found is None or value > self.found:
            self.found = value
        return value

    def end(self, state: GameState) -> float:
        """The score of `state`, an end of turn."""
        key = merge_key(state)
        if (value := self._seen(key)) is None:
            value = self.values[key] = self.evaluator(state, self.perspective)
        return self._note(value)

    def best(self, state: GameState) -> float:
        """The best end of the turn reachable from `state`, depth first, the
        greedy policy's moves first. A state already listed costs nothing."""
        key = merge_key(state)
        if (value := self._seen(key)) is not None:
            return self._note(value)
        # Claimed before its children, so the budget counts it; no play leads
        # back to a state of the same turn, so nothing reads it before it's set.
        self.values[key] = -math.inf
        value = -math.inf
        for action in _ordered(state):
            end = _end(state, action)
            if end is not None:
                value = max(value, self.end(end))
            else:
                value = max(value, self.best(rules.advance_legal(state, action)))
        self.values[key] = value
        return value


def _choose(
    first: GameState,
    values: list[list[float]],
    line: Callable[[int], tuple[Action, ...]],
    visited: int,
    complete: bool,
) -> Plan:
    """The plan whose candidate `c` is worth `values[c]` in the worlds that
    reached it and has the line `line(c)`."""
    reached = max(map(len, values), default=0)
    if reached == 0:
        return Plan((_ordered(first)[0],), None, visited, complete=False)
    best = max(
        (c for c in range(len(values)) if len(values[c]) == reached),
        key=lambda c: sum(values[c]) / reached,
    )
    return Plan(line(best), sum(values[best]) / reached, visited, complete)


def plan(
    worlds: Sequence[GameState],
    budget: int = BUDGET,
    evaluator: Evaluator = LINEAR,
    backend: Backend = "python",
) -> Plan:
    """The best line from `worlds`, sampled worlds of one view (or one true
    state), at most `budget` states visited in all, scoring ends with
    `evaluator` from the mover's side, the ends listed by `backend`'s
    enumeration (the Rust one needs a `RowEvaluator`).

    The shared part, the states before the first chance event, is listed
    first, in the first world, with at most half the budget: each line that
    leaves it (by a chance event or the end of the turn) is a candidate. The
    rest of the budget is split evenly among the worlds, and each lists the
    ends through each candidate in turn, in the order found. Where a world's
    share runs out, the candidate it was on is worth the best end found so
    far and the later ones aren't reached. The candidates reached in the most
    worlds compete by their mean, ties to the first found; if none is
    reached, the line is the greedy policy's move.
    """
    if not worlds:
        msg = "a plan needs at least 1 world"
        raise ValueError(msg)
    _check(backend, evaluator)
    if backend == "rust":
        assert isinstance(evaluator, RowEvaluator)
        return _plan_rust(worlds, budget, evaluator)
    first = worlds[0]
    me = first.current
    shared: list[tuple[int, Action]] = [(-1, EndTurn())]  # the root's entry is unused
    states = [first]
    seen = {merge_key(first)}
    candidates: list[tuple[int, Action]] = []
    limit = budget // 2
    complete = True

    def walk(index: int) -> None:
        state = states[index]
        for action in _ordered(state):
            if isinstance(action, EndTurn):
                candidates.append((index, action))
                continue
            child = rules.advance_legal(state, action)
            if rules.winner(child) is not None or reveals(state, child):
                candidates.append((index, action))
                continue
            key = merge_key(child)
            if key in seen:
                continue
            if len(seen) >= limit:
                raise _OverBudget
            seen.add(key)
            shared.append((index, action))
            states.append(child)
            walk(len(states) - 1)

    if limit < 1:
        complete = False
    else:
        try:
            walk(0)
        except _OverBudget:
            complete = False

    share = (budget - len(seen)) // len(worlds)
    values: list[list[float]] = [[] for _ in candidates]
    visited = len(seen)
    for i, world in enumerate(worlds):
        search = _World(world, share, evaluator, me)
        if i == 0:
            search.states = dict(enumerate(states))
        for value, (index, action) in zip(values, candidates, strict=True):
            search.found = None
            try:
                state = search.replay(index, shared)
                end = _end(state, action)
                if end is not None:
                    value.append(search.end(end))
                else:
                    value.append(search.best(rules.advance_legal(state, action)))
            except _OverBudget:
                complete = False
                if search.found is not None:
                    value.append(search.found)
                break
        visited += len(search.values)

    def line(candidate: int) -> tuple[Action, ...]:
        index, action = candidates[candidate]
        line = [action]
        while index > 0:
            index, action = shared[index]
            line.append(action)
        return tuple(reversed(line))

    return _choose(first, values, line, visited, complete)


def _check(backend: Backend, evaluator: Evaluator) -> None:
    if backend not in ("python", "rust"):
        msg = f"no planner backend {backend!r}: 'python' or 'rust'"
        raise ValueError(msg)
    if backend == "rust" and not isinstance(evaluator, RowEvaluator):
        msg = (
            "the Rust backend scores ends by their feature rows, which"
            f" {evaluator.describe()} doesn't (it needs a learned value)"
        )
        raise ValueError(msg)


def _names(cards: tuple[Card, ...]) -> list[str]:
    return [card.name for card in cards]


def _native_player(p: PlayerState) -> dict[str, Any]:
    return {
        "authority": p.authority,
        "deck": _names(p.deck),
        "hand": _names(p.hand),
        "in_play": _names(p.in_play),
        "discard": _names(p.discard),
        "bases": _names(p.bases),
        "to_discard": p.to_discard,
        "known_top": p.known_top,
    }


def native_state(s: GameState) -> dict[str, Any]:
    """`s` as the Rust engine takes it: a dict of its fields, cards by name."""
    return {
        "players": [_native_player(p) for p in s.players],
        "rng_seed": s.rng_seed,
        "trade_deck": _names(s.trade_deck),
        "trade_row": _names(s.trade_row),
        "explorers": s.explorers,
        "scrap_heap": _names(s.scrap_heap),
        "current": s.current,
        "turn": s.turn,
        "trade": s.trade,
        "combat": s.combat,
        "allies_used": _names(s.allies_used),
        "bases_used": _names(s.bases_used),
        "may_scrap": s.may_scrap,
        "must_scrap": s.must_scrap,
        "may_destroy": s.may_destroy,
        "may_scrap_row": s.may_scrap_row,
        "choosing": _names(s.choosing),
        "recycling": s.recycling,
        "factions_played": list(s.factions_played),
        "may_on_top": s.may_on_top,
        "free_ships": s.free_ships,
        "scrap_to_draw": s.scrap_to_draw,
        "copying": s.copying,
    }


def _plan_rust(
    worlds: Sequence[GameState], budget: int, evaluator: RowEvaluator
) -> Plan:
    """`plan` with the Rust enumeration: the ends' rows scored in one call,
    each candidate worth, in each world that reached it, the best score
    among the rows the enumeration gives it there."""
    first = worlds[0]
    found = native.plan_ends([native_state(w) for w in worlds], budget)
    rows, offsets = found["rows"], found["offsets"]
    best: list[float] = []
    if len(rows):
        assert np.all(np.diff(offsets) > 0), "a candidate reached no end"
        scores = evaluator.score_rows(
            found["features"], found["winner"], found["current"], first.current
        )
        best = np.maximum.reduceat(scores[rows], offsets[:-1]).tolist()
    lines: list[list[int]] = found["lines"]
    values: list[list[float]] = [[] for _ in lines]
    scored = iter(best)
    for reached in found["reached"]:
        for candidate in range(reached):
            values[candidate].append(next(scored))

    def line(candidate: int) -> tuple[Action, ...]:
        state, actions = first, []
        for i in lines[candidate]:
            actions.append(rules.legal_actions(state)[i])
            state = rules.advance_legal(state, actions[-1])
        return tuple(actions)

    return _choose(first, values, line, found["states"], found["complete"])


def _sorted(cards: tuple[Card, ...]) -> tuple[Card, ...]:
    return tuple(sorted(cards, key=_name))


def _player_seen(player: PlayerView) -> PlayerView:
    return replace(
        player,
        in_play=_sorted(player.in_play),
        discard=_sorted(player.discard),
        bases=_sorted(player.bases),
    )


def seen(view: View) -> View:
    """`view` up to the order of its piles: what a line of plays that reveals
    nothing reaches in every world, though automatic play can order the
    cards in play differently (it plays them in hand order, which views
    don't keep)."""
    first, second = view.players
    return replace(
        view,
        players=(_player_seen(first), _player_seen(second)),
        trade_row=_sorted(view.trade_row),
        scrap_heap=_sorted(view.scrap_heap),
        allies_used=_sorted(view.allies_used),
        bases_used=_sorted(view.bases_used),
    )


class PlannerBot:
    """Plans whole turns over `worlds` worlds sampled from its view.

    Each plan visits at most `budget` states (`plan`), scoring ends of turn
    with `evaluator` (by default the linear `evaluate`), and the bot plays its
    line to the next chance event or the end of the turn before planning
    again. The worlds are drawn from an rng seeded with `seed`, so a game
    replays identically. `plans` and `turns` count the plans it has made and
    the turns it has played. `backend` picks the enumeration (`plan`): the
    Rust one needs a learned value, and plays the same game.

    The line is followed while the game goes as planned: before each of its
    moves the bot checks that its view is, up to pile order (`seen`), the one
    the line reached in the first world, and plans again if not
    (`diverged` counts those plans). The order of a hand, which views don't
    keep, decides which of two cards automatic play plays first, and with it
    whether a card still waits as a decision: one played before a ship that
    could scrap it from hand is played automatically, one after it waits
    (`rules._hand_scrap_ahead`), and a choice made at once stops automatic
    play until it is made. So a world's line can meet a decision the real
    hand order doesn't offer, or miss one it does.
    """

    def __init__(
        self,
        seed: int,
        worlds: int = WORLDS,
        budget: int = BUDGET,
        evaluator: Evaluator = LINEAR,
        backend: Backend = "python",
    ) -> None:
        _check(backend, evaluator)
        if worlds < 1:
            msg = f"the planner needs at least 1 world, not {worlds}"
            raise ValueError(msg)
        if budget < 2 * worlds:
            msg = (
                f"the budget must be at least 2 states per world:"
                f" {budget} states for {worlds} worlds"
            )
            raise ValueError(msg)
        self.worlds = worlds
        self.budget = budget
        self.evaluator = evaluator
        self.backend: Backend = backend
        self.plans = 0
        self.turns = 0
        self.diverged = 0
        self._rng = random.Random(seed)
        self._turn: int | None = None
        self._line: list[tuple[View, Action]] = []
        """The rest of the line: each move, and the view (`seen`) expected
        before it."""

    def settings(self) -> str:
        return (
            f"{self.worlds} worlds, budget {self.budget} states per plan"
            f" shared by the worlds, {self.evaluator.describe()}"
            + (", Rust enumeration" if self.backend == "rust" else "")
        )

    def choose(self, view: View, actions: tuple[Action, ...]) -> Action:
        if view.turn != self._turn:
            self._turn = view.turn
            self.turns += 1
            self._line = []
        if self._line and self._line[0][0] != seen(view):
            self.diverged += 1
            self._line = []
        if not self._line:
            state, made = self._plan(view)
            self.plans += 1
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("planned %s", made)
            for action in made.line:
                self._line.append((seen(observe(state, view.observer)), action))
                state = rules.advance_legal(state, action)
        _, action = self._line.pop(0)
        assert action in actions, (action, actions)
        return action

    def _plan(self, view: View) -> tuple[GameState, Plan]:
        """A plan from `view`, and the world its line was planned in."""
        worlds = [sample_world(view, self._rng) for _ in range(self.worlds)]
        return worlds[0], plan(worlds, self.budget, self.evaluator, self.backend)
