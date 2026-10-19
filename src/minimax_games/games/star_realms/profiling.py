"""Profile the cheat's search: where a budgeted decision's time goes.

The workload is the cheat's real one: budgeted decisions (`cheat.decide`) at
the nodes/sec benchmark's positions (`bench.POSITIONS`) and, by default, in
the opening of a seeded cheat-vs-heuristic game (`game_decisions`). It runs
under the standard library's deterministic profiler, `cProfile`, which needs
no dependency and counts every call exactly, so the call counts are the same
on every run (the decisions are) and only the times vary. The profiler slows
each call by about the same amount, so the many small calls of a search run
about 3x slower under it: read its times relative to each other.

`breakdown` splits the search's time into areas (move generation, the rules,
`evaluate`, the transposition table and the search's own work) and `hashing`
totals the hash and equality methods, which every area calls. `report` puts
them together with `pstats`'s own tables.
"""

import cProfile
import io
import pstats
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from minimax_games.games.star_realms import bench, cheat, evaluate, node, rules
from minimax_games.games.star_realms.action import PlayCard
from minimax_games.games.star_realms.bots import HeuristicBot
from minimax_games.games.star_realms.card import Card, Faction
from minimax_games.games.star_realms.state import GameState, PlayerState
from minimax_games.games.star_realms.view import observe
from minimax_games.search import alphabeta

type _Function = tuple[str, int, str]
"""A function as `pstats` keys it: file, first line, name."""

_DICT_GET: _Function = ("~", 0, "<method 'get' of 'dict' objects>")
_HASH: _Function = ("~", 0, "<built-in method builtins.hash>")


@dataclass(frozen=True, slots=True)
class Profile:
    """A profiled workload: its decisions, how long they took with and
    without the profiler, and the profiler's statistics."""

    decisions: tuple[cheat.Decision, ...]
    seconds: float
    """The workload's time without the profiler."""
    profiled_seconds: float
    stats: pstats.Stats

    @property
    def nodes(self) -> int:
        return sum(decision.nodes for decision in self.decisions)


def game_decisions(
    seed: int, turns: int, budget: int, horizon: int = 2
) -> list[GameState]:
    """The states the cheat decides in, over the first `turns` turns of the
    seeded game it plays first against the heuristic, searching `horizon`
    turns deep within `budget` nodes per decision. Deterministic, like the
    game."""
    cheat_bot = cheat.CheatBot(turns=horizon, budget=budget)
    heuristic = HeuristicBot()
    states = []
    state = rules.auto_play(rules.new_game(seed))
    while state.turn <= turns and rules.winner(state) is None:
        actions = rules.legal_actions(state)
        if state.current == 0:
            states.append(state)
            action = cheat_bot.choose_from_state(state, actions)
        else:
            action = heuristic.choose(observe(state, state.current), actions)
        state = rules.advance(state, action)
    return states


def profile(states: Sequence[GameState], turns: int, budget: int) -> Profile:
    """Profile the cheat's decision in each of `states`, after timing them
    without the profiler (which slows the many small calls a search makes
    several times over, so its times are relative, not absolute)."""
    start = time.perf_counter()
    decisions = tuple(cheat.decide(state, turns, budget) for state in states)
    seconds = time.perf_counter() - start
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    profiled = tuple(cheat.decide(state, turns, budget) for state in states)
    profiler.disable()
    profiled_seconds = time.perf_counter() - start
    assert profiled == decisions  # decisions are deterministic
    return Profile(decisions, seconds, profiled_seconds, pstats.Stats(profiler))


@dataclass(frozen=True, slots=True)
class Area:
    """Part of the search's time: `seconds` in it, including what it calls
    (cumulative), over `calls` calls of its main function. `level` nests it:
    an area is part of the nearest area above it at a lower level."""

    level: int
    name: str
    seconds: float
    calls: int


def _key(function: Callable[..., object]) -> _Function:
    code = function.__code__  # ty: ignore[unresolved-attribute]
    return code.co_filename, code.co_firstlineno, code.co_name


def breakdown(stats: pstats.Stats) -> list[Area]:
    """The search's time by area, in nested order: the search, then its
    children's generation (move generation and `rules.advance_legal`), `evaluate`,
    the transposition table, and the rest, which is the search's own work
    (`alphabeta`, node comparisons, the budget)."""
    table: dict[_Function, tuple] = stats.stats  # ty: ignore[unresolved-attribute]

    def total(function: _Function) -> tuple[float, int]:
        _, calls, _, seconds, _ = table.get(function, (0, 0, 0.0, 0.0, {}))
        return seconds, calls

    def called_by(function: _Function, caller: _Function) -> tuple[float, int]:
        callers = table.get(function, (0, 0, 0.0, 0.0, {}))[4]
        _, calls, _, seconds = callers.get(caller, (0, 0, 0.0, 0.0))
        return seconds, calls

    def area(level: int, name: str, timed: tuple[float, int]) -> Area:
        return Area(level, name, *timed)

    legal = _key(rules.legal_actions)
    search = area(0, "search (cheat.decide)", total(_key(cheat.decide)))
    children = area(
        1,
        "children (StarRealmsNode.children)",
        total(_key(node.StarRealmsNode.children)),
    )
    search_fn = _key(alphabeta.alphabeta)
    tt_parts = [
        total(_key(node.transposition_key)),
        called_by(_DICT_GET, search_fn),
        total(_key(alphabeta._rebase)),
        called_by(_key(GameState.__hash__), search_fn),
    ]
    tt = area(
        1,
        "transposition table",
        (sum(s for s, _ in tt_parts), tt_parts[1][1]),
    )
    scores = area(1, "evaluate", total(_key(evaluate.evaluate)))
    game, player = total(_key(GameState.replace)), total(_key(PlayerState.replace))
    rest = search.seconds - children.seconds - scores.seconds - tt.seconds
    return [
        search,
        children,
        area(
            2,
            "move generation (legal_actions of each node)",
            called_by(legal, _key(node.StarRealmsNode.children)),
        ),
        area(2, "rules.advance_legal", total(_key(rules.advance_legal))),
        area(
            3,
            "automatic-play scan (rules._next_automatic)",
            total(_key(rules._next_automatic)),
        ),
        area(3, "applying actions (rules._apply)", total(_key(rules._apply))),
        area(
            4,
            "building states (GameState/PlayerState.replace)",
            (game[0] + player[0], game[1] + player[1]),
        ),
        scores,
        tt,
        Area(1, "search overhead (the rest)", rest, search.calls),
    ]


def _hashing_names() -> dict[_Function, str]:
    """The hash and equality methods the search calls, by name. Every
    superenums variant's generated method has the same code location, so
    `pstats` merges the effects' and actions' into one."""
    names = {_HASH: "builtins.hash (C, self time only)"}
    for cls, label in (
        (Card, "Card"),
        (PlayerState, "PlayerState"),
        (GameState, "GameState"),
        (Faction, "Enum"),
        (PlayCard, "Effect/Action (superenums variants)"),
    ):
        names[_key(cls.__hash__)] = f"{label}.__hash__"
        if cls is not Faction:
            names[_key(cls.__eq__)] = f"{label}.__eq__"
    return names


def hashing(stats: pstats.Stats) -> list[Area]:
    """The self time of every hash and equality method, wherever it is
    called (by `dict.fromkeys` in move generation, `in` checks in the rules,
    the transposition table...), then their total. Each hashes its fields, so
    much of a method's cumulative time is its fields' methods, listed on
    their own rows: self time doesn't count anything twice."""
    table: dict[_Function, tuple] = stats.stats  # ty: ignore[unresolved-attribute]
    names = _hashing_names()
    rows = []
    for function, (_, calls, self_seconds, _, _) in table.items():
        name = names.get(function)
        if name is None and function[2] in {"__hash__", "__eq__"}:
            name = f"{function[0]}:{function[1]}({function[2]})"
        if name is not None:
            rows.append(Area(1, name, self_seconds, calls))
    rows.sort(key=lambda row: -row.seconds)
    total = Area(
        0,
        "all hashing and equality",
        sum(row.seconds for row in rows),
        sum(row.calls for row in rows),
    )
    return [total, *rows]


def report(profiled: Profile, workload: str, top: int = 30) -> list[str]:
    """The profile as plain-text lines: `workload` (what was profiled), its
    speed with and without the profiler, the time by area (`breakdown`),
    hashing and equality (`hashing`), then the `top` functions by self time
    and by cumulative time, as `pstats` prints them."""
    nodes = profiled.nodes
    rate = nodes / profiled.seconds if profiled.seconds else 0.0
    overhead = profiled.profiled_seconds / profiled.seconds if profiled.seconds else 0.0
    areas = breakdown(profiled.stats)
    search = areas[0].seconds

    def rows(areas: list[Area]) -> list[str]:
        table = [("Area", "s", "% of search", "calls", "us/node")]
        for area in areas:
            table.append(
                (
                    "  " * area.level + area.name,
                    f"{area.seconds:.3f}",
                    f"{100 * area.seconds / search:.1f}" if search else "-",
                    str(area.calls),
                    f"{1e6 * area.seconds / nodes:.1f}" if nodes else "-",
                )
            )
        return bench.table(table)

    out = io.StringIO()
    stats = profiled.stats
    stats.stream = out  # ty: ignore[unresolved-attribute]
    stats.sort_stats("tottime").print_stats(top)
    stats.sort_stats("cumulative").print_stats(top)
    return [
        workload,
        f"Unprofiled: {nodes} nodes in {profiled.seconds:.2f}s ({rate:.0f} nodes/s)",
        f"Profiled: {profiled.profiled_seconds:.2f}s"
        f" ({overhead:.1f}x overhead: compare areas with each other)",
        "",
        "Time by area (cumulative: an area includes the areas indented under it;"
        " us/node is per node generated, under the profiler)",
        "",
        *rows(areas),
        "",
        "Hashing and equality, wherever called (self time: spread over the"
        " areas above)",
        "",
        *rows(hashing(stats)),
        "",
        *out.getvalue().splitlines(),
    ]
