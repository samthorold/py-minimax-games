"""The nodes/sec benchmark: the cheat's search on fixed seeded positions.

Each position is the start of a turn in a seeded greedy-vs-greedy game, so
the node counts are the same on every run; only the timings vary. A search
that passes its node budget is stopped there, so the benchmark always
finishes, and reports the nodes per second it measured up to then.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Self

from minimax_games.games.star_realms import cheat, rules
from minimax_games.games.star_realms.action import Action, EndTurn
from minimax_games.games.star_realms.bots import GreedyBot
from minimax_games.games.star_realms.node import StarRealmsNode
from minimax_games.games.star_realms.state import GameState
from minimax_games.games.star_realms.view import observe


def position(seed: int, turn: int) -> GameState:
    """The first decision of `turn` in the greedy-vs-greedy game from `seed`.

    Deterministic, so the same `seed` and `turn` always give the same state.
    """
    bot = GreedyBot()
    state = rules.auto_play(rules.new_game(seed))
    while state.turn < turn:
        if rules.winner(state) is not None:
            msg = f"seed {seed}'s greedy game ends before turn {turn}"
            raise ValueError(msg)
        actions = rules.legal_actions(state)
        state = rules.advance(state, bot.choose(observe(state, state.current), actions))
    return state


class _OverBudget(Exception):
    """Raised inside the search once it passes its node budget."""


class _Counter:
    """The nodes a search has reached and, if asked, its distinct states."""

    def __init__(self, max_nodes: int, count_states: bool) -> None:
        self.max_nodes = max_nodes
        self.nodes = 0
        self.states: set[GameState] | None = set() if count_states else None

    def visit(self, node: StarRealmsNode) -> None:
        if self.nodes >= self.max_nodes:
            raise _OverBudget
        self.nodes += 1
        if self.states is not None:
            self.states.add(node.state)


class _CountedNode(StarRealmsNode):
    """A node that counts itself on `counter` as the search reaches it."""

    counter: _Counter

    def children(self) -> Iterator[Self]:
        for child in super().children():
            child.counter = self.counter
            self.counter.visit(child)
            yield child


@dataclass(frozen=True, slots=True)
class Search:
    """One search: its node count is reproducible, its time is measured."""

    nodes: int
    """Nodes the search reached, the root included; at most the budget."""
    seconds: float
    complete: bool
    """False if the search passed its node budget and was stopped."""
    move: Action | None
    """The move the cheat plays, or None if the search was stopped."""
    states: int | None = None
    """Distinct states among the nodes, if counted. The cheat's transposition
    table expands each state once (per depth and move count), but a node is
    counted each time it is generated, before the table is consulted, so
    `nodes` is still above this."""

    @property
    def nodes_per_second(self) -> float:
        return self.nodes / self.seconds if self.seconds else 0.0


def search(
    state: GameState, turns: int, max_nodes: int, *, count_states: bool = False
) -> Search:
    """Run the cheat's `turns`-turn search on `state`, stopping it past
    `max_nodes` nodes.

    With `count_states`, also count the distinct states it reaches (hashing
    every state, which slows the search: time it without).
    """
    root = _CountedNode(state, state.current)
    counter = root.counter = _Counter(max_nodes, count_states)
    counter.visit(root)
    start = time.perf_counter()
    try:
        best = cheat.search(root, turns)
    except _OverBudget:
        complete, move = False, None
    else:
        complete, move = True, best.moves[0]
    seconds = time.perf_counter() - start
    states = None if counter.states is None else len(counter.states)
    return Search(counter.nodes, seconds, complete, move, states)


@dataclass(frozen=True, slots=True)
class TurnTree:
    """The whole of one turn's search tree, from its first decision."""

    nodes: int
    """Nodes in the tree: what a 1-turn search visits without pruning or a
    transposition table (a 1-turn search prunes nothing, since every node in
    it is the same player's)."""
    states: int
    """Distinct states among them: what a search that merged transpositions
    would visit."""


def turn_tree(state: GameState) -> TurnTree:
    """Count the one-turn tree from `state` exactly, without searching it.

    Each distinct state is expanded once and its subtree's size reused, so
    this is cheap even when the tree has millions of nodes.
    """
    sizes: dict[GameState, int] = {}

    def size(state: GameState) -> int:
        if state not in sizes:
            total = 1
            for action in rules.legal_actions(state):
                child = rules.advance(state, action)
                if isinstance(action, EndTurn) or rules.winner(child) is not None:
                    sizes.setdefault(child, 1)
                    total += 1
                else:
                    total += size(child)
            sizes[state] = total
        return sizes[state]

    return TurnTree(nodes=size(state), states=len(sizes))


POSITIONS: tuple[tuple[int, int], ...] = (
    (1, 3),
    (1, 7),
    (1, 11),
    (1, 15),
    (1, 19),
    (2, 5),
    (2, 13),
    (2, 21),
)
"""The default positions, as (seed, turn): early, mid and late game. Seed 1's
greedy game lasts 23 turns and seed 2's 27."""

HORIZONS = (1, 2)
"""The default horizons in turns: the cheat's default 2, and 1 to compare."""

MAX_NODES = 200_000
"""The default node budget per search."""


@dataclass(frozen=True, slots=True)
class Entry:
    """One search of the benchmark: `turns` deep from turn `turn` of `seed`."""

    seed: int
    turn: int
    turns: int
    search: Search


def run(
    positions: tuple[tuple[int, int], ...] = POSITIONS,
    horizons: tuple[int, ...] = HORIZONS,
    max_nodes: int = MAX_NODES,
    *,
    count_states: bool = False,
) -> list[Entry]:
    """Search each (seed, turn) position at each horizon, in that order."""
    entries = []
    for seed, turn in positions:
        state = position(seed, turn)
        for turns in horizons:
            result = search(state, turns, max_nodes, count_states=count_states)
            entries.append(Entry(seed, turn, turns, result))
    return entries


def report(entries: list[Entry]) -> list[str]:
    """The benchmark as plain-text lines: a row per search, then a summary per
    horizon. A stopped search's nodes read `>N`."""
    rows = [("Seed", "Turn", "Horizon", "Nodes", "States", "nodes/s", "ms")]
    for entry in entries:
        found = entry.search
        rows.append(
            (
                str(entry.seed),
                str(entry.turn),
                str(entry.turns),
                f"{found.nodes}" if found.complete else f">{found.nodes}",
                "-" if found.states is None else str(found.states),
                f"{found.nodes_per_second:.0f}",
                f"{1000 * found.seconds:.1f}",
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    table = [
        "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip()
        for row in rows
    ]
    summary = []
    for turns in sorted({entry.turns for entry in entries}):
        searches = [entry.search for entry in entries if entry.turns == turns]
        done = [found for found in searches if found.complete]
        nodes = sum(found.nodes for found in searches)
        seconds = sum(found.seconds for found in searches)
        rate = nodes / seconds if seconds else 0.0
        ms = (
            f"{1000 * sum(found.seconds for found in done) / len(done):.1f} ms"
            if done
            else "-"
        )
        summary.append(
            f"Horizon {turns}: {len(done)}/{len(searches)} searches complete,"
            f" {nodes} nodes in {seconds:.1f}s ({rate:.0f} nodes/s),"
            f" {ms} per completed decision"
        )
    return [
        "Rows: one search each; a stopped search's nodes read >N (the budget).",
        "",
        *table,
        "",
        *summary,
    ]
