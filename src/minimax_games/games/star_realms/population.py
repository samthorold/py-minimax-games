"""The reference population (docs/star-realms.md, How progress is measured):
a player against every member, on fixed seeds with seats alternating, all in
Rust at full speed (`selfplay.play`).

A player is named: `heuristic`, `greedy`, a committed model (`value.load`,
played by the whole-turn planner with it), or the path of a saved model
(`.npz`, named by its file's stem). Every opponent plays the same seeds,
`0` up, the player moving first on even ones, as the ladder seated them, so
a player's games against the heuristic on seeds 0-499 are the ladder's.
"""

import csv
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from minimax_games.games.star_realms import selfplay, value
from minimax_games.games.star_realms.fit import MAX_TURNS
from minimax_games.games.star_realms.planner import BUDGET, WORLDS
from minimax_games.games.star_realms.sprt import Z, wilson

POPULATION = ("heuristic", "mlp", "pi03", "pf01")
"""The heuristic and every champion, oldest first."""

GAMES = 500
"""Games against each opponent, on seeds 0 up."""

BOTS = ("heuristic", "greedy")
"""The players that are scripted bots, not planners with a model."""


@dataclass(frozen=True, slots=True)
class Game:
    """One game by its players' names, as the per-seed CSVs record it."""

    seed: int
    first: str
    second: str
    winner: str | None
    """`None` if the game passed the turn cap."""
    turns: int


@dataclass(frozen=True, slots=True)
class Match:
    """A player's games against one opponent."""

    player: str
    opponent: str
    games: tuple[Game, ...]

    @property
    def wins(self) -> int:
        return sum(game.winner == self.player for game in self.games)

    @property
    def finished(self) -> int:
        """Games with a winner: the denominator of the win rate."""
        return sum(game.winner is not None for game in self.games)

    @property
    def unfinished(self) -> tuple[int, ...]:
        """The seeds of the games that passed the turn cap."""
        return tuple(game.seed for game in self.games if game.winner is None)

    @property
    def rate(self) -> float:
        """The player's share of the finished games (0 if none finished)."""
        return self.wins / self.finished if self.finished else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        """The rate's 95% Wilson interval."""
        return wilson(self.wins, self.finished)

    def as_first(self) -> tuple[int, int]:
        """The player's wins and finished games moving first."""
        seated = [
            game
            for game in self.games
            if game.first == self.player and game.winner is not None
        ]
        return sum(game.winner == self.player for game in seated), len(seated)


def mean(matches: Sequence[Match]) -> float:
    """The mean of the matches' win rates: each opponent counts the same."""
    return sum(match.rate for match in matches) / len(matches)


def mean_interval(matches: Sequence[Match]) -> tuple[float, float]:
    """An approximate 95% interval of `mean`: the normal one, from the
    variance of each rate (p(1 - p) / n) summed and divided by the number of
    opponents squared, clipped to [0, 1]."""
    variance = (
        sum(m.rate * (1 - m.rate) / m.finished for m in matches if m.finished)
        / len(matches) ** 2
    )
    spread = Z * math.sqrt(variance)
    centre = mean(matches)
    return max(0.0, centre - spread), min(1.0, centre + spread)


def name(player: str) -> str:
    """The name a player's games record: a saved model's file stem."""
    return Path(player).stem if player.endswith(".npz") else player


def _player(player: str, worlds: int, budget: int) -> selfplay.Player:
    if player == "heuristic" or player == "greedy":  # BOTS, spelt out for ty
        return player
    model = (
        value.Model.load(Path(player))
        if player.endswith(".npz")
        else value.load(player)
    )
    return selfplay.Planner(model, worlds, budget)


def play(
    player: str,
    *,
    opponents: Sequence[str] = POPULATION,
    games: int = GAMES,
    threads: int = 1,
    worlds: int = WORLDS,
    budget: int = BUDGET,
    max_turns: int = MAX_TURNS,
) -> tuple[Match, ...]:
    """`player` against each of `opponents` but itself, `games` games each
    on seeds 0 up, `player` first on even seeds; all the games in one batch
    on `threads` threads."""
    me = name(player)
    rivals = [o for o in opponents if name(o) != me]
    players = {p: _player(p, worlds, budget) for p in (player, *rivals)}
    named = [
        (seed, player, rival) if seed % 2 == 0 else (seed, rival, player)
        for rival in rivals
        for seed in range(games)
    ]
    played = selfplay.play(
        [(seed, players[a], players[b]) for seed, a, b in named], threads, max_turns
    )
    found = [
        Game(
            seed,
            name(a),
            name(b),
            None if winner is None else name((a, b)[winner]),
            turns,
        )
        for (seed, a, b), winner, turns in zip(
            named, played.winners, played.turns, strict=True
        )
    ]
    return tuple(
        Match(me, name(rival), tuple(found[i * games : (i + 1) * games]))
        for i, rival in enumerate(rivals)
    )


HEADER = ("seed", "first", "second", "winner", "turns")
"""The per-seed CSVs' columns, as `iterate` writes its gating and ladder
matches: the winner by name, empty if the game was unfinished."""


def write(matches: Iterable[Match], out: Path) -> None:
    """Each match's games to `out/<opponent>.csv`, whole or not at all."""
    out.mkdir(parents=True, exist_ok=True)
    for match in matches:
        path = out / f"{match.opponent}.csv"
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            for g in match.games:
                writer.writerow([g.seed, g.first, g.second, g.winner or "", g.turns])
        os.replace(temporary, path)


def read(path: Path) -> tuple[Game, ...]:
    """The games of a per-seed CSV: this module's, or a gating or ladder
    match's."""
    with path.open(newline="") as f:
        return tuple(
            Game(
                int(row["seed"]),
                row["first"],
                row["second"],
                row["winner"] or None,
                int(row["turns"]),
            )
            for row in csv.DictReader(f)
        )


@dataclass(frozen=True, slots=True)
class Paired:
    """A run against one opponent, compared seed by seed with a recorded
    run's against the same opponent on the same seeds and seats."""

    player: str
    recorded: str
    """The recorded run's player."""
    opponent: str
    seeds: int
    """Seeds both runs played and finished."""
    only_player: tuple[int, ...]
    """Seeds the player won and the recorded player lost."""
    only_recorded: tuple[int, ...]

    @property
    def z(self) -> float:
        """The sign test's z: (only the player - only the recorded) / the
        square root of their sum (0 if no seed differs)."""
        a, b = len(self.only_player), len(self.only_recorded)
        return (a - b) / math.sqrt(a + b) if a + b else 0.0


def pair(match: Match, recorded: Sequence[Game]) -> Paired:
    """`match` against `recorded`, another player's games against the same
    opponent, on the seeds both played and finished."""
    theirs = {game.seed: game for game in recorded}
    rivals = {p for g in recorded for p in (g.first, g.second) if p != match.opponent}
    if len(rivals) != 1 or any(
        match.opponent not in (g.first, g.second) for g in recorded
    ):
        msg = f"recorded games aren't one player's against opponent {match.opponent}"
        raise ValueError(msg)
    (rival,) = rivals
    seeds, only_player, only_recorded = 0, [], []
    for game in match.games:
        other = theirs.get(game.seed)
        if other is None:
            continue
        if (game.first == match.player) != (other.first == rival):
            msg = f"seed {game.seed}: the recorded player played in the other seat"
            raise ValueError(msg)
        if game.winner is None or other.winner is None:
            continue
        seeds += 1
        won, they_won = game.winner == match.player, other.winner == rival
        if won and not they_won:
            only_player.append(game.seed)
        elif they_won and not won:
            only_recorded.append(game.seed)
    return Paired(
        match.player,
        rival,
        match.opponent,
        seeds,
        tuple(only_player),
        tuple(only_recorded),
    )


def report(matches: Sequence[Match], paired: Sequence[Paired] = ()) -> list[str]:
    """The run as plain-text lines: a table of each opponent's win rate with
    its interval and the player's record moving first, their mean, the
    unfinished games, and each pairing with a recorded run."""

    def rate(wins: int, games: int) -> str:
        return f"{wins / games:.1%}" if games else "-"

    def span(interval: tuple[float, float]) -> str:
        return f"{interval[0]:.1%}-{interval[1]:.1%}"

    rows = [("Opponent", "Wins", "Win%", "95% interval", "As 1st")]
    for m in matches:
        first = m.as_first()
        rows.append(
            (
                m.opponent,
                f"{m.wins}/{m.finished}",
                rate(m.wins, m.finished),
                span(m.interval),
                f"{first[0]}/{first[1]}",
            )
        )
    if matches:
        rows.append(
            ("Mean", "", f"{mean(matches):.1%}", span(mean_interval(matches)), "")
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    table = [
        "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip()
        for row in rows
    ]
    unfinished = [
        f"{m.opponent}: seed{'s' if len(m.unfinished) > 1 else ''}"
        f" {', '.join(map(str, m.unfinished))}"
        for m in matches
        if m.unfinished
    ]
    total = sum(len(m.unfinished) for m in matches)
    lines = [
        *table,
        "",
        f"Unfinished: {total}" + (f" ({'; '.join(unfinished)})" if unfinished else ""),
        "Win rates are of the finished games; the mean's interval is approximate.",
    ]
    for p in paired:
        lines.append(
            f"{p.player} against {p.opponent}, paired with {p.recorded} on"
            f" {p.seeds} seeds: won only by {p.player} {len(p.only_player)},"
            f" only by {p.recorded} {len(p.only_recorded)} (sign z {p.z:.2f})"
        )
    return lines
