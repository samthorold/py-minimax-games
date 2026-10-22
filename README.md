Minimax search to play various games.

<details>

<summary>Tic-tac-toe</summary>

```
uv run python -m minimax_games.games.tic_tac_toe
```

</details>

<details>

<summary>Wordle</summary>

```
uv run python -m minimax_games.games.wordle
```

</details>

## Development

Tooling is managed with [uv](https://docs.astral.sh/uv/): ruff (lint + format),
ty (type checking) and pytest (tests + doctests). Common commands are in
`Taskfile.yml`, run with [Task](https://taskfile.dev) (`brew install go-task`).

The package includes a Rust extension module, `minimax_games._native` (the
compiled Star Realms engine, from the crate in `rust/`), built by
[maturin](https://www.maturin.rs) as the package's build backend. It needs
[rustup](https://rustup.rs): `rust-toolchain.toml` pins the Rust version, which
rustup installs on first use. `uv sync` compiles the extension, and `uv run`
recompiles it whenever the Rust sources change, so there is no separate build
step. cargo fmt, clippy and cargo test run as pre-commit hooks beside the
Python ones.

```sh
uv sync                     # create .venv with dev tools, compile the extension
uv run pre-commit install   # install git hooks
uv run pre-commit run --all-files  # everything CI runs
task                        # list tasks
task test                   # uv run pytest (extra args: task test -- -k name)
task cargo-test             # the Rust engine's own tests
task lint                   # ruff, ty, cargo fmt --check, clippy
task fmt                    # ruff fix + format, cargo fmt
task native-bench           # Star Realms: the Rust engine's speed (games/s, advance calls/s)
task bench                  # Star Realms: the cheat's search speed (nodes/sec)
task turn-sizes             # Star Realms: distinct ends per turn (planner workload)
task worlds                 # Star Realms: does the champion's plan change with more sampled worlds
task profile                # Star Realms: profile the cheat's search
task tune                   # Star Realms: tune evaluate's weights by tournament
task fit                    # Star Realms: fit learned values, gated against evaluate
task population -- pf01     # Star Realms: a player against the reference population
task probe -- synergy       # Star Realms: probe the champion's judgement (or: denial)
```

### Stopping a Star Realms tournament early (SPRT)

`tournament --sprt` plays the paired match as usual (consecutive seeds,
alternating seats) but stops as soon as a sequential probability ratio test
decides between two hypotheses about bot A's win rate in finished games:
`--h0` (0.5) against `--h1` (0.55), with error rates `--alpha` and `--beta`
(0.05 each). `--games` is then the most it plays. It prints the decision (H0
or H1 accepted, or undecided at the limit), the games played, A's win rate
with its 95% Wilson interval, and the log-likelihood ratio and its bounds.
A clear result settles in tens of games:

```sh
uv run python -m minimax_games.games.star_realms tournament --bots greedy heuristic --games 500 --sprt
```

accepts H0 after 44 games (greedy wins 18%). The test only weighs H0 against
H1: accepting H0 means A is nearer 50% than 55%, or worse, not that the bots
are equal. With `--jobs N` games are played ahead in N processes, but the
decision is taken on completed games in seed order, so it is the same as one
process's.

### Tuning the Star Realms evaluation

`task tune` (`python -m minimax_games.games.star_realms tune`) tunes
`evaluate.Weights` by tournament, starting from the current defaults: a
coordinate search that, for each weight in `--fields` (all by default) in
order, plays the cheat with the weight doubled and halved (`--factor`)
against the heuristic, on the same `--games` seeds from `--base-seed` (200
from 3000 by default; keep them clear of seeds 0-499, the ladder's, and
1000-1499, milestone 5's confirmation) at `--budget` nodes, and keeps the
better change only if its extra wins over the incumbent pass `--z` (1.64)
standard deviations of noise, counted from the seeds whose outcome changed.
It prints each trial and ends with the tuned `Weights`; copy them into
`evaluate.Weights` by hand. At the defaults each trial is a 200-game
tournament (about 10 minutes with `--jobs 3`), so pass `--cache FILE` to keep
every trial: rerunning the same command replays them from the file and
carries on where it stopped. `--screen` is the cheap first step: per weight,
the share of the cheat's decisions (every `--every`-th of the `--games` games)
that doubling or halving it changes, most first, to pick which weights to
tune. Results: "Milestone 5: tuning" in
[docs/star-realms-results.md](docs/star-realms-results.md).

### Fitting a learned Star Realms value

`task fit` (`python -m minimax_games.games.star_realms fit`) plays seeded
heuristic-vs-heuristic games (8,000 from seed 20000 to train on, and the
next 2,000 held out by default), fits logistic regression and a
one-hidden-layer network, hand-written in numpy, to who won from each
turn-boundary state, and prints each one's held-out log-loss beside
`sigmoid(evaluate / 16)`'s on the same states. It is deterministic given
the seeds and takes under a minute with `--jobs 2`. `--save DIR` writes the
models as `.npz` files; a model that beats `evaluate` is committed in
`src/minimax_games/games/star_realms/models/` and loads by name
(`value.load("mlp")`). The data are never stored: the command regenerates
them. Results: "Milestone 8: offline value fit" in
[docs/star-realms-results.md](docs/star-realms-results.md).

### Measuring a Star Realms player against the reference population

`task population -- PLAYER` (`python -m minimax_games.games.star_realms
population PLAYER`) plays a player against the reference population (the
heuristic and every champion: `mlp`, `pi03`, `pf01`), leaving the player
itself out, all in Rust: 500 games against each on seeds 0-499 (`--games`),
the player first on even seeds. A player is `heuristic`, `greedy`, a
committed model (played by the planner with it) or the path of a saved
model (`.npz`). It prints the win rate against each opponent with its 95%
Wilson interval and the player's record moving first, their mean, and the
unfinished games, and writes the per-seed results (seed, first, second,
winner, turns, as `iterate` writes its matches) to
`runs/population/<player>/<opponent>.csv` (`--out`). `--paired DIR` pairs
each opponent's games seed by seed with a recorded run's, such as the
champion's in `docs/population/pf01/`, and reports the seeds won only by
each player and the sign test's z. The results are the same with any
`--threads` (8 by default). The champion's run takes a few minutes:

```sh
uv run python -m minimax_games.games.star_realms population my-candidate.npz --paired docs/population/pf01
```

Results: "Milestone 9: the reference population and pf01's baseline" in
[docs/star-realms-results.md](docs/star-realms-results.md).

### Probing a Star Realms judgement

`task probe -- synergy` (`python -m minimax_games.games.star_realms probe
synergy`) puts a question from Star Realms strategy to a trained judgement
(docs/star-realms.md, How its understanding is judged). It plays the
champion's games against itself in Rust (`--model`, `pf01` by default;
`--games` 200 from seed 12,000,000 on 2 `--threads`), takes every turn's end
as a position, changes one fact in each, and reports how the judged chance of
winning responds. The synergy probe adds a first, second and third card that
draws, scraps or makes the opponent discard (and, as a control, cards that do
none of these) to the deck of the player who just ended, and prints what each
card added, by game phase, in chances and in logits. The games take about 3
minutes; the probe itself plays none and takes a second or two. Results:
"Milestone 9: probe, same-function synergy" in
[docs/star-realms-results.md](docs/star-realms-results.md).

`task probe -- denial` asks whether a position is judged worse for the player
who just ended when a card the next player would value is in the Trade Row
(against one they wouldn't), by phase, and how many of the champion's buys
and Trade Row scraps are worth less in its own deck than kept from the
opponent, beside the heuristic's in its own games on the same seeds (recorded
in Rust, a few seconds). Results: "Milestone 9: probe, denial".

`task probe -- investment` weighs trade against combat: it adds, apart, a
ship giving only trade and one of the same cost giving only combat to the
deck of the player who just ended, and prints what each added per unit by
phase; then it plays the heuristic against itself on the same seeds and
prints the share of each player's buys that are economy, combat, mixed or
neither, by turn. About 2 minutes. Results: "Milestone 9: probe, investment
against the race".

### Profiling the Star Realms search

`task profile` (`python -m minimax_games.games.star_realms profile`) runs the
cheat's real workload under a profiler: its budgeted decisions (2 turns, 5,000
nodes) at the 8 benchmark positions and in the first 6 turns of seed 0's
cheat-vs-heuristic game (`--positions`, `--budget`, `--turns`, `--game-seed`,
`--game-turns`). It takes about 12 s and writes two files to `profiles/`
(`--out`), which git ignores:

- `profile.txt`, the report. First the workload's speed without and with the
  profiler. Then **time by area**: the search's cumulative time split into
  generating children (move generation, then `rules.advance_legal`: the
  automatic-play scan and applying actions), `evaluate`, the
  transposition table, and the search's own work; indented areas are part of
  the area above them, and us/node is per node generated. Then **hashing and
  equality**, the self time of every `__hash__` and `__eq__`, which all the
  areas call. Last, `pstats`'s own tables: the top 30 functions by self time
  (`tottime`, time in the function itself) and by cumulative time (`cumtime`,
  including what it calls); `ncalls` reads `total/primitive` for recursive
  functions.
- `profile.pstats`, the raw profile, for any `pstats` viewer: for instance
  `uvx snakeviz profiles/profile.pstats` for an interactive icicle chart, or
  `python -m pstats profiles/profile.pstats` to sort and filter it.

The profiler is the standard library's `cProfile`: no dependency, it runs
unattended, and it counts every call exactly, so call counts are the same on
every run (the decisions are deterministic) and only the times vary. It
slows the search about 3x, by roughly the same amount per call, so read its
times relative to each other, and confirm a hotspot's real cost by timing
without the profiler. (`pyinstrument` was tried: its overhead on this workload
is about the same, since it also hooks every call, and its call tree follows
`alphabeta`'s recursion many levels deep. `py-spy` samples without overhead
but needs `sudo` on macOS, so it can't run unattended.)

## Releasing

The version lives in `pyproject.toml` only; `minimax_games.__version__` reads it
from the installed package metadata.

Feature PRs don't change the version. To release, open a PR that only bumps it
(`task bump -- patch|minor|major`) and merge it. The Release workflow then tags
`vX.Y.Z` and publishes a GitHub Release with the built distributions and
generated notes. On PRs, the workflow's summary says whether merging will
publish a release.
