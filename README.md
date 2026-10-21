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

```sh
uv sync                     # create .venv with dev tools
uv run pre-commit install   # install git hooks
task                        # list tasks
task test                   # uv run pytest (extra args: task test -- -k name)
task lint                   # ruff check, ruff format --check, ty check
task fmt                    # ruff fix + format
task bench                  # Star Realms: the cheat's search speed (nodes/sec)
task turn-sizes             # Star Realms: distinct ends per turn (planner workload)
task profile                # Star Realms: profile the cheat's search
task tune                   # Star Realms: tune evaluate's weights by tournament
task fit                    # Star Realms: fit learned values, gated against evaluate
task pypy -- tournament ... # Star Realms: a long tournament under PyPy
```

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
tune. Tuning is a long run: `task pypy -- tune ...` runs it under PyPy
(below), faster with the same results. Results: "Milestone 5: tuning" in
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

### Long Star Realms tournaments under PyPy

Search-bot tournaments run about 1.5-1.6x faster under PyPy 3.12, with
identical results (Milestone 5 in `docs/star-realms-results.md`). The project
supports CPython only (`requires-python = ">=3.13"`, and CI), so PyPy runs from
its own environment, `.venv-pypy` (git ignores it):

```sh
task pypy -- tournament --bots sampler heuristic --games 500 --jobs 3
```

which runs

```sh
uv venv --allow-existing --python pypy@3.12 .venv-pypy
uv pip install --python .venv-pypy --python-version 3.13 --reinstall-package minimax-games .
.venv-pypy/bin/python -m minimax_games.games.star_realms tournament ...
```

PyPy has no 3.13; `--python-version 3.13` makes uv resolve as if it did, so the
package installs on 3.12 (nothing in it needs 3.13). The package is installed,
not editable, and reinstalled from the working tree on every run. Use it only
for runs of many games: PyPy's JIT needs a few games to warm up, so short runs
(`bench --budget`, single games, the tests) are slower than on CPython. The
tests, `task profile` (PyPy's `cProfile` breaks its breakdown) and everything
else stay on CPython.

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
