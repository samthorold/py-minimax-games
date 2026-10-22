# minimax-games

Search applied to games, in the broad sense: game-tree search (alpha-beta, ISMCTS), planning, constraint search, and learned values and policies. Parts that need speed are ported to Rust inside this repo, with Python orchestrating. src layout: the package is `src/minimax_games/` (`search/` holds the generic `alphabeta` and the `Node` protocol, `games/` the tic-tac-toe and Wordle implementations), tests are in `tests/`, and the Rust crate behind the extension module `minimax_games._native` is in `rust/`. Distribution name `minimax-games`, Python >= 3.13.

A Star Realms game is planned: its agreed design is in [docs/star-realms.md](docs/star-realms.md). Follow it, and update it in the same PR as any change that departs from it. Its **End state** section is the player the work moves towards and how progress is measured: frame each experiment by the belief it tests. What players believe decides a game is in [docs/star-realms/strategy.md](docs/star-realms/strategy.md), and the terms (position, judgement, champion, reference population, probe) in [CONTEXT.md](CONTEXT.md).

How learned play and search compare, and where each is strong, is in [docs/learning-and-search.md](docs/learning-and-search.md); research notes behind open proposals (#127 a compiled engine, #128 a learned evaluation) are in `docs/research/`.

## Keep the feedback loop tight

Iterating on ideas quickly matters more than any single experiment. When a tournament, a benchmark, a training run or anything else in the loop is heading for tens of minutes, stop and ask what could make it cheaper before running it: a smaller or sequential test (stop a match as soon as its result is clear), an offline metric in place of a tournament, a cheaper opponent, a measurement that replaces a guess, or moving the hot path to Rust. Don't measure what is already known (for example, that CPython is too slow for self-play at scale).

## Tooling

The repo follows the Commons' Python Guide, including its Rust extensions section. Its specifics:

- Commands are in `Taskfile.yml` and the README's Development section. `uv run pre-commit run --all-files` also runs cargo fmt, clippy and test, and the Rust-vs-Python differential tests.
- **Rust**: the crate `rust/` builds `minimax_games._native` (the Rust Star Realms engine, `minimax_games._native.star_realms`; docs/star-realms.md, Compiled engine). Rust commands: `task cargo-test`, `task lint` / `task fmt` (cargo included), `task native-bench`. Keep the `.pyi` stubs in `src/minimax_games/_native/` in step with `rust/src/python.rs`. The Python engine is the reference: a Rust change must keep `tests/star_realms/test_native.py` passing. Keep builds light on this 8 GB machine (`CARGO_BUILD_JOBS=2` when others are building).
- PyPy isn't supported: maturin builds a PyPy-tagged wheel, and the Rust engine is the route to speed. Earlier results measured under PyPy stand.
- **superenums** (Star Realms' `Action` and `Effect` enums) isn't on PyPI: it comes from GitHub, pinned to a tag in `[tool.uv.sources]` in `pyproject.toml`, and `uv.lock` pins the commit. To bump it, change the `tag` there to the new release (e.g. `v0.5.0`) and run `uv lock`, in its own PR. End every `match` over a superenum with `case _: assert_never(x)` so ty proves it exhaustive.
- `pyrightconfig.json` is gitignored: a local, optional strict pyright config. ty is the checker CI enforces.
- Wordle reads its vocabulary from `words/words.txt` (or `--vocabulary`), which isn't in the repo.

## Versioning

Releases follow the Python Guide (one version in `pyproject.toml`, checked by `tests/test_version.py`; a release is its own version-bump PR, `task bump -- <part>`; release on merge, never tags by hand). Here `release.yml` runs on a macOS arm64 runner and publishes the sdist and the macOS arm64 wheel (abi3, Python 3.13+); its logic is in `.github/scripts/release-check.sh`.

## Workflow

The PR workflow (branch, PR, squash-merge when green, merging `main` in) is the Commons' Agentic development Guide. Here specifically:

- The required status check is the CI job `pre-commit`; the `main` ruleset requires PRs to be up to date.
- Version-bump PRs publish a release (Versioning, above): ask before merging them.
- The old `dev` branch predates this flow.
