# minimax-games

Search applied to games, in the broad sense: game-tree search (alpha-beta, ISMCTS), planning, constraint search, and learned values and policies. Parts that need speed are ported to Rust inside this repo, with Python orchestrating. src layout: the package is `src/minimax_games/` (`search/` holds the generic `alphabeta` and the `Node` protocol, `games/` the tic-tac-toe and Wordle implementations), tests are in `tests/`, and the Rust crate behind the extension module `minimax_games._native` is in `rust/`. Distribution name `minimax-games`, Python >= 3.13.

A Star Realms game is planned: its agreed design is in [docs/star-realms.md](docs/star-realms.md). Follow it, and update it in the same PR as any change that departs from it.

How learned play and search compare, and where each is strong, is in [docs/learning-and-search.md](docs/learning-and-search.md); research notes behind open proposals (#127 a compiled engine, #128 a learned evaluation) are in `docs/research/`.

## Keep the feedback loop tight

Iterating on ideas quickly matters more than any single experiment. When a tournament, a benchmark, a training run or anything else in the loop is heading for tens of minutes, stop and ask what could make it cheaper before running it: a smaller or sequential test (stop a match as soon as its result is clear), an offline metric in place of a tournament, a cheaper opponent, a measurement that replaces a guess, or moving the hot path to Rust. Don't measure what is already known (for example, that CPython is too slow for self-play at scale).

## Tooling

Managed with uv. See the README's Development section for commands.

- Common commands live in `Taskfile.yml`. If `task` isn't installed, run it via `uvx --from go-task-bin task <name>`.
- `uv run pre-commit run --all-files` runs everything: ruff check, ruff format, ty, actionlint (which also runs shellcheck on workflow `run:` steps), shellcheck on shell scripts, cargo fmt, clippy and test, and pytest (including doctests and the Rust-vs-Python differential tests).
- **Rust**: the crate `rust/` builds the extension module `minimax_games._native` (the Rust Star Realms engine, `minimax_games._native.star_realms`), with maturin as the build backend (docs/star-realms.md, Compiled engine). `uv sync` compiles it; `uv run` recompiles it whenever a file in `[tool.uv] cache-keys` changes (the Rust sources among them), so there is no separate build step. The toolchain is pinned in `rust-toolchain.toml` at the repo root (rustup installs it on first use); upgrade it in its own PR. Rust commands: `cargo test --manifest-path rust/Cargo.toml` (`task cargo-test`), `task lint` / `task fmt` (which include cargo fmt and clippy), `task native-bench` (greedy games/s and `advance` calls/s).
- Rust traps: the PyO3 code is behind the crate's `python` feature (maturin turns it on), so plain `cargo test` needs no Python; clippy uses `--all-features` and runs through `uv run` so PyO3 finds the venv's Python. `src/minimax_games/_native/` holds only `.pyi` stubs for ty; keep them in step with `rust/src/python.rs`, and never add an `__init__.py` there (it would shadow the compiled module). The Python engine is the reference: a Rust rules change must keep `tests/star_realms/test_native.py` passing. Keep builds light on this 8 GB machine (`CARGO_BUILD_JOBS=2` when others are building). `task pypy` no longer installs (maturin builds a PyPy-tagged wheel that uv rejects under its `--python-version 3.13` workaround).
- Hooks call `uv run --locked`, so after changing dependencies run `uv lock` or the hooks fail. Dev dependencies are pinned exactly (`==`).
- **uv itself is pinned** by `[tool.uv] required-version` in `pyproject.toml`. CI's setup-uv reads it, and uv refuses to run locally at any other version. Upgrade deliberately in one PR, then `uv lock`. The build backend is maturin, bounded to a minor version in `[build-system]`; upgrade that deliberately too, checking `uv build` still makes the sdist and the wheel.
- CI (`.github/workflows/ci.yml`) installs the pinned Rust toolchain, restores the cargo cache, then runs the same pre-commit hooks on PRs. On `main` it only runs when `uv.lock`, `rust/Cargo.lock` or `rust-toolchain.toml` changes, to refresh the uv and cargo caches that PRs restore from.
- The CI job `pre-commit` is a required status check in the `main` ruleset. Renaming the job breaks merging until the ruleset is updated to the new name.
- `actionlint-py` ships only a source package, and building it downloads the binary from GitHub. On macOS with python.org Python, that fails with `CERTIFICATE_VERIFY_FAILED` until `Install Certificates.command` has been run. A one-off workaround is `SSL_CERT_FILE=/etc/ssl/cert.pem uv sync`.
- Pin GitHub Actions to exact versions. `astral-sh/setup-uv` doesn't publish floating major tags.
- **superenums** (Star Realms' `Action` and `Effect` enums) isn't on PyPI: it comes from GitHub, pinned to a tag in `[tool.uv.sources]` in `pyproject.toml`, and `uv.lock` pins the commit. To bump it, change the `tag` there to the new release (e.g. `v0.5.0`) and run `uv lock`, in its own PR. End every `match` over a superenum with `case _: assert_never(x)` so ty proves it exhaustive.
- `pyrightconfig.json` is gitignored: a local, optional strict pyright config. ty is the checker CI enforces.
- Wordle reads its vocabulary from `words/words.txt` (or `--vocabulary`), which isn't in the repo.

## Versioning

- Single source: `version` in `pyproject.toml`. `minimax_games.__version__` uses `importlib.metadata`, and `tests/test_version.py` checks they match.
- Releases happen on merge. `.github/workflows/release.yml` runs on every push to `main`, on a macOS arm64 runner: if `v<version>` isn't tagged yet, it builds the sdist and the macOS arm64 wheel (abi3, Python 3.13+), tags and publishes a GitHub Release. On PRs it's a dry run that also builds both. Its logic is in `.github/scripts/release-check.sh`.
- **Feature PRs never change the version.** A release is its own PR that only bumps the version (`task bump -- <part>`). Never create or push `v*` tags by hand. Hotfixes fix forward.
- **Merging a PR that changes the version publishes a release, so ask before merging it.**

## Workflow

- Work on a branch, open a PR against `main`. (The old `dev` branch predates this flow.)
- **PRs with passing CI may be squash-merged** (`gh pr merge <n> --squash --delete-branch`) without asking, except version bumps. Confirm CI is green (`gh pr checks <n> --watch`) first; never merge with failing or pending checks.
- The `main` ruleset requires PRs to be up to date. If a PR falls behind, run `gh pr update-branch <n>` and wait for CI again.
