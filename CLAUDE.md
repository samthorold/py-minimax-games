# minimax-games

Minimax (alpha-beta) search applied to games. src layout: the package is `src/minimax_games/` (`search/` holds the generic `alphabeta` and the `Node` protocol, `games/` the tic-tac-toe and Wordle implementations), tests are in `tests/`. Distribution name `minimax-games`, Python >= 3.13.

A Star Realms game is planned: its agreed design is in [docs/star-realms.md](docs/star-realms.md). Follow it, and update it in the same PR as any change that departs from it.

## Tooling

Managed with uv. See the README's Development section for commands.

- Common commands live in `Taskfile.yml`. If `task` isn't installed, run it via `uvx --from go-task-bin task <name>`.
- `uv run pre-commit run --all-files` runs everything: ruff check, ruff format, ty, actionlint (which also runs shellcheck on workflow `run:` steps), shellcheck on shell scripts, and pytest (including doctests).
- Hooks call `uv run --locked`, so after changing dependencies run `uv lock` or the hooks fail. Dev dependencies are pinned exactly (`==`).
- **uv itself is pinned** by `[tool.uv] required-version` in `pyproject.toml`. CI's setup-uv reads it, and uv refuses to run locally at any other version. Upgrade deliberately in one PR: `required-version` and the `uv_build` bound in `[build-system]` together, then `uv lock`.
- CI (`.github/workflows/ci.yml`) runs the same pre-commit hooks on PRs. On `main` it only runs when `uv.lock` changes, to refresh the uv cache that PRs restore from.
- The CI job `pre-commit` is a required status check in the `main` ruleset. Renaming the job breaks merging until the ruleset is updated to the new name.
- `actionlint-py` ships only a source package, and building it downloads the binary from GitHub. On macOS with python.org Python, that fails with `CERTIFICATE_VERIFY_FAILED` until `Install Certificates.command` has been run. A one-off workaround is `SSL_CERT_FILE=/etc/ssl/cert.pem uv sync`.
- Pin GitHub Actions to exact versions. `astral-sh/setup-uv` doesn't publish floating major tags.
- `pyrightconfig.json` is gitignored: a local, optional strict pyright config. ty is the checker CI enforces.
- Wordle reads its vocabulary from `words/words.txt` (or `--vocabulary`), which isn't in the repo.

## Versioning

- Single source: `version` in `pyproject.toml`. `minimax_games.__version__` uses `importlib.metadata`, and `tests/test_version.py` checks they match.
- Releases happen on merge. `.github/workflows/release.yml` runs on every push to `main`: if `v<version>` isn't tagged yet, it builds, tags and publishes a GitHub Release. On PRs it's a dry run. Its logic is in `.github/scripts/release-check.sh`.
- **Feature PRs never change the version.** A release is its own PR that only bumps the version (`task bump -- <part>`). Never create or push `v*` tags by hand. Hotfixes fix forward.
- **Merging a PR that changes the version publishes a release, so ask before merging it.**

## Workflow

- Work on a branch, open a PR against `main`. (The old `dev` branch predates this flow.)
- **PRs with passing CI may be squash-merged** (`gh pr merge <n> --squash --delete-branch`) without asking, except version bumps. Confirm CI is green (`gh pr checks <n> --watch`) first; never merge with failing or pending checks.
- The `main` ruleset requires PRs to be up to date. If a PR falls behind, run `gh pr update-branch <n>` and wait for CI again.
