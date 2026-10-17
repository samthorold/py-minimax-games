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
```

## Releasing

The version lives in `pyproject.toml` only; `minimax_games.__version__` reads it
from the installed package metadata.

Feature PRs don't change the version. To release, open a PR that only bumps it
(`task bump -- patch|minor|major`) and merge it. The Release workflow then tags
`vX.Y.Z` and publishes a GitHub Release with the built distributions and
generated notes. On PRs, the workflow's summary says whether merging will
publish a release.
