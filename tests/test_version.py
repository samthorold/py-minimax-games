import tomllib
from pathlib import Path

import minimax_games

PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"


def test_version_matches_pyproject() -> None:
    with PYPROJECT.open("rb") as f:
        expected = tomllib.load(f)["project"]["version"]

    assert minimax_games.__version__ == expected
