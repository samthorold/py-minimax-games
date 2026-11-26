"""One recording of the champion's games, read by every instrument
(`recording`): reading it gives exactly what each instrument gives playing
its own games on the same seeds and settings."""

from pathlib import Path

import pytest

from minimax_games.games.star_realms import recording, value
from minimax_games.games.star_realms.cli import main

SMALL = ("--worlds", "1", "--budget", "40")
"""A planner cheap enough for tests."""


def _numbers(out: str) -> list[str]:
    """A probe's output without its timing line, which varies by run, and
    without the line saying where its games came from."""
    return [
        line
        for line in out.splitlines()
        if not line.startswith(("Games ", "The champion's games read from "))
    ]


def _record(tmp_path, *argv: str) -> str:
    path = tmp_path / "games.npz"
    main(["record", *argv, *SMALL, "--out", str(path)])
    return str(path)


@pytest.fixture(scope="module")
def self_play(tmp_path_factory) -> str:
    """The champion's games against itself on seeds 40-42, recorded."""
    return _record(
        tmp_path_factory.mktemp("self-play"),
        *("--mix", "self-play", "--champion", "pf01"),
        *("--games", "3", "--base-seed", "40"),
    )


MIXED = ("--champion", "pf01", "--pool", "mlp", "pf01")
MIXED_SEEDS = ("--games", "8", "--base-seed", "20")


@pytest.fixture(scope="module")
def mixed(tmp_path_factory) -> str:
    """The champion's games on seeds 20-27 mixed as a policy-iteration round
    mixes them, recorded."""
    return _record(
        tmp_path_factory.mktemp("mixed"),
        *("--mix", "policy-iteration", *MIXED, *MIXED_SEEDS),
    )


@pytest.mark.parametrize(
    ("command", "every"),
    [("gap", "1"), ("gap", "2"), ("diagnose", "3"), ("diagnose", "1")],
)
def test_gap_and_diagnose_read_a_recording_as_they_play_their_own_games(
    capsys, mixed, command, every
):
    asked = (command, *MIXED, *MIXED_SEEDS, "--every", every, *SMALL)
    capsys.readouterr()
    main([*asked, "--threads", "2"])
    played = capsys.readouterr().out
    main([*asked, "--recording", mixed])
    out = capsys.readouterr().out
    assert f"The champion's games read from {mixed}" in out
    assert _numbers(out) == played.splitlines()


def test_a_recording_names_its_champion_seeds_mix_and_settings(mixed):
    found = recording.load(Path(mixed))
    assert found.settings == recording.Settings(
        champion="pf01",
        seeds=range(20, 28),
        mix="policy-iteration",
        pool=("mlp", "pf01"),
        worlds=1,
        budget=40,
    )
    assert set(found.digests) == {"mlp", "pf01"}
    assert len(found.names) == 8
    assert {"pf01", "heuristic"} <= {name for pair in found.names for name in pair}


@pytest.mark.parametrize(
    "asked",
    [
        ("gap", *MIXED, "--games", "8", "--base-seed", "21", *SMALL),
        ("gap", *MIXED, *MIXED_SEEDS, "--worlds", "2", "--budget", "40"),
        ("diagnose", "--champion", "pf01", "--pool", "pf01", *MIXED_SEEDS, *SMALL),
        ("probe", "synergy", "--model", "pf01", *MIXED_SEEDS, *SMALL),
    ],
)
def test_an_instrument_refuses_a_recording_of_other_games(capsys, mixed, asked):
    with pytest.raises(SystemExit):
        main([*asked, "--recording", mixed])
    assert "isn't a recording of the games asked for" in capsys.readouterr().err


def test_an_instrument_refuses_a_recording_whose_champion_has_changed(capsys, tmp_path):
    champion = tmp_path / "cand.npz"
    value.load("pf01").save(champion)
    path = _record(
        tmp_path,
        *("--mix", "self-play", "--champion", str(champion)),
        *("--games", "1", "--base-seed", "40"),
    )
    value.load("mlp").save(champion)
    asked = ("--model", str(champion), "--games", "1", "--base-seed", "40", *SMALL)
    with pytest.raises(SystemExit):
        main(["probe", "synergy", *asked, "--recording", path])
    assert f"model {champion} has changed" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("probe", "extra"),
    [
        ("synergy", ("--samples", "1")),
        ("faction-pair", ("--samples", "1")),
        ("denial", ("--judge", "mlp")),
        ("investment", ()),
    ],
)
def test_a_probe_reads_a_recording_as_it_plays_its_own_games(
    capsys, self_play, probe, extra
):
    probed = ("--model", "pf01", "--games", "3", "--base-seed", "40", *SMALL)
    capsys.readouterr()
    main(["probe", probe, *probed, *extra])
    played = _numbers(capsys.readouterr().out)
    main(["probe", probe, *probed, *extra, "--recording", self_play])
    out = capsys.readouterr().out
    assert f"The champion's games read from {self_play}" in out
    assert _numbers(out) == played
