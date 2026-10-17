from collections.abc import Iterator

import pytest

from minimax_games.games.wordle.game import AutoScorer, UserGuesser, UserScorer


def feed_input(monkeypatch: pytest.MonkeyPatch, *answers: str) -> list[str]:
    """Replace ``input`` with one returning ``answers`` in order.

    Returns the list of prompts shown, so tests can check how often
    the user was asked.
    """
    remaining: Iterator[str] = iter(answers)
    prompts: list[str] = []

    def fake_input(prompt: str = "") -> str:
        prompts.append(prompt)
        return next(remaining)

    monkeypatch.setattr("builtins.input", fake_input)
    return prompts


def test_auto_scorer_scores_guess_against_truth():
    scorer = AutoScorer(truth="crate")
    assert scorer("trace") == "-==-="
    assert scorer("crate") == "====="
    assert scorer("bulky") == "....."


def test_user_scorer_returns_normalised_score(monkeypatch, capsys):
    feed_input(monkeypatch, "  -==-=  ")
    assert UserScorer()("trace") == "-==-="
    assert "Guess: trace" in capsys.readouterr().out


def test_user_scorer_reprompts_until_five_characters(monkeypatch, capsys):
    prompts = feed_input(monkeypatch, "==", "-==-=")
    assert UserScorer()("trace") == "-==-="
    assert prompts == ["Score: ", "Score: "]
    assert "Score must be 5 letters long" in capsys.readouterr().out


def test_user_guesser_returns_normalised_known_word(monkeypatch):
    feed_input(monkeypatch, " CRATE ")
    guesser = UserGuesser(vocabulary=["crate", "trace"])
    assert guesser(guesses=[], scores=[]) == "crate"


def test_user_guesser_reprompts_for_wrong_length_and_unknown_words(monkeypatch, capsys):
    prompts = feed_input(monkeypatch, "cat", "bulky", "trace")
    guesser = UserGuesser(vocabulary=["crate", "trace"])
    assert guesser(guesses=[], scores=[]) == "trace"
    assert prompts == ["Guess: "] * 3
    out = capsys.readouterr().out
    assert "Guess must be 5 letters long" in out
    assert "Guess not a known word" in out


@pytest.mark.parametrize(
    "name", ["Guesser", "Scorer", "AutoScorer", "UserScorer", "UserGuesser"]
)
def test_player_classes_are_defined_only_in_game_module(name):
    from minimax_games.games.wordle import game, node

    assert getattr(game, name).__module__ == game.__name__
    assert not hasattr(node, name)
