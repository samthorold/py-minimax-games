from collections.abc import Iterator

import pytest

from minimax_games.games.wordle.game import AutoScorer, UserGuesser, UserScorer
from minimax_games.games.wordle.node import (
    WordleNode,
    evaluate,
    prune,
    score_evaluation,
)


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


@pytest.mark.parametrize(
    ("aim", "guess", "expected"),
    [
        # The answer's only "e" is claimed by the exact match at the end.
        ("abcde", "eeeee", "....="),
        # Two "o"s guessed, one in the answer: the exact match claims it,
        # so the earlier copy is missing.
        ("thorn", "oxoxx", "..=.."),
        # Two "o"s in the answer, one guessed: it is present.
        ("robot", "oxxxx", "-...."),
        # Two "o"s in both: one exact, the other present.
        ("robot", "ooxxx", "-=..."),
        # Three "o"s guessed, two in the answer: the third is missing.
        ("robot", "oxxoo", "-..=."),
        ("crate", "crate", "====="),
        ("crate", "bulky", "....."),
    ],
)
def test_evaluate_follows_wordle_duplicate_letter_rules(aim, guess, expected):
    assert evaluate(aim, guess) == expected


PRUNE_VOCABULARY = [
    "robot",
    "thorn",
    "abcde",
    "eeeee",
    "oxoxx",
    "crate",
    "trace",
    "geese",
    "eerie",
    "speed",
]


@pytest.mark.parametrize("truth", PRUNE_VOCABULARY)
@pytest.mark.parametrize("guess", PRUNE_VOCABULARY)
def test_prune_never_discards_the_true_word(truth, guess):
    pruned = prune(PRUNE_VOCABULARY, [guess], [evaluate(truth, guess)])
    assert truth in pruned


SCORE_VOCABULARY = ["crate", "trace", "bogus", "bulky"]


def test_evaluation_node_is_scored_by_its_feedback():
    node = WordleNode(moves=["crate", "-==-."], vocabulary=SCORE_VOCABULARY, depth=3)
    assert node.score() == score_evaluation("-==-.")


@pytest.mark.parametrize("depth", [2, 4, 6])
def test_guess_node_scores_without_raising(depth):
    node = WordleNode(
        moves=["crate", ".....", "bogus"], vocabulary=SCORE_VOCABULARY, depth=depth
    )
    assert isinstance(node.score(), int)


def evaluation_node(feedback: str, guesses: int) -> WordleNode:
    """The node reached once ``guesses`` guesses have been scored."""
    moves = ["crate", "....."] * (guesses - 1) + ["crate", feedback]
    return WordleNode(moves=moves, vocabulary=SCORE_VOCABULARY, depth=1 + 2 * guesses)


def test_winning_leaf_scores_higher_with_fewer_guesses():
    scores = [evaluation_node("=====", guesses).score() for guesses in range(1, 7)]
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == len(scores)


@pytest.mark.parametrize("feedback", [".....", "-.-..", "====-", "====."])
def test_winning_leaf_beats_any_non_winning_leaf(feedback):
    slowest_win = evaluation_node("=====", guesses=6).score()
    for guesses in range(1, 7):
        assert evaluation_node(feedback, guesses).score() < slowest_win


def test_losing_leaf_is_scored_by_its_feedback():
    losing = evaluation_node("-.=..", guesses=6)
    assert losing.is_terminal()
    assert losing.score() == score_evaluation("-.=..")


def test_real_scores_lie_strictly_between_the_sentinels():
    node = evaluation_node("=====", guesses=1)
    assert node.minimum().score() < node.score() < node.maximum().score()
    assert node.minimum() < evaluation_node(".....", guesses=6)
