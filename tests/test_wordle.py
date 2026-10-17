from collections.abc import Iterator

import pytest

from minimax_games.games.wordle.game import (
    AlphaBetaGuesser,
    AutoScorer,
    UserGuesser,
    UserScorer,
)
from minimax_games.games.wordle.node import (
    WordleNode,
    evaluate,
    prune,
    score_evaluation,
)
from minimax_games.search.alphabeta import alphabeta


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


@pytest.mark.parametrize("invalid", ["GGYBB", "xx-=.", "-= .="])
def test_user_scorer_reprompts_for_characters_other_than_dot_dash_equals(
    monkeypatch, capsys, invalid
):
    prompts = feed_input(monkeypatch, invalid, "-==-=")
    assert UserScorer()("trace") == "-==-="
    assert prompts == ["Score: ", "Score: "]
    assert "Score may only contain '.', '-' and '='" in capsys.readouterr().out


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


def test_prune_rejects_unknown_evaluation_characters():
    with pytest.raises(ValueError, match=r"^Unknown evaluation\.$"):
        prune(["crate", "trace"], ["crate"], ["GGYBB"])


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


def test_guess_node_score_does_not_depend_on_absolute_depth():
    scores = {
        WordleNode(
            moves=["crate", ".....", "bogus"], vocabulary=SCORE_VOCABULARY, depth=depth
        ).score()
        for depth in [2, 4, 6, 8]
    }
    assert len(scores) == 1


def test_guesser_picks_the_guess_with_the_best_guaranteed_feedback():
    # After two guesses three candidates remain. "abcde" shares three
    # letters in place with each of the others, so whatever the answer it
    # earns feedback worth 6. The other two share only their middle "c",
    # so guessing either can earn feedback worth just 2. The best guess is
    # last in the list: the guesser must search, not take the first word.
    vocabulary = ["abcxx", "vwcde", "abcde"]
    guesses = ["qqqqq", "rrrrr"]
    scores = [".....", "....."]
    assert AlphaBetaGuesser(vocabulary)(guesses, scores) == "abcde"


def test_deeper_guesser_plays_the_next_guess_of_its_line():
    # Three candidates remain. Guessing "fghij" earns "....." against both
    # others, so it can't tell them apart; "abcde" and "abcdk" can. Searching
    # two guesses ahead, the best line opens with "abcde" or "abcdk" and its
    # second guess is the answer, which may be "fghij". The guesser must
    # play the line's first guess, a legal candidate given all feedback.
    vocabulary = ["qqqqq", "rrrrr", "fghij", "abcde", "abcdk"]
    guesses = ["qqqqq", "rrrrr"]
    scores = [".....", "....."]
    guess = AlphaBetaGuesser(vocabulary, depth=4)(guesses, scores)
    assert guess in prune(vocabulary, guesses, scores, final_only=False)
    assert guess in {"abcde", "abcdk"}


def test_guesser_rejects_feedback_no_candidate_is_consistent_with():
    # Every remaining word contains "a", so "....." for "abcde" rules
    # them all out: the feedback can't have come from any of them.
    vocabulary = ["qqqqq", "abcde", "axxxx"]
    with pytest.raises(ValueError, match="inconsistent"):
        AlphaBetaGuesser(vocabulary)(["qqqqq", "abcde"], [".....", "....."])


# Words with repeated letters, so many answers share an evaluation.
SEARCH_VOCABULARY = [
    "crate",
    "trace",
    "eerie",
    "geese",
    "creep",
    "terse",
    "cheer",
    "reset",
    "tease",
    "crest",
]


@pytest.mark.parametrize(
    ("depth", "expected_line", "expected_score"),
    [
        (2, ["dolly", ".....", "terse", ".--.-"], 3),
        (4, ["dolly", ".....", "crate", "-==-=", "trace", "====="], 47),
    ],
)
def test_search_result_is_pinned_on_a_small_vocabulary(
    depth, expected_line, expected_score
):
    node = WordleNode(moves=["dolly", "....."], vocabulary=SEARCH_VOCABULARY, depth=3)
    line = alphabeta(node, node.minimum(), node.maximum(), depth=depth)
    assert line.moves == expected_line
    assert line.score() == expected_score


@pytest.mark.parametrize(
    ("guesses", "scores", "expected"),
    [
        ([], [], {2: "terse", 4: "crate"}),
        (["eerie"], ["---.."], {2: "creep", 4: "creep"}),
        (["eerie"], ["..-.="], {2: "crate", 4: "crate"}),
        (["geese"], [".=.=="], {2: "terse", 4: "terse"}),
    ],
)
def test_guesser_choice_is_pinned_on_a_small_vocabulary(guesses, scores, expected):
    # Two misses first, so the guesser searches rather than playing its
    # fixed openings.
    guesses = ["dolly", "kudzu", *guesses]
    scores = [".....", ".....", *scores]
    for depth, guess in expected.items():
        guesser = AlphaBetaGuesser(SEARCH_VOCABULARY, depth=depth)
        assert guesser(guesses, scores) == guess


def test_guess_node_has_one_child_per_distinct_evaluation():
    # "creep", "cheer" and "crest" all score "---.." against "eerie".
    node = WordleNode(
        moves=["dolly", ".....", "eerie"], vocabulary=SEARCH_VOCABULARY, depth=4
    )
    evaluations = [child.moves[-1] for child in node.children()]
    distinct = {evaluate(aim, "eerie") for aim in SEARCH_VOCABULARY}
    assert len(distinct) < len(SEARCH_VOCABULARY)
    assert sorted(evaluations) == sorted(distinct)


def test_search_leaves_the_root_vocabulary_unchanged():
    vocabulary = list(SEARCH_VOCABULARY)
    node = WordleNode(moves=["dolly", "....."], vocabulary=vocabulary, depth=3)
    alphabeta(node, node.minimum(), node.maximum(), depth=4)
    assert vocabulary == SEARCH_VOCABULARY
