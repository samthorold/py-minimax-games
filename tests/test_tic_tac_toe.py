import pytest

from minimax_games.games.tic_tac_toe.game import parse_move, read_move
from minimax_games.games.tic_tac_toe.node import Board, Player


def test_parse_move_reads_row_and_column_digits():
    assert parse_move("12") == (1, 2)


@pytest.mark.parametrize("text", ["1 2", "1,2", " 12 ", "1, 2", "  1   2"])
def test_parse_move_accepts_separated_and_padded_forms(text):
    assert parse_move(text) == (1, 2)


@pytest.mark.parametrize("text", ["33", "13", "30", "9 9"])
def test_parse_move_rejects_cells_outside_the_board(text):
    with pytest.raises(ValueError, match="Row and column must each be 0, 1 or 2"):
        parse_move(text)


@pytest.mark.parametrize("text", ["", "1", "111", "1 1 1", "12,"])
def test_parse_move_rejects_wrong_number_of_digits(text):
    with pytest.raises(ValueError, match="Enter a row and a column"):
        parse_move(text)


@pytest.mark.parametrize("text", ["ab", "1a", "-1 2", "1.2", "١٢"])
def test_parse_move_rejects_non_digits(text):
    with pytest.raises(ValueError, match="Enter a row and a column"):
        parse_move(text)


def test_read_move_plays_the_entered_move(feed_input):
    board = Board.from_string("o........", Player.X, depth=1)
    feed_input("12")

    result = read_move(board)

    assert result.string() == board.move((1, 2)).string()
    assert result.moves == [(1, 2)]


def test_read_move_reprompts_after_malformed_input(feed_input, capsys):
    board = Board.from_string("o........", Player.X, depth=1)
    prompts = feed_input("1 1 1", "33", "11")

    result = read_move(board)

    assert result.moves == [(1, 1)]
    assert len(prompts) == 3
    out = capsys.readouterr().out
    assert "Enter a row and a column" in out
    assert "Row and column must each be 0, 1 or 2" in out


def test_read_move_reprompts_when_the_cell_is_occupied(feed_input, capsys):
    board = Board.from_string("o........", Player.X, depth=1)
    prompts = feed_input("00", "01")

    result = read_move(board)

    assert result.moves == [(0, 1)]
    assert len(prompts) == 2
    assert "row=0, col=0 is occupied" in capsys.readouterr().out
