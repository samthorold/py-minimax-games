import copy
import dataclasses

import pytest

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.catalogue import EXPLORER
from minimax_games.games.star_realms.state import PlayerState


def changed(value):
    """A different value of the same type as a state field's `value`."""
    if isinstance(value, int):
        return value + 1
    if value and isinstance(value[0], PlayerState):  # the players: swap them
        first, second = value
        return (second, first)
    return (*value, EXPLORER)


@pytest.mark.parametrize(
    "state",
    [rules.new_game(seed=1), rules.new_game(seed=1).players[1]],
    ids=["GameState", "PlayerState"],
)
def test_replace_changes_only_the_given_field_like_dataclasses_replace(state):
    before = copy.deepcopy(state)
    for field in dataclasses.fields(state):
        value = changed(getattr(state, field.name))
        new = state.replace(**{field.name: value})
        assert new == dataclasses.replace(state, **{field.name: value})
        assert getattr(new, field.name) == value != getattr(state, field.name)
    assert state == before  # unchanged
