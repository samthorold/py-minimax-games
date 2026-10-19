import dataclasses
from collections import Counter

import pytest

from minimax_games.games.star_realms import catalogue
from minimax_games.games.star_realms.card import Card, CardType, Faction
from minimax_games.games.star_realms.effect import (
    Authority,
    Choice,
    Combat,
    DestroyBase,
    Draw,
    Trade,
)


def test_parse_reads_a_card_with_primary_and_ally_effects():
    text = """
    [[card]]
    name = "Blob Fighter"
    faction = "Blob"
    type = "ship"
    cost = 1
    copies = 3
    primary = [{ combat = 3 }]
    ally = [{ draw = 1 }]
    """
    assert catalogue.parse(text) == (
        Card(
            name="Blob Fighter",
            cost=1,
            faction=Faction.BLOB,
            type=CardType.SHIP,
            primary=(Combat(3),),
            ally=(Draw(1),),
            copies=3,
        ),
    )


def card_toml(**fields: str) -> str:
    """One `[[card]]` table: a plain ship, with `fields` replacing or adding keys."""
    base = {
        "name": '"Test Ship"',
        "faction": '"Blob"',
        "type": '"ship"',
        "cost": "1",
        "copies": "1",
    }
    lines = (f"{key} = {value}" for key, value in (base | fields).items())
    return "[[card]]\n" + "\n".join(lines) + "\n"


@pytest.mark.parametrize(
    "effects",
    ["[{ teleport = 3 }]", '["teleport"]', "[{ trade = 1, combat = 1 }]", "[3]"],
)
def test_parse_rejects_an_unknown_effect_naming_the_card(effects: str):
    with pytest.raises(ValueError, match="Test Ship"):
        catalogue.parse(card_toml(primary=effects))


def test_parse_reads_choices_unit_effects_and_base_defense():
    text = card_toml(
        name='"Test Base"',
        type='"outpost"',
        defense="4",
        primary="[{ choice = [[{ authority = 1 }], [{ trade = 1 }, { draw = 1 }]] }]",
        scrap='[{ draw = 1 }, "destroy_base"]',
    )
    (card,) = catalogue.parse(text)
    assert card.type is CardType.OUTPOST
    assert card.defense == 4
    assert card.primary == (Choice(((Authority(1),), (Trade(1), Draw(1)))),)
    assert card.scrap == (Draw(1), DestroyBase())


def test_the_trade_deck_has_twenty_cards_per_faction_and_eighty_in_all():
    counts = Counter(card.faction for card in catalogue.TRADE_DECK)
    assert counts == dict.fromkeys(Faction, 20)
    assert len(catalogue.TRADE_DECK) == 80


def test_the_explorer_pile_holds_ten_unaligned_explorers():
    explorer = catalogue.EXPLORER
    assert explorer.faction is None
    assert explorer.cost == 2
    assert explorer.primary == (Trade(2),)
    assert explorer.scrap == (Combat(2),)
    assert explorer.copies == 10
    assert explorer not in catalogue.TRADE_DECK


SLICE_1_EFFECTS = frozenset({Trade, Combat, Authority, Draw})


def test_a_card_is_playable_when_every_effect_is_of_an_implemented_kind():
    for card in catalogue.CATALOGUE.values():
        effects = card.primary + card.ally + card.scrap
        assert catalogue.playable(card, SLICE_1_EFFECTS) == all(
            isinstance(e, Trade | Combat | Authority | Draw) for e in effects
        )


def test_the_slice_1_cards_are_the_ones_with_only_simple_effects():
    cards = catalogue.CATALOGUE.values()
    assert {
        card.name for card in cards if catalogue.playable(card, SLICE_1_EFFECTS)
    } == {
        "Scout",
        "Viper",
        "Explorer",
        "Blob Fighter",
        "Trade Pod",
        "Ram",
        "Battle Blob",
        "Mothership",
        "Blob Wheel",
        "The Hive",
        "Federation Shuttle",
        "Cutter",
        "Trade Escort",
        "Flagship",
        "Corvette",
        "Dreadnaught",
        "Space Station",
        "War World",
        "Battle Station",
    }


def test_the_starting_deck_is_eight_scouts_and_two_vipers_from_the_catalogue():
    deck = Counter(catalogue.STARTING_DECK)
    assert deck == {catalogue.CATALOGUE["Scout"]: 8, catalogue.CATALOGUE["Viper"]: 2}


@pytest.mark.parametrize(
    "fields",
    [
        {"faction": '"Pirates"'},
        {"type": '"station"'},
        {"defense": "4"},  # a ship has no defense
        {"type": '"base"'},  # a base needs one
        {"flavour": '"Blobs are scary"'},
        {"cost": '"one"'},
        {"copies": "0"},
    ],
)
def test_parse_rejects_malformed_cards_naming_the_card(fields: dict[str, str]):
    with pytest.raises(ValueError, match="Test Ship"):
        catalogue.parse(card_toml(**fields))


def test_parse_rejects_duplicate_names():
    with pytest.raises(ValueError, match="Test Ship"):
        catalogue.parse(card_toml() + card_toml())


def test_loading_is_deterministic_and_the_definitions_are_immutable():
    assert catalogue.load() == catalogue.load() == tuple(catalogue.CATALOGUE.values())
    card = catalogue.CATALOGUE["Ram"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        card.cost = 0  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError):
        catalogue.CATALOGUE["Ram"] = card  # ty: ignore[invalid-assignment]


def test_card_hashes_agree_with_equality():
    for card in catalogue.CATALOGUE.values():
        copy = dataclasses.replace(card)
        assert copy is not card
        assert copy == card
        assert hash(copy) == hash(card)
        assert dataclasses.replace(card, cost=card.cost + 1) != card
    cards = list(catalogue.CATALOGUE.values())
    assert len(set(cards)) == len(cards)
    assert {card: card.name for card in cards} == {
        card: card.name for card in catalogue.load()
    }
