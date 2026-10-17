from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.action import (
    Acquire,
    Action,
    AttackBase,
    AttackPlayer,
    EndTurn,
    PlayCard,
    Scrap,
    UseAlly,
    UseBase,
)
from minimax_games.games.star_realms.bots import GreedyBot, RandomBot
from minimax_games.games.star_realms.catalogue import CATALOGUE, EXPLORER
from minimax_games.games.star_realms.match import play_match
from minimax_games.games.star_realms.view import observe

VIEW = observe(rules.new_game(seed=1), 0)

CUTTER = CATALOGUE["Cutter"]  # cost 2
CORVETTE = CATALOGUE["Corvette"]  # cost 2
BATTLE_BLOB = CATALOGUE["Battle Blob"]  # cost 6
WHEEL = CATALOGUE["Blob Wheel"]  # base, defense 5
FRIGATE = CATALOGUE["Imperial Frigate"]  # scrap: draw a card
REDOUBT = CATALOGUE["Royal Redoubt"]  # outpost, defense 6
STATION = CATALOGUE["Space Station"]  # outpost, defense 4
TRADING_POST = CATALOGUE["Trading Post"]  # outpost, defense 4


def greedy(*actions: Action) -> Action:
    return GreedyBot().choose(VIEW, actions)


def test_greedy_acquires_the_most_expensive_affordable_card():
    assert greedy(
        Acquire(CUTTER), Acquire(BATTLE_BLOB), Acquire(EXPLORER), EndTurn()
    ) == Acquire(BATTLE_BLOB)


def test_greedy_breaks_ties_between_equal_costs_by_name_whatever_the_order():
    buys = (Acquire(EXPLORER), Acquire(CUTTER), Acquire(CORVETTE))
    for actions in (buys, buys[::-1]):
        assert greedy(*actions, EndTurn()) == Acquire(CORVETTE)


def test_greedy_plays_cards_and_uses_abilities_before_buying_or_scrapping():
    rest = (Scrap(FRIGATE), Acquire(BATTLE_BLOB), EndTurn())
    for first in (PlayCard(WHEEL), UseAlly(CUTTER), UseBase(WHEEL)):
        assert greedy(first, *rest) == first


def test_greedy_attacks_the_player_rather_than_bases_whenever_it_can():
    actions = (Scrap(FRIGATE), AttackBase(WHEEL), AttackPlayer(), EndTurn())
    assert greedy(*actions) == AttackPlayer()


def test_greedy_destroys_the_weakest_outpost_first_when_outposts_guard_the_player():
    outposts = (AttackBase(REDOUBT), AttackBase(TRADING_POST), AttackBase(STATION))
    for actions in (outposts, outposts[::-1]):
        assert greedy(Scrap(FRIGATE), *actions, EndTurn()) == AttackBase(STATION)


def test_greedy_keeps_its_cards_and_leaves_bases_that_dont_guard_the_player():
    assert greedy(Scrap(FRIGATE), AttackBase(WHEEL), EndTurn()) == EndTurn()


def test_greedy_beats_random_in_the_large_majority_of_seeded_games():
    """30 seeds, alternating seats. Measured: greedy wins all 30 (199 of
    200 over seeds 0-199), in about 22 turns on average."""
    wins = 0
    for seed in range(30):
        seat = seed % 2
        bots = [RandomBot(seed + 1000), RandomBot(seed + 1000)]
        bots[seat] = GreedyBot()
        # Greedy always attacks, so its games end: no MatchTooLong.
        result = play_match((bots[0], bots[1]), seed, max_turns=100)
        wins += result.winner == seat
    assert wins >= 27
