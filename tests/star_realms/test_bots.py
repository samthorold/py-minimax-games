import dataclasses

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.action import (
    Acquire,
    Action,
    AttackBase,
    AttackPlayer,
    Choose,
    DestroyTarget,
    Discard,
    EndTurn,
    PlayCard,
    Recycle,
    Scrap,
    ScrapDiscardCard,
    ScrapHandCard,
    ScrapTradeRowCard,
    UseAlly,
    UseBase,
)
from minimax_games.games.star_realms.bots import (
    BLOB_WORLD_DRAWS,
    LOW_AUTHORITY,
    GreedyBot,
    HeuristicBot,
    RandomBot,
)
from minimax_games.games.star_realms.card import Card
from minimax_games.games.star_realms.catalogue import CATALOGUE, EXPLORER, SCOUT, VIPER
from minimax_games.games.star_realms.effect import (
    Authority,
    Choice,
    Combat,
    DiscardThenDraw,
    DrawPerFactionCard,
    Trade,
)
from minimax_games.games.star_realms.match import play_match
from minimax_games.games.star_realms.view import View, observe

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


def test_greedy_discards_its_cheapest_card_ties_by_name_whatever_the_order():
    discards = (Discard(CUTTER), Discard(SCOUT), Discard(VIPER), Discard(EXPLORER))
    for actions in (discards, discards[::-1]):
        assert greedy(*actions) == Discard(SCOUT)


def test_greedy_makes_an_owed_scrap_from_hand_of_its_cheapest_card():
    view = dataclasses.replace(VIEW, must_scrap=1)
    scraps = (ScrapHandCard(CUTTER), ScrapHandCard(VIPER), ScrapHandCard(SCOUT))
    for actions in (scraps, scraps[::-1]):
        assert GreedyBot().choose(view, actions) == ScrapHandCard(SCOUT)


def test_greedy_never_scraps_when_it_may():
    view = dataclasses.replace(VIEW, may_scrap=1)
    actions = (ScrapHandCard(VIPER), ScrapDiscardCard(SCOUT), EndTurn())
    assert GreedyBot().choose(view, actions) == EndTurn()


def test_greedy_never_uses_a_destroy_base_ability():
    actions = (DestroyTarget(WHEEL), DestroyTarget(REDOUBT), EndTurn())
    assert greedy(*actions) == EndTurn()


def test_greedy_never_scraps_from_the_trade_row():
    actions = (ScrapTradeRowCard(CUTTER), ScrapTradeRowCard(BATTLE_BLOB), EndTurn())
    assert greedy(*actions) == EndTurn()


PATROL_MECH = CATALOGUE["Patrol Mech"]  # 3 trade or 5 combat
DEFENSE_CENTER = CATALOGUE["Defense Center"]  # 3 Authority or 2 combat


def choices(card: Card) -> tuple[Choose, ...]:
    """The `Choose` actions for `card`'s choice, in the card's order."""
    (choice,) = card.primary
    assert isinstance(choice, Choice)
    return tuple(Choose(card, option) for option in choice.options)


def test_greedy_chooses_the_option_with_the_most_trade_and_combat():
    for card, best in (
        (TRADING_POST, (Trade(1),)),
        (PATROL_MECH, (Combat(5),)),
        (DEFENSE_CENTER, (Combat(2),)),
    ):
        options = choices(card)
        for actions in (options, options[::-1]):
            assert greedy(*actions) == Choose(card, best)


RECYCLING_STATION = CATALOGUE["Recycling Station"]  # 1 trade or recycle 2
BLOB_WORLD = CATALOGUE["Blob World"]  # 5 combat or a card per Blob played


def test_greedy_takes_trade_or_combat_over_drawing_and_recycles_little():
    assert greedy(*choices(RECYCLING_STATION)) == Choose(RECYCLING_STATION, (Trade(1),))
    assert greedy(*choices(BLOB_WORLD)) == Choose(BLOB_WORLD, (Combat(5),))
    recycles = (Recycle((SCOUT, VIPER)), Recycle((CUTTER,)), Recycle((SCOUT,)))
    assert greedy(*recycles) == Recycle((SCOUT,))


def test_greedy_beats_random_in_the_large_majority_of_seeded_games():
    """30 seeds, alternating seats. Measured: greedy wins all 30 (500 of
    500 over seeds 0-499, with the Milestone 4 "or" choices deck), in
    about 23 turns on average."""
    wins = 0
    for seed in range(30):
        seat = seed % 2
        bots = [RandomBot(seed + 1000), RandomBot(seed + 1000)]
        bots[seat] = GreedyBot()
        # Greedy always attacks, so its games end: no MatchTooLong.
        result = play_match((bots[0], bots[1]), seed, max_turns=100)
        wins += result.winner == seat
    assert wins >= 27


def view_with(
    *,
    owned: tuple[Card, ...] = (),
    in_play: tuple[Card, ...] = (),
    opponent_authority: int = 50,
    opponent_bases: tuple[Card, ...] = (),
    turn: int = 1,
    combat: int = 0,
    authority: int = 50,
) -> View:
    """VIEW, observer 0 to move, owning the starting deck plus `owned` (in
    the discard pile) with `in_play`, against an opponent at
    `opponent_authority` with `opponent_bases`, with `combat` to spend."""
    me, them = VIEW.players
    me = dataclasses.replace(me, discard=owned, in_play=in_play, authority=authority)
    them = dataclasses.replace(them, authority=opponent_authority, bases=opponent_bases)
    return dataclasses.replace(VIEW, players=(me, them), turn=turn, combat=combat)


def heuristic(*actions: Action, view: View = VIEW) -> Action:
    return HeuristicBot().choose(view, actions)


def test_heuristic_buys_into_a_faction_it_already_owns():
    """Cutter (Trade Federation) and Corvette (Star Empire) both cost 2;
    late on, Corvette's 1 combat favours it unless the bot owns Federation."""
    buys = (Acquire(CUTTER), Acquire(CORVETTE), EndTurn())
    federation = view_with(owned=(CATALOGUE["Federation Shuttle"],) * 4, turn=30)
    empire = view_with(owned=(CATALOGUE["Imperial Frigate"],) * 4, turn=30)
    assert heuristic(*buys, view=federation) == Acquire(CUTTER)
    assert heuristic(*buys, view=empire) == Acquire(CORVETTE)


def test_heuristic_plays_cards_and_uses_abilities_before_anything_else():
    rest = (Scrap(EXPLORER), Acquire(BATTLE_BLOB), AttackPlayer(), EndTurn())
    for first in (PlayCard(WHEEL), UseAlly(CUTTER), UseBase(WHEEL)):
        assert heuristic(first, *rest) == first


WAR_WORLD = CATALOGUE["War World"]  # outpost, defense 4, cost 5
BATTLE_STATION = CATALOGUE["Battle Station"]  # outpost, defense 5, cost 3


def test_heuristic_destroys_the_most_valuable_outpost_it_cannot_get_past():
    """6 combat can't clear 13 defense of Outposts this turn, so it takes out
    the dearest one it can (War World, cost 5), not the weakest."""
    outposts = (WAR_WORLD, STATION, BATTLE_STATION)
    view = view_with(opponent_bases=outposts, combat=6)
    attacks = tuple(AttackBase(o) for o in outposts)
    for actions in (attacks, attacks[::-1]):
        assert heuristic(*actions, EndTurn(), view=view) == AttackBase(WAR_WORLD)


def test_heuristic_destroys_the_weakest_outpost_first_when_it_can_clear_them():
    """Order doesn't matter when all go: weakest first, as greedy does."""
    outposts = (REDOUBT, STATION)
    view = view_with(opponent_bases=outposts, combat=10)
    attacks = tuple(AttackBase(o) for o in outposts)
    for actions in (attacks, attacks[::-1]):
        assert heuristic(*actions, EndTurn(), view=view) == AttackBase(STATION)


HIVE = CATALOGUE["The Hive"]  # base, defense 5, cost 5


def test_heuristic_destroys_a_valuable_base_before_hitting_the_player():
    view = view_with(opponent_bases=(WHEEL, HIVE), combat=7)
    actions = (AttackBase(WHEEL), AttackBase(HIVE), AttackPlayer(), EndTurn())
    assert heuristic(*actions, view=view) == AttackBase(HIVE)


def test_heuristic_destroys_even_a_cheap_base_before_hitting_the_player():
    """A base keeps working every turn; the damage to the player waits."""
    view = view_with(opponent_bases=(WHEEL,), combat=7)
    actions = (AttackBase(WHEEL), AttackPlayer(), EndTurn())
    assert heuristic(*actions, view=view) == AttackBase(WHEEL)


def test_heuristic_hits_the_player_whatever_the_bases_when_that_wins():
    view = view_with(opponent_bases=(HIVE,), combat=7, opponent_authority=7)
    actions = (AttackBase(HIVE), AttackPlayer(), EndTurn())
    assert heuristic(*actions, view=view) == AttackPlayer()


def test_heuristic_destroys_the_toughest_base_it_may_before_buying_or_attacking():
    """A free destroy saves the most combat on the base with the most
    defense, Outpost or not (ties: the dearest, then by name)."""
    bases = (WHEEL, REDOUBT, HIVE, STATION)
    view = view_with(opponent_bases=bases, combat=7)
    destroys = tuple(DestroyTarget(b) for b in bases)
    rest = (Acquire(BATTLE_BLOB), AttackBase(STATION), EndTurn())
    for actions in (destroys, destroys[::-1]):
        assert heuristic(*actions, *rest, view=view) == DestroyTarget(REDOUBT)
    ties = (DestroyTarget(WHEEL), DestroyTarget(HIVE))
    for actions in (ties, ties[::-1]):
        assert heuristic(*actions, EndTurn(), view=view) == DestroyTarget(HIVE)
    assert heuristic(PlayCard(WHEEL), *destroys, view=view) == PlayCard(WHEEL)


def row_view(*, opponent_owned: tuple[Card, ...], trade_deck: tuple[Card, ...]) -> View:
    """VIEW late in the game, the opponent owning `opponent_owned` (in the
    discard pile), with `trade_deck` left to refill the Trade Row."""
    me, them = VIEW.players
    them = dataclasses.replace(them, discard=opponent_owned)
    return dataclasses.replace(VIEW, players=(me, them), trade_deck=trade_deck, turn=30)


def test_heuristic_scraps_the_row_card_the_opponent_wants_most_after_buying():
    """Cutter (Trade Federation) and Corvette (Star Empire) both cost 2; the
    opponent owns Federation cards, so it denies them the Cutter."""
    view = row_view(
        opponent_owned=(CATALOGUE["Federation Shuttle"],) * 4,
        trade_deck=(SCOUT, VIPER),
    )
    scraps = (ScrapTradeRowCard(CUTTER), ScrapTradeRowCard(CORVETTE))
    for actions in (scraps, scraps[::-1]):
        assert heuristic(*actions, EndTurn(), view=view) == ScrapTradeRowCard(CUTTER)
    buy = Acquire(CORVETTE)
    assert heuristic(*scraps, buy, EndTurn(), view=view) == buy
    assert heuristic(PlayCard(WHEEL), *scraps, view=view) == PlayCard(WHEEL)
    attack = (AttackPlayer(), EndTurn())
    assert heuristic(*scraps, *attack, view=view) == ScrapTradeRowCard(CUTTER)


def test_heuristic_leaves_the_row_when_a_refill_would_likely_be_better():
    """A refill is drawn from the Trade Deck the view shows: when the row's
    best card is worth less to the opponent than the average card left
    there, scrapping it would likely hand them something better."""
    view = row_view(opponent_owned=(), trade_deck=(BATTLE_BLOB, WAR_WORLD))
    scraps = (ScrapTradeRowCard(CUTTER), ScrapTradeRowCard(CORVETTE))
    assert heuristic(*scraps, EndTurn(), view=view) == EndTurn()


RAM = CATALOGUE["Ram"]  # scrap: 3 trade


def test_heuristic_scraps_cards_for_combat_when_that_wins_the_game():
    """3 combat, plus 4 from scrapping Battle Blob, reaches 7 Authority."""
    view = view_with(in_play=(BATTLE_BLOB,), combat=3, opponent_authority=7)
    actions = (Scrap(BATTLE_BLOB), AttackPlayer(), EndTurn())
    assert heuristic(*actions, view=view) == Scrap(BATTLE_BLOB)


def test_heuristic_keeps_its_good_cards_when_scrapping_wouldnt_win():
    view = view_with(in_play=(BATTLE_BLOB, RAM), combat=3, opponent_authority=8)
    actions = (Scrap(BATTLE_BLOB), Scrap(RAM), AttackPlayer(), EndTurn())
    assert heuristic(*actions, view=view) == AttackPlayer()


SHUTTLE = CATALOGUE["Federation Shuttle"]  # cost 1, 2 trade
FIGHTER = CATALOGUE["Blob Fighter"]  # cost 1, 3 combat


def test_heuristic_prefers_economy_early_and_combat_late():
    buys = (Acquire(SHUTTLE), Acquire(FIGHTER), EndTurn())
    assert heuristic(*buys, view=view_with(turn=1)) == Acquire(SHUTTLE)
    assert heuristic(*buys, view=view_with(turn=30)) == Acquire(FIGHTER)


def test_heuristic_chooses_trade_early_combat_late_and_authority_when_low():
    early, late = view_with(turn=1), view_with(turn=30)
    low = view_with(turn=30, authority=LOW_AUTHORITY)
    for card, view, best in (
        (PATROL_MECH, early, (Trade(3),)),
        (PATROL_MECH, late, (Combat(5),)),
        (DEFENSE_CENTER, early, (Combat(2),)),
        (DEFENSE_CENTER, late, (Combat(2),)),
        (DEFENSE_CENTER, low, (Authority(3),)),
        (TRADING_POST, late, (Trade(1),)),
        (TRADING_POST, low, (Authority(1),)),
    ):
        options = choices(card)
        for actions in (options, options[::-1]):
            assert heuristic(*actions, view=view) == Choose(card, best)


def test_heuristic_recycles_starting_cards_when_its_hand_holds_any():
    me = dataclasses.replace(VIEW.players[0], hand=(CUTTER, SCOUT, VIPER, SCOUT))
    view = dataclasses.replace(VIEW, players=(me, VIEW.players[1]), turn=30)
    recycle = Choose(RECYCLING_STATION, (DiscardThenDraw(2),))
    assert heuristic(*choices(RECYCLING_STATION), view=view) == recycle
    recycles = (
        Recycle((CUTTER,)),
        Recycle((CUTTER, SCOUT)),
        Recycle((SCOUT,)),
        Recycle((SCOUT, SCOUT)),
        Recycle((SCOUT, VIPER)),
        Recycle((VIPER,)),
    )
    # Late, a Scout is the card it least wants: the two Scouts go.
    assert heuristic(*recycles, view=view) == Recycle((SCOUT, SCOUT))
    me = dataclasses.replace(me, hand=(CUTTER, CORVETTE))
    view = dataclasses.replace(view, players=(me, VIEW.players[1]))
    assert heuristic(*choices(RECYCLING_STATION), view=view) == Choose(
        RECYCLING_STATION, (Trade(1),)
    )


def test_heuristic_draws_with_blob_world_once_it_would_draw_enough():
    def played(blobs: int) -> View:
        return dataclasses.replace(VIEW, factions_played=(blobs, 0, 0, 0))

    draw = Choose(BLOB_WORLD, (DrawPerFactionCard(),))
    combat = Choose(BLOB_WORLD, (Combat(5),))
    assert heuristic(*choices(BLOB_WORLD), view=played(BLOB_WORLD_DRAWS)) == draw
    assert heuristic(*choices(BLOB_WORLD), view=played(BLOB_WORLD_DRAWS - 1)) == combat


def test_heuristic_discards_starting_cards_first_viper_early_and_scout_late():
    """It keeps the card worth most to play: trade early, combat late."""
    discards = (Discard(EXPLORER), Discard(SCOUT), Discard(VIPER), Discard(CUTTER))
    for actions in (discards, discards[::-1]):
        assert heuristic(*actions, view=view_with(turn=1)) == Discard(VIPER)
        assert heuristic(*actions, view=view_with(turn=30)) == Discard(SCOUT)


def test_heuristic_scraps_explorers_for_combat_once_it_has_bought():
    """An Explorer's trade only helps the turn it is played: once spent, its
    2 combat and a thinner deck are worth more than keeping it."""
    view = view_with(in_play=(EXPLORER,), combat=3)
    buying = (Scrap(EXPLORER), Acquire(CUTTER), AttackPlayer(), EndTurn())
    bought = (Scrap(EXPLORER), AttackPlayer(), EndTurn())
    assert heuristic(*buying, view=view) == Acquire(CUTTER)
    assert heuristic(*bought, view=view) == Scrap(EXPLORER)


def test_heuristic_makes_an_owed_scrap_from_hand_of_the_card_it_least_wants():
    scraps = (ScrapHandCard(SCOUT), ScrapHandCard(VIPER), ScrapHandCard(CUTTER))
    for turn, least in ((1, VIPER), (30, SCOUT)):
        view = dataclasses.replace(view_with(turn=turn), must_scrap=1)
        for actions in (scraps, scraps[::-1]):
            assert heuristic(*actions, view=view) == ScrapHandCard(least)


def test_heuristic_scraps_vipers_and_scouts_it_may_before_anything_else():
    """From the discard pile first, where the card does nothing this turn;
    the one it least wants to play first: a Viper early, a Scout late."""
    rest = (PlayCard(SCOUT), Acquire(CUTTER), AttackPlayer(), EndTurn())
    both = (ScrapDiscardCard(SCOUT), ScrapDiscardCard(VIPER), ScrapDiscardCard(CUTTER))
    for turn, least in ((1, VIPER), (30, SCOUT)):
        view = dataclasses.replace(view_with(turn=turn), may_scrap=1)
        for scraps in (both, both[::-1]):
            actions = (ScrapHandCard(VIPER), *scraps, *rest)
            assert heuristic(*actions, view=view) == ScrapDiscardCard(least)
    view = dataclasses.replace(view_with(), may_scrap=1)
    in_hand = (ScrapHandCard(CUTTER), ScrapHandCard(SCOUT), ScrapDiscardCard(CUTTER))
    assert heuristic(*in_hand, *rest, view=view) == ScrapHandCard(SCOUT)
    # Other cards are kept.
    others = (ScrapHandCard(CUTTER), ScrapDiscardCard(EXPLORER))
    assert heuristic(*others, *rest, view=view) == PlayCard(SCOUT)


ESCORT = CATALOGUE["Trade Escort"]  # ship, cost 5, 4 combat


def test_heuristic_values_a_base_for_the_turns_it_stays_in_play():
    """The Hive (base, 3 combat) over Trade Escort (ship, 4 combat), both
    cost 5: a base works every turn until destroyed."""
    view = view_with(turn=30)
    buys = (Acquire(ESCORT), Acquire(HIVE), EndTurn())
    assert heuristic(*buys, view=view) == Acquire(HIVE)


def test_heuristic_beats_greedy_in_clearly_more_than_half_of_seeded_games():
    """40 seeds, alternating seats. Measured: the heuristic wins 32 of 40
    (448 of 500 over seeds 0-499) with the Milestone 4 "or" choices deck.
    Tuned on seeds from 1000 with the 30-card first slice."""
    wins = 0
    for seed in range(40):
        seat = seed % 2
        bots = [GreedyBot(), GreedyBot()]
        bots[seat] = HeuristicBot()
        result = play_match((bots[0], bots[1]), seed, max_turns=100)
        wins += result.winner == seat
    assert wins >= 28
