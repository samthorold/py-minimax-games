import random
from dataclasses import fields, replace

import pytest

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.action import Action, EndTurn
from minimax_games.games.star_realms.bench import position
from minimax_games.games.star_realms.bots import BenchmarkBot, GreedyBot
from minimax_games.games.star_realms.catalogue import CATALOGUE, SCOUT, VIPER
from minimax_games.games.star_realms.evaluate import evaluate
from minimax_games.games.star_realms.match import MatchTooLong, play_match
from minimax_games.games.star_realms.planner import (
    BUDGET,
    WORLDS,
    PlannerBot,
    merge_key,
    plan,
    reveals,
)
from minimax_games.games.star_realms.state import GameState, PlayerState
from minimax_games.games.star_realms.turn_sizes import (
    turn_ends,
    turn_starts,
    unordered,
)
from minimax_games.games.star_realms.value import ValueEvaluator, load
from minimax_games.games.star_realms.view import View, observe, sample_world


def ends_turn(action: Action, child: GameState) -> bool:
    return isinstance(action, EndTurn) or rules.winner(child) is not None


def brute_force(state: GameState) -> tuple[list[GameState], bool]:
    """Every end of the turn from `state`, by walking every order of every
    play with nothing merged, each end as `EndTurn` leaves it; and whether
    any play within the turn drew or revealed a card."""
    ends: list[GameState] = []
    chance = False

    def walk(state: GameState) -> None:
        nonlocal chance
        for action in rules.legal_actions(state):
            child = rules.advance_legal(state, action)
            if isinstance(action, EndTurn):
                ends.append(rules.apply(state, action))
            elif rules.winner(child) is not None:
                ends.append(child)
            else:
                chance = chance or reveals(state, child)
                walk(child)

    walk(state)
    return ends, chance


def follow(state: GameState, line: tuple[Action, ...]) -> GameState:
    """`state` after `line`, each action checked legal, the last `EndTurn`
    taken without the next player's automatic play."""
    for i, action in enumerate(line):
        if i == len(line) - 1 and isinstance(action, EndTurn):
            return rules.apply(state, action)
        state = rules.advance(state, action)
    return state


def no_chance_turns() -> list[GameState]:
    """Small turn starts of seeded heuristic games with the Trade Deck
    emptied, so buying reveals nothing, whose turn draws no card before it
    ends."""
    found = []
    for seed in (0, 1, 2):
        for start in turn_starts(seed):
            start = replace(start, trade_deck=())
            if not turn_ends(start, 300).complete:
                continue
            if not brute_force(start)[1]:
                found.append(start)
    return found


def test_with_one_world_and_no_chance_event_it_finds_the_brute_force_best():
    turns = no_chance_turns()
    assert len(turns) >= 15
    for start in turns:
        ends, _ = brute_force(start)
        best = max(evaluate(end, start.current) for end in ends)
        found = plan([start], budget=10_000)
        assert found.complete
        assert isinstance(found.line[-1], EndTurn) or rules.winner(
            follow(start, found.line)
        )
        assert evaluate(follow(start, found.line), start.current) == best
        assert found.value == best


def with_a_draw() -> GameState:
    """Player 0 holds a Survey Ship (trade and a draw, a decision) and an
    Explorer to scrap, with a Viper and Scouts to play and a buy to make."""
    survey, explorer = CATALOGUE["Survey Ship"], CATALOGUE["Explorer"]
    me = PlayerState(
        authority=40,
        hand=(survey, explorer, VIPER, SCOUT, SCOUT),
        deck=(SCOUT,) * 3 + (VIPER,) * 3 + (CATALOGUE["Cutter"],),
    )
    them = PlayerState(authority=40, hand=(SCOUT,) * 5, deck=(SCOUT,) * 5)
    row = tuple(CATALOGUE[n] for n in ("Cutter", "Corvette", "Blob Fighter"))
    deck = tuple(CATALOGUE[n] for n in ("Battle Pod", "Trade Pod", "Barter World"))
    return rules.auto_play(
        GameState(players=(me, them), rng_seed=3, trade_row=row, trade_deck=deck)
    )


def test_it_commits_only_up_to_the_next_chance_event():
    state = with_a_draw()
    worlds = [sample_world(observe(state, 0), random.Random(i)) for i in range(3)]
    found = plan(worlds, budget=3_000)
    assert not isinstance(found.line[-1], EndTurn)
    for world in worlds:
        # every action of the line but the last reveals nothing, the last does
        for i, action in enumerate(found.line):
            child = rules.advance(world, action)
            assert reveals(world, child) == (i == len(found.line) - 1)
            world = child


def turn_positions() -> list[GameState]:
    return [position(seed, turn) for seed in (0, 1, 2) for turn in (3, 8, 12)]


def test_every_line_stops_at_its_first_chance_event_in_every_world():
    for state in turn_positions():
        view = observe(state, state.current)
        worlds = [sample_world(view, random.Random(i)) for i in range(2)]
        found = plan(worlds, budget=600)
        for world in worlds:
            for i, action in enumerate(found.line):
                assert action in rules.legal_actions(world)
                child = rules.advance_legal(world, action)
                last = i == len(found.line) - 1
                assert last or not (reveals(world, child) or ends_turn(action, child))
                world = child


def test_a_play_that_reveals_nothing_leaves_every_world_with_the_same_view():
    # what `reveals` calls no chance event really is the same in every world
    for state in turn_positions():
        view = observe(state, state.current)
        worlds = [sample_world(view, random.Random(i)) for i in range(4)]
        for action in rules.legal_actions(worlds[0]):
            children = [rules.advance_legal(w, action) for w in worlds]
            if any(ends_turn(action, c) for c in children):
                continue
            hidden = {reveals(w, c) for w, c in zip(worlds, children, strict=True)}
            assert len(hidden) == 1
            if not hidden.pop():
                assert len({observe(c, state.current) for c in children}) == 1


def test_the_budget_is_deterministic_and_never_exceeded():
    state = position(1, 12)
    view = observe(state, state.current)
    worlds = [sample_world(view, random.Random(i)) for i in range(4)]
    full = plan(worlds, budget=200_000)
    for budget in (8, 50, 400, 3_000):
        found = plan(worlds, budget=budget)
        assert found.states <= budget
        assert found == plan(worlds, budget=budget)
        assert found.line[0] in rules.legal_actions(state)
        if found.states < full.states:
            assert not found.complete
    assert full.complete


def test_it_scores_ends_with_a_learned_value():
    state = with_a_draw()
    worlds = [sample_world(observe(state, 0), random.Random(i)) for i in range(2)]
    found = plan(worlds, budget=2_000, evaluator=ValueEvaluator(load("logistic")))
    assert found.complete
    assert -(10**6) <= (found.value or 0) <= 10**6
    assert found.line[0] in rules.legal_actions(state)


def test_the_bot_plans_at_each_turn_start_and_after_each_chance_event():
    bot = PlannerBot(seed=5, worlds=2, budget=300)
    state = rules.auto_play(rules.new_game(5))
    turn = None
    previous: tuple[GameState, GameState] | None = None
    other = GreedyBot()
    for _ in range(400):
        if rules.winner(state) is not None or state.turn > 12:
            break
        actions = rules.legal_actions(state)
        if state.current == 1:
            state = rules.advance(state, other.choose(observe(state, 1), actions))
            continue
        plans, diverged = bot.plans, bot.diverged
        action = bot.choose(observe(state, 0), actions)
        planned = bot.plans > plans
        new_turn = state.turn != turn
        after_chance = previous is not None and reveals(*previous)
        assert planned == (new_turn or after_chance or bot.diverged > diverged)
        turn = state.turn
        child = rules.advance(state, action)
        previous = (state, child) if not isinstance(action, EndTurn) else None
        state = child
    assert bot.turns >= 6
    assert bot.turns <= bot.plans


class OnlyViews:
    def __init__(self, bot: PlannerBot) -> None:
        self.bot = bot

    def choose(self, view: View, actions: tuple[Action, ...]) -> Action:
        assert isinstance(view, View)
        assert view.observer == view.current
        return self.bot.choose(view, actions)


def test_it_is_only_ever_given_its_own_view_and_replays_identically():
    assert not isinstance(PlannerBot(0), BenchmarkBot)

    def game() -> list[str]:
        lines: list[str] = []
        bots = (OnlyViews(PlannerBot(3, worlds=2, budget=200)), GreedyBot())
        with pytest.raises(MatchTooLong):
            play_match(bots, 3, max_turns=8, log=lines.append)
        return lines

    first = game()
    assert any("Player 1" in line for line in first)
    assert game() == first


def test_its_settings_and_defaults():
    bot = PlannerBot(0)
    assert (bot.worlds, bot.budget) == (WORLDS, BUDGET) == (4, 20_000)
    assert bot.settings() == (
        "4 worlds, budget 20000 states per plan shared by the worlds, linear evaluation"
    )
    with pytest.raises(ValueError, match="world"):
        PlannerBot(0, worlds=0)
    with pytest.raises(ValueError, match="budget"):
        PlannerBot(0, worlds=4, budget=7)


def test_past_a_tiny_budget_it_still_plays_a_legal_line():
    state = with_a_draw()
    for budget in (2, 3, 5):
        found = plan([state], budget=budget)
        assert found.states <= budget
        assert not found.complete
        assert found.line[0] in rules.legal_actions(state)


def test_states_merge_exactly_when_equal_up_to_pile_order():
    states = set()
    for start in turn_positions():
        frontier = [start]
        for _ in range(3):
            frontier = [
                rules.advance_legal(state, action)
                for state in frontier
                for action in rules.legal_actions(state)
                if not isinstance(action, EndTurn)
            ][:200]
            states.update(frontier)
    states = list(states)
    assert len(states) > 500
    keys = [merge_key(state) for state in states]
    assert len(set(keys)) == len({unordered(state) for state in states})
    assert len(set(keys)) < len(states)  # some differ only in pile order
    pairs = list(zip(keys, map(unordered, states), strict=True))
    assert len(set(pairs)) == len(set(keys))
    # every field is in the key
    key = merge_key(states[0])
    assert isinstance(key, tuple)
    assert len(key) == len(fields(GameState)) + 1
    assert len(key[0]) == len(fields(PlayerState))


def test_it_plans_again_when_automatic_play_goes_otherwise_than_planned():
    # Views don't keep the hand's order, which decides what automatic play
    # plays first after the discard: here Trade Escort, before Patrol Mech
    # could scrap it from hand, so it is played automatically, while in the
    # sampled worlds (hand sorted by name) Patrol Mech comes first and Trade
    # Escort waits as a decision.
    escort, mech = CATALOGUE["Trade Escort"], CATALOGUE["Patrol Mech"]
    me = PlayerState(
        authority=40,
        hand=(escort, mech, VIPER, SCOUT, SCOUT),
        deck=(SCOUT,) * 6,
        to_discard=1,
    )
    station = CATALOGUE["Battle Station"]
    them = PlayerState(
        authority=40, hand=(SCOUT,) * 5, deck=(SCOUT,) * 5, bases=(station,)
    )
    state = rules.auto_play(GameState(players=(me, them), rng_seed=1))
    bot = PlannerBot(0, worlds=2, budget=2_000)
    while state.current == 0:
        actions = rules.legal_actions(state)
        state = rules.advance(state, bot.choose(observe(state, 0), actions))
    assert (bot.plans, bot.diverged) == (2, 1)
    assert state.players[0].authority == 44  # Trade Escort was played
