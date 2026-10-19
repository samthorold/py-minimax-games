import dataclasses
import random
from collections import Counter

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.action import EndTurn, PlayCard
from minimax_games.games.star_realms.bots import RandomBot
from minimax_games.games.star_realms.catalogue import CATALOGUE, EXPLORER
from minimax_games.games.star_realms.state import GameState, PlayerState
from minimax_games.games.star_realms.view import observe, sample_world


def midgame(seed: int, moves: int = 60) -> GameState:
    """A state some random moves into the game from `seed`."""
    state = rules.new_game(seed)
    bot = RandomBot(seed)
    for _ in range(moves):
        if rules.winner(state) is not None:
            break
        actions = rules.legal_actions(state)
        state = rules.apply(state, bot.choose(observe(state, state.current), actions))
    return state


def shuffled[T](cards: tuple[T, ...], rng: random.Random) -> tuple[T, ...]:
    return tuple(rng.sample(cards, len(cards)))


def with_player(state: GameState, index: int, player: PlayerState) -> GameState:
    players = list(state.players)
    players[index] = player
    first, second = players
    return dataclasses.replace(state, players=(first, second))


def hide(state: GameState, observer: int, rng: random.Random) -> GameState:
    """`state` with only what `observer` can't see changed: every deck's order,
    which of the opponent's cards are in hand and which in the deck (keeping
    the sizes), the Trade Deck's order and the shuffle seed."""
    opponent = 1 - observer
    mine = state.players[observer]
    theirs = state.players[opponent]
    unseen = shuffled(theirs.hand + theirs.deck, rng)
    hand_size = len(theirs.hand)
    state = with_player(
        state, observer, dataclasses.replace(mine, deck=shuffled(mine.deck, rng))
    )
    state = with_player(
        state,
        opponent,
        dataclasses.replace(theirs, hand=unseen[:hand_size], deck=unseen[hand_size:]),
    )
    return dataclasses.replace(
        state,
        trade_deck=shuffled(state.trade_deck, rng),
        rng_seed=rng.getrandbits(64),
    )


def test_a_view_never_reveals_hidden_order_or_the_opponents_hand():
    rng = random.Random(0)
    for seed in range(20):
        state = midgame(seed, moves=rng.randrange(120))
        for observer in (0, 1):
            for _ in range(5):
                other = hide(state, observer, rng)
                assert observe(other, observer) == observe(state, observer)
                assert hash(observe(other, observer)) == hash(observe(state, observer))


def test_hiding_really_changes_the_state():
    rng = random.Random(1)
    state = midgame(3)
    assert all(hide(state, observer, rng) != state for observer in (0, 1))


def public_changes(state: GameState, observer: int) -> list[GameState]:
    """`state` with one thing the observer can see changed, in each way."""
    opponent = 1 - observer
    mine, theirs = state.players[observer], state.players[opponent]
    card = CATALOGUE["Blob Fighter"]
    changes = [
        dataclasses.replace(state, current=1 - state.current),
        dataclasses.replace(state, turn=state.turn + 1),
        dataclasses.replace(state, trade=state.trade + 1),
        dataclasses.replace(state, combat=state.combat + 1),
        dataclasses.replace(state, explorers=state.explorers - 1),
        dataclasses.replace(state, trade_row=(*state.trade_row, card)),
        dataclasses.replace(state, trade_deck=(*state.trade_deck, card)),
        dataclasses.replace(state, scrap_heap=(*state.scrap_heap, card)),
        dataclasses.replace(state, allies_used=(*state.allies_used, card)),
        dataclasses.replace(state, bases_used=(*state.bases_used, card)),
        # the observer's own hand, even with the same size and unseen cards
        with_player(state, observer, dataclasses.replace(mine, hand=(card,))),
    ]
    for index, player in ((observer, mine), (opponent, theirs)):
        changes += [
            with_player(state, index, dataclasses.replace(player, **{pile: value}))
            for pile, value in (
                ("authority", player.authority - 1),
                ("deck", (*player.deck, card)),
                ("in_play", (*player.in_play, card)),
                ("discard", (*player.discard, card)),
                ("bases", (*player.bases, card)),
                ("to_discard", player.to_discard + 1),
            )
        ]
    # the opponent's hand size, with the same cards unseen
    unseen = theirs.hand + theirs.deck
    changes.append(
        with_player(
            state,
            opponent,
            dataclasses.replace(theirs, hand=unseen[:3], deck=unseen[3:]),
        )
    )
    return changes


def test_a_view_shows_everything_public():
    for seed in range(5):
        state = midgame(seed)
        for observer in (0, 1):
            view = observe(state, observer)
            for changed in public_changes(state, observer):
                assert observe(changed, observer) != view


def test_a_sampled_world_looks_the_same_to_the_observer():
    rng = random.Random(2)
    for seed in range(20):
        for state in (rules.new_game(seed), midgame(seed, moves=rng.randrange(300))):
            for observer in (0, 1):
                view = observe(state, observer)
                assert observe(sample_world(view, rng), observer) == view


def test_a_sampled_world_keeps_pending_discards_and_so_the_legal_actions():
    fighter = CATALOGUE["Imperial Fighter"]
    rng = random.Random(5)
    for seed in range(10):
        state = midgame(seed)
        if rules.winner(state) is not None:
            continue
        # The player to move made the opponent discard, then ended the turn.
        me = state.players[state.current]
        state = with_player(
            state, state.current, dataclasses.replace(me, hand=(fighter,))
        )
        state = rules.apply(rules.apply(state, PlayCard(fighter)), EndTurn())
        pending = state.players[state.current].to_discard
        assert pending >= 1
        for observer in (0, 1):
            view = observe(state, observer)
            world = sample_world(view, rng)
            assert observe(world, observer) == view
            assert world.players[state.current].to_discard == pending
            if observer == state.current:
                assert set(rules.legal_actions(world)) == set(
                    rules.legal_actions(state)
                )


def test_the_same_rng_state_samples_the_same_world():
    view = observe(midgame(4), 0)
    assert sample_world(view, random.Random(7)) == sample_world(view, random.Random(7))


def small_game() -> GameState:
    """Player 0 has a two-card deck; player 1 has three distinct cards, one in
    hand and two in the deck; the Trade Deck has two cards."""
    blob, scout, viper = (CATALOGUE[n] for n in ("Blob Fighter", "Scout", "Viper"))
    return GameState(
        players=(
            PlayerState(50, deck=(scout, viper), hand=(blob,)),
            PlayerState(50, deck=(scout, viper), hand=(blob,)),
        ),
        rng_seed=0,
        trade_deck=(blob, scout),
    )


def test_sampling_reaches_every_hidden_arrangement():
    view = observe(small_game(), 0)
    rng = random.Random(3)
    worlds = [sample_world(view, rng) for _ in range(200)]
    assert {w.players[0].deck for w in worlds} == {
        tuple(CATALOGUE[n] for n in names)
        for names in (("Scout", "Viper"), ("Viper", "Scout"))
    }
    assert {w.players[1].hand[0].name for w in worlds} == {
        "Blob Fighter",
        "Scout",
        "Viper",
    }
    assert len({w.players[1].deck for w in worlds}) == 6
    assert len({w.trade_deck for w in worlds}) == 2
    assert len({w.rng_seed for w in worlds}) == 200


def test_sampling_never_mutates_the_view():
    view = observe(midgame(5), 1)
    copy = dataclasses.replace(view)
    sample_world(view, random.Random(0))
    assert view == copy


def all_cards(state: GameState) -> Counter[str]:
    """Every card in the game, by name, wherever it is."""
    piles = [
        state.trade_deck,
        state.trade_row,
        (EXPLORER,) * state.explorers,
        state.scrap_heap,
    ]
    for player in state.players:
        piles += [
            player.deck,
            player.hand,
            player.in_play,
            player.discard,
            player.bases,
        ]
    return Counter(card.name for pile in piles for card in pile)


def test_random_games_from_sampled_worlds_finish_and_conserve_cards():
    rng = random.Random(4)
    for seed in range(10):
        state = rules.auto_play(midgame(seed, moves=rng.randrange(120)))
        if rules.winner(state) is not None:
            continue
        world = sample_world(observe(state, state.current), rng)
        cards = all_cards(state)
        assert all_cards(world) == cards
        bot = RandomBot(seed)
        for _ in range(5000):
            if rules.winner(world) is not None:
                break
            actions = rules.legal_actions(world)
            world = rules.advance(
                world, bot.choose(observe(world, world.current), actions)
            )
            assert all_cards(world) == cards
        assert rules.winner(world) is not None
