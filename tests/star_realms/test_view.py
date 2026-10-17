import dataclasses
import random

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.bots import RandomBot
from minimax_games.games.star_realms.catalogue import CATALOGUE
from minimax_games.games.star_realms.state import GameState, PlayerState
from minimax_games.games.star_realms.view import observe


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
