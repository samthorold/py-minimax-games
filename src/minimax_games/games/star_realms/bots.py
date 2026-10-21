"""Bots: anything that picks one of the legal actions.

A bot sees only its player's view of the game (`view.View`), never the state
(docs/star-realms.md). The one exception is a `BenchmarkBot`, such as the
cheat: a benchmark, not a legitimate player, which the match runner gives
the true state.
"""

import random
from typing import Protocol, runtime_checkable

from minimax_games.games.star_realms.action import (
    Acquire,
    AcquireFreeShip,
    AcquireOnTop,
    Action,
    AttackBase,
    AttackPlayer,
    Choose,
    Copy,
    DestroyTarget,
    Discard,
    EndTurn,
    PlayCard,
    Recycle,
    Scrap,
    ScrapAndDraw,
    ScrapDiscardCard,
    ScrapHandCard,
    ScrapTradeRowCard,
    UseAlly,
    UseBase,
)
from minimax_games.games.star_realms.card import Card, CardType, Faction
from minimax_games.games.star_realms.catalogue import EXPLORER, SCOUT, VIPER
from minimax_games.games.star_realms.effect import (
    Authority,
    Choice,
    Combat,
    DiscardThenDraw,
    DrawPerFactionCard,
    Effect,
    Trade,
)
from minimax_games.games.star_realms.state import GameState
from minimax_games.games.star_realms.view import PlayerView, View


class Bot(Protocol):
    def choose(self, view: View, actions: tuple[Action, ...]) -> Action:
        """Pick one of `actions`, the legal actions in the game `view` shows."""
        ...


@runtime_checkable
class BenchmarkBot(Protocol):
    """A benchmark-only bot: it sees the true state, so it is a yardstick, not
    a legitimate player. Implementing `choose_from_state` is how a bot marks
    itself benchmark-only; every other bot is a `Bot` and gets views only."""

    def choose_from_state(
        self, state: GameState, actions: tuple[Action, ...]
    ) -> Action:
        """Pick one of `actions`, the legal actions in the true `state`."""
        ...


@runtime_checkable
class Budgeted(Protocol):
    """A search bot: one that states its search settings (horizon, budget and
    so on), which the tournament command prints."""

    def settings(self) -> str:
        """The bot's search settings, as a phrase."""
        ...


@runtime_checkable
class Planning(Protocol):
    """A bot that plans ahead and plays its plan over several decisions (the
    whole-turn planner): it counts its plans and turns, so the tournament
    can report time per plan and plans per turn."""

    plans: int
    """Plans made so far."""
    turns: int
    """Turns played so far."""


type AnyBot = Bot | BenchmarkBot
"""A bot the match runner accepts: a `Bot`, or a `BenchmarkBot`."""


class RandomBot:
    """Picks uniformly at random among the legal actions, from its own seed."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def choose(self, view: View, actions: tuple[Action, ...]) -> Action:
        return self._rng.choice(actions)


class GreedyBot:
    """The greedy baseline: play everything, buy the dearest card, attack.

    Deterministic: the same actions give the same choice, whatever their
    order. It needs nothing from the view beyond what the legal actions
    already say. In priority order, it:

    0. when it must discard (an opponent's `OpponentDiscards`) or scrap a
       card from hand (Machine Base), discards or scraps its cheapest card,
       ties broken by name: that is the only decision offered then. When it
       must make an "or" choice (Patrol Mech, Trading Post, ...), it takes
       the option with the most trade plus combat, ties broken by the card's
       order (so the first option when neither has any): so it never
       draws with Blob World or recycles with Recycling Station, and if made
       to recycle it gives up as few and as cheap cards as it can. When it
       must take a free ship (Blob Carrier), it takes the most expensive,
       ties broken by name. When its Stealth Needle must copy a ship, it
       copies the one with the most primary trade plus combat, ties broken
       by name. With Brain World it scraps nothing, as it never scraps.
    1. plays every card it can and uses every ally and base ability, in
       `actions` order. Bots are only offered the plays automatic play leaves
       (draws, bases, and ships that could be scrapped from hand instead),
       and all of them are gains; playing before buying
       makes the most trade. Drawing first may reshuffle the discard pile
       before this turn's buys reach it, which a baseline ignores.
    2. acquires the most expensive affordable card, ties broken by name
       (Explorers are cost-2 options), and repeats while it can afford one,
       putting it on top of its deck when it may (Freighter, Central
       Office): a ship drawn next turn rather than shuffled in later.
    3. attacks the opponent: the player whenever it is allowed, which spends
       all the combat; otherwise, while Outposts guard the player, destroys
       the one with the least defense (ties by name). Weakest first destroys
       the most Outposts with the combat it has, so it reaches the player
       soonest. Other bases are ignored.
    4. ends the turn. It never scraps when it may: greedy keeps every card.
       Nor does it use destroy-base abilities (Missile Mech, Command Ship,
       ...): it leaves the targeting to the heuristic, as it leaves bases
       that don't guard the player. Nor does it scrap from the Trade Row
       (Battle Pod, Blob Destroyer).
    """

    def choose(self, view: View, actions: tuple[Action, ...]) -> Action:
        return greedy_move(view.must_scrap, actions)


def greedy_move(must_scrap: int, actions: tuple[Action, ...]) -> Action:
    """`GreedyBot`'s choice among `actions`, the legal actions, when the
    player owes `must_scrap` scraps from hand: all it reads, so a search's
    playouts can call it on a state without building a view."""
    if (forced := _forced(must_scrap, actions)) is not None:
        return min(forced, key=_cheapest)
    if options := _choices(actions):
        return min(
            options,
            key=lambda a: (-_trade(a.option) - _combat(a.option), _order(a)),
        )
    if recycles := _recycles(actions):
        return min(recycles, key=_fewest_cheapest)
    if frees := _free_ships(actions):
        return min(frees, key=lambda a: (-a.card.cost, a.card.name))
    if ScrapAndDraw((), ()) in actions:
        return ScrapAndDraw((), ())
    if copies := _copies(actions):
        return min(
            copies,
            key=lambda a: (
                -_trade(a.card.primary) - _combat(a.card.primary),
                a.card.name,
            ),
        )
    for action in actions:
        if isinstance(action, PlayCard | UseAlly | UseBase):
            return action
    buys = [a for a in actions if isinstance(a, Acquire)]
    if buys:
        buy = min(buys, key=lambda a: (-a.card.cost, a.card.name))
        return _on_top(buy, actions)
    if AttackPlayer() in actions:
        return AttackPlayer()
    outposts = [
        a
        for a in actions
        if isinstance(a, AttackBase) and a.card.type is CardType.OUTPOST
    ]
    if outposts:
        return min(outposts, key=lambda a: (a.card.defense or 0, a.card.name))
    return EndTurn()


type CardChoice = Discard | ScrapHandCard | ScrapDiscardCard
"""An action that is a choice of one of the player's cards to give up."""


def _forced(must_scrap: int, actions: tuple[Action, ...]) -> list[CardChoice] | None:
    """The card choices of a decision the bot must make before anything else,
    if one is pending: the discards it owes, or the scraps from hand it owes
    (`must_scrap`, as in `View.must_scrap`), else `None`."""
    discards: list[CardChoice] = [a for a in actions if isinstance(a, Discard)]
    if discards:
        return discards
    if must_scrap:
        return [a for a in actions if isinstance(a, ScrapHandCard)]
    return None


def _choices(actions: tuple[Action, ...]) -> list[Choose]:
    """The options of the "or" choice the bot must make, if one is pending
    (they are then the only actions), else empty."""
    return [a for a in actions if isinstance(a, Choose)]


def _recycles(actions: tuple[Action, ...]) -> list[Recycle]:
    """The sets of cards the bot may recycle, if it must choose one (they are
    then the only actions), else empty."""
    return [a for a in actions if isinstance(a, Recycle)]


def _on_top(buy: Acquire, actions: tuple[Action, ...]) -> Acquire | AcquireOnTop:
    """`buy`, putting the ship on top of the deck instead when the bot may
    (Freighter, Central Office): it is drawn next turn rather than shuffled
    in later."""
    on_top = AcquireOnTop(buy.card)
    return on_top if on_top in actions else buy


def _copies(actions: tuple[Action, ...]) -> list[Copy]:
    """The ships the bot's Stealth Needle may copy, if it must choose one
    (they are then the only actions), else empty."""
    return [a for a in actions if isinstance(a, Copy)]


def _scraps_to_draw(actions: tuple[Action, ...]) -> list[ScrapAndDraw]:
    """The choices of cards to scrap and draw for (Brain World), if the bot
    must make one (they are then the only actions), else empty."""
    return [a for a in actions if isinstance(a, ScrapAndDraw)]


def _free_ships(actions: tuple[Action, ...]) -> list[AcquireFreeShip]:
    """The ships the bot may acquire for free (Blob Carrier), if it must
    choose one (they are then the only actions), else empty."""
    return [a for a in actions if isinstance(a, AcquireFreeShip)]


def _fewest_cheapest(action: Recycle) -> tuple[int, int, tuple[str, ...]]:
    """A sort key for recycling: the fewest cards, then the cheapest, ties
    by name."""
    cards = action.cards
    return len(cards), sum(c.cost for c in cards), tuple(c.name for c in cards)


def _order(action: Choose) -> int:
    """Where `action`'s option comes among its card's options."""
    effects = action.card.primary + action.card.ally + action.card.scrap
    choice = next(e for e in effects if isinstance(e, Choice))
    return choice.options.index(action.option)


SYNERGY = 0.5
"""What each card of its faction the heuristic bot owns adds to a buy."""

ECONOMY_TURNS = 2
"""For this many of its own first turns, the heuristic bot buys for trade;
after them, for combat."""

ECONOMY = 1.0
"""What each point of primary trade adds to a buy in the economy turns."""

AGGRESSION = 1.5
"""What each point of primary combat adds to a buy after the economy turns."""

BASE_DEFENSE = 0.5
"""What each point of defense adds to buying a base that isn't an Outpost."""

BLOB_WORLD_DRAWS = 3
"""The heuristic bot draws with Blob World instead of taking 5 combat once
that draws at least this many cards."""

LOW_AUTHORITY = 10
"""At or below this much Authority, the heuristic bot's "or" choices take
Authority first."""


def _trade(effects: tuple[Effect, ...]) -> int:
    """The trade among `effects`."""
    return sum(e.amount for e in effects if isinstance(e, Trade))


def _combat(effects: tuple[Effect, ...]) -> int:
    """The combat among `effects`."""
    return sum(e.amount for e in effects if isinstance(e, Combat))


def _authority(effects: tuple[Effect, ...]) -> int:
    """The Authority among `effects`."""
    return sum(e.amount for e in effects if isinstance(e, Authority))


def _owned(view: View) -> tuple[Card, ...]:
    """Every card the observer owns, in whichever pile."""
    me = view.players[view.observer]
    return me.hand + me.unseen + me.in_play + me.discard + me.bases


def _guard(player: PlayerView) -> int:
    """The total defense of `player`'s Outposts."""
    return sum(c.defense or 0 for c in player.bases if c.type is CardType.OUTPOST)


def _worth(card: Card, own_turn: int, owned: tuple[Card, ...]) -> float:
    """How much the heuristic bot wants to acquire `card` in its `own_turn`th
    turn, owning `owned` (see its docstring). It values the opponent's buys
    the same way."""
    worth = card.cost
    if card.faction is not None:
        worth += SYNERGY * sum(c.faction is card.faction for c in owned)
    if card.type is CardType.BASE:
        worth += BASE_DEFENSE * (card.defense or 0)
    if own_turn <= ECONOMY_TURNS:
        worth += ECONOMY * _trade(card.primary)
    else:
        worth += AGGRESSION * _combat(card.primary)
    return worth


def _keep(card: Card, view: View) -> float:
    """How much the heuristic bot wants to keep `card` in hand rather than
    discard it (see its docstring)."""
    if (view.turn + 1) // 2 <= ECONOMY_TURNS:
        return card.cost + ECONOMY * _trade(card.primary)
    return card.cost + AGGRESSION * _combat(card.primary)


def _cheapest(action: CardChoice) -> tuple[int, str]:
    """A sort key for giving up cards: the cheapest first, ties by name."""
    return action.card.cost, action.card.name


def _least_wanted(choices: list[CardChoice], view: View) -> CardChoice:
    """Of `choices`, the card the heuristic bot least wants to play (see its
    docstring), ties broken by name."""

    def key(action: CardChoice) -> tuple[float, str]:
        return _keep(action.card, view), action.card.name

    return min(choices, key=key)


def _dearest(action: AttackBase) -> tuple[int, str]:
    """A sort key for attacks: the dearest base first, ties by name."""
    return -action.card.cost, action.card.name


def _toughest(action: DestroyTarget) -> tuple[int, int, str]:
    """A sort key for destroying bases for free: the most defense first,
    then the dearest, ties by name."""
    return -(action.card.defense or 0), -action.card.cost, action.card.name


class HeuristicBot:
    """The heuristic baseline: faction synergy, bases, scrapping and targets.

    Deterministic, and it uses only its view and the legal actions. The
    weights are module constants, tuned against greedy on seeds outside the
    test range. In priority order, it:

    0. when it must discard (an opponent's `OpponentDiscards`) or scrap a
       card from hand (Machine Base), discards or scraps the card it least
       wants to play, ties broken by name: the lowest cost plus, in its first
       `ECONOMY_TURNS` turns, `ECONOMY` per point of primary trade, and after
       them `AGGRESSION` per point of primary combat, the weights it buys
       with. So Vipers and Scouts go first: a Viper early, when trade builds
       the deck, and a Scout late.
       When it must make an "or" choice: with Recycling Station, it
       discards and draws while its hand holds a Viper or Scout, and then
       recycles as many of them as it may (up to 2), the ones it least wants
       first, otherwise it takes the trade; with Blob World, it draws once
       that draws at least `BLOB_WORLD_DRAWS` cards (Blob cards played this
       turn), otherwise it takes the 5 combat. Between trade, combat and
       Authority, it takes the most Authority at or below `LOW_AUTHORITY`,
       otherwise the most trade in its first `ECONOMY_TURNS` turns and the
       most combat after them, ties broken by trade, then combat, then
       Authority, then the card's order: so trade early and combat late
       (Patrol Mech), combat over Authority (Defense Center), and trade over
       Authority (Trading Post, Barter World) unless its Authority is low.
       When it must take a free ship (Blob Carrier), it takes the one worth
       most, as it buys (4). Using Blob Carrier's ally ability is one of the
       abilities it uses at once (2), so the free ship is the one it wants
       most, whatever it costs, before it buys.
       With Brain World, it scraps as many Vipers and Scouts as it may (up
       to 2), from the discard pile first, then the ones it least wants
       (as above), and draws for them; other cards are kept. When its
       Stealth Needle must copy a ship, it copies the one it would least
       want to give up (as above, the highest such worth), ties broken by
       name: so trade early and combat late.
    1. while it may scrap from hand or discard pile (Trade Bot, Junkyard,
       ...), scraps a Viper or Scout, the one it least wants to play (as in
       0) first: from the discard pile while one is there, where it does
       nothing this turn, otherwise from hand, before playing it, since a
       thinner deck is worth more than one point of trade or combat. Other
       cards are kept.
    2. plays every card it can and uses every ally and base ability, in
       `actions` order, as greedy does: all of them are gains.
    3. while it may destroy one of the opponent's bases (Missile Mech,
       Command Ship, ...), destroys the one with the most defense, Outpost
       or not, ties broken by the dearest then by name: that is the combat
       the free destroy saves, and Outposts don't protect bases from it.
       Before attacking, so its combat goes to what is left.
    4. acquires the affordable card worth most, ties broken by name, and
       repeats while it can afford one. A card is worth its cost, plus
       `SYNERGY` for each card of its faction it already owns (for ally
       abilities), plus `BASE_DEFENSE` per point of defense for a base that
       isn't an Outpost (it keeps working every turn), plus, in its first
       `ECONOMY_TURNS` turns, `ECONOMY` per point of primary trade, and after
       them `AGGRESSION` per point of primary combat: economy early, combat
       late. Outposts get no bonus for their defense: tuning found it lost
       games, as an Outpost's defense only delays the opponent. It puts the
       ship on top of its deck when it may (Freighter, Central Office), as
       greedy does: so the first ship it buys then goes there.
    5. while it may scrap from the Trade Row (Battle Pod, Blob Destroyer),
       scraps the row card the opponent would most want, by the worth in 4
       for the opponent's cards and their next turn, ties broken by name,
       unless it is worth less to them than the average card left in the
       Trade Deck (the view shows its contents), which the refill would
       likely hand them instead. After buying, so it never scraps a card it
       would buy, and it then buys the refill if it wants and can afford it.
    6. scraps cards for combat (after buying, so their trade is spent):
       every card with a combat scrap ability when the combat in hand and in
       those abilities reaches the opponent's Authority plus the defense of
       their Outposts, so this turn wins; otherwise only Explorers, whose
       trade helps only the turn they are played, so 2 combat and a thinner
       deck are worth more than keeping them. Other cards are kept. (Scrap
       abilities without combat, such as trade or a draw and a base to
       destroy, aren't used.)
    7. attacks: while Outposts guard the opponent, destroys the weakest one
       first if the combat clears them all (the order doesn't matter then),
       otherwise the dearest one it can (ties by name), as it can't reach the
       player this turn anyway. With no Outposts, it hits the player if that
       wins; otherwise it destroys the dearest base it can first (ties by
       name), since a base works every turn and the damage to the player
       can wait, then hits the player with the rest.
    8. ends the turn.
    """

    def choose(self, view: View, actions: tuple[Action, ...]) -> Action:
        if (forced := _forced(view.must_scrap, actions)) is not None:
            return _least_wanted(forced, view)
        if options := _choices(actions):
            return self._choose(view, options)
        if recycles := _recycles(actions):
            return self._recycle(view, recycles)
        if frees := _free_ships(actions):
            return self._most_wanted(view, frees)
        if culls := _scraps_to_draw(actions):
            return self._scrap_to_draw(view, culls)
        if copies := _copies(actions):
            return min(copies, key=lambda a: (-_keep(a.card, view), a.card.name))
        if (thin := self._thin(view, actions)) is not None:
            return thin
        for action in actions:
            if isinstance(action, PlayCard | UseAlly | UseBase):
                return action
        destroys = [a for a in actions if isinstance(a, DestroyTarget)]
        if destroys:
            return min(destroys, key=_toughest)
        if (buy := self._buy(view, actions)) is not None:
            return buy
        if (deny := self._deny(view, actions)) is not None:
            return deny
        if (scrap := self._scrap(view, actions)) is not None:
            return scrap
        if (attack := self._attack(view, actions)) is not None:
            return attack
        return EndTurn()

    def _choose(self, view: View, options: list[Choose]) -> Action:
        hand = view.players[view.observer].hand
        gains = []
        for action in options:
            if any(isinstance(e, DrawPerFactionCard) for e in action.option):
                faction = action.card.faction
                if (
                    faction is not None
                    and view.factions_played[list(Faction).index(faction)]
                    >= BLOB_WORLD_DRAWS
                ):
                    return action
            elif any(isinstance(e, DiscardThenDraw) for e in action.option):
                if SCOUT in hand or VIPER in hand:
                    return action
            else:
                gains.append(action)
        return self._best_gain(view, gains or options)

    def _recycle(self, view: View, recycles: list[Recycle]) -> Action:
        starters = (SCOUT, VIPER)
        only = [a for a in recycles if all(c in starters for c in a.cards)] or [
            a for a in recycles if len(a.cards) == 1
        ]

        def key(action: Recycle) -> tuple[int, float, tuple[str, ...]]:
            keep = sum(_keep(card, view) for card in action.cards)
            return -len(action.cards), keep, tuple(c.name for c in action.cards)

        return min(only, key=key)

    def _scrap_to_draw(self, view: View, culls: list[ScrapAndDraw]) -> Action:
        starters = (SCOUT, VIPER)
        only = [a for a in culls if all(c in starters for c in a.hand + a.discard)]

        def key(action: ScrapAndDraw) -> tuple[int, int, float, tuple[str, ...]]:
            cards = action.hand + action.discard
            keep = sum(_keep(card, view) for card in cards)
            names = tuple(c.name for c in cards)
            return -len(cards), -len(action.discard), keep, names

        return min(only, key=key)

    def _best_gain(self, view: View, options: list[Choose]) -> Action:
        low = view.players[view.observer].authority <= LOW_AUTHORITY
        economy = (view.turn + 1) // 2 <= ECONOMY_TURNS

        def key(action: Choose) -> tuple[int, int, int, int]:
            trade, combat = _trade(action.option), _combat(action.option)
            authority = _authority(action.option)
            first = authority if low else trade if economy else combat
            return -first, -trade, -combat, -authority

        return min(options, key=lambda a: (key(a), _order(a)))

    def _thin(self, view: View, actions: tuple[Action, ...]) -> Action | None:
        starters = (SCOUT, VIPER)
        from_discard: list[CardChoice] = [
            a for a in actions if isinstance(a, ScrapDiscardCard) and a.card in starters
        ]
        from_hand: list[CardChoice] = [
            a for a in actions if isinstance(a, ScrapHandCard) and a.card in starters
        ]
        scraps = from_discard or from_hand
        return _least_wanted(scraps, view) if scraps else None

    def _buy(self, view: View, actions: tuple[Action, ...]) -> Action | None:
        buys = [a for a in actions if isinstance(a, Acquire)]
        return _on_top(self._most_wanted(view, buys), actions) if buys else None

    def _most_wanted[A: Acquire | AcquireFreeShip](
        self, view: View, acquires: list[A]
    ) -> A:
        """Of `acquires`, the card worth most (see 4), ties broken by name."""
        owned = _owned(view)
        own_turn = (view.turn + 1) // 2
        return min(
            acquires, key=lambda a: (-_worth(a.card, own_turn, owned), a.card.name)
        )

    def _deny(self, view: View, actions: tuple[Action, ...]) -> Action | None:
        scraps = [a for a in actions if isinstance(a, ScrapTradeRowCard)]
        if not scraps:
            return None
        opponent = view.players[1 - view.observer]
        owned = opponent.unseen + opponent.in_play + opponent.discard + opponent.bases
        # The opponent's next turn is the game's next.
        their_turn = (view.turn + 2) // 2

        def worth(card: Card) -> float:
            return _worth(card, their_turn, owned)

        best = min(scraps, key=lambda a: (-worth(a.card), a.card.name))
        if view.trade_deck:
            refill = sum(map(worth, view.trade_deck)) / len(view.trade_deck)
            if worth(best.card) < refill:
                return None
        return best

    def _scrap(self, view: View, actions: tuple[Action, ...]) -> Action | None:
        scraps = [
            a for a in actions if isinstance(a, Scrap) and _combat(a.card.scrap) > 0
        ]
        if not scraps:
            return None
        me, opponent = view.players[view.observer], view.players[1 - view.observer]
        in_play = me.in_play + me.bases
        reach = view.combat + sum(
            _combat(a.card.scrap) * in_play.count(a.card) for a in scraps
        )
        if reach >= opponent.authority + _guard(opponent):
            return min(scraps, key=lambda a: a.card.name)
        if Scrap(EXPLORER) in scraps:
            return Scrap(EXPLORER)
        return None

    def _attack(self, view: View, actions: tuple[Action, ...]) -> Action | None:
        opponent = view.players[1 - view.observer]
        bases = [a for a in actions if isinstance(a, AttackBase)]
        outposts = [a for a in bases if a.card.type is CardType.OUTPOST]
        if outposts:
            if view.combat >= _guard(opponent):
                return min(outposts, key=lambda a: (a.card.defense or 0, a.card.name))
            return min(outposts, key=_dearest)
        if AttackPlayer() in actions and view.combat >= opponent.authority:
            return AttackPlayer()
        if bases:
            return min(bases, key=_dearest)
        if AttackPlayer() in actions:
            return AttackPlayer()
        return None
