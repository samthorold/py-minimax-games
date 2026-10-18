"""A hand-written linear evaluation of a state (docs/star-realms.md, Search).

Each player's terms are scored alike and differenced, so the score is
antisymmetric: `evaluate(s, p) == -evaluate(s, 1 - p)`. A finished game
scores `WIN` or `-WIN`, beyond any score of a game still going on.
"""

from dataclasses import dataclass

from minimax_games.games.star_realms import rules
from minimax_games.games.star_realms.card import Card, CardType
from minimax_games.games.star_realms.catalogue import SCOUT, VIPER
from minimax_games.games.star_realms.effect import Combat, Trade
from minimax_games.games.star_realms.state import GameState, PlayerState


@dataclass(frozen=True, slots=True)
class Weights:
    """What each term of the evaluation is worth, in points of Authority."""

    base_defense: float = 1
    """Per point of defense of a base in play (not an Outpost)."""
    outpost_defense: float = 2
    """Per point of defense of an Outpost in play."""
    trade_per_card: float = 20
    """Per point of average primary trade per card the player owns."""
    combat_per_card: float = 20
    """Per point of average primary combat per card the player owns."""
    starter_card: float = -1
    """Per Scout or Viper the player still owns (a penalty, so negative)."""


WEIGHTS = Weights()
"""The weights `evaluate` uses unless given others."""

WIN = 10**9
"""The score of a won game (and, negated, of a lost one). Scores of games
still going on are capped strictly inside `-WIN`..`WIN`."""


def _owned(player: PlayerState) -> tuple[Card, ...]:
    """Every card `player` owns, in whichever pile."""
    return player.deck + player.hand + player.in_play + player.discard + player.bases


def _value(player: PlayerState, weights: Weights) -> float:
    value: float = player.authority
    owned = _owned(player)
    if owned:
        trade = sum(e.amount for c in owned for e in c.primary if isinstance(e, Trade))
        combat = sum(
            e.amount for c in owned for e in c.primary if isinstance(e, Combat)
        )
        value += weights.trade_per_card * trade / len(owned)
        value += weights.combat_per_card * combat / len(owned)
        value += weights.starter_card * sum(c in (SCOUT, VIPER) for c in owned)
    for base in player.bases:
        per_point = (
            weights.outpost_defense
            if base.type is CardType.OUTPOST
            else weights.base_defense
        )
        value += per_point * (base.defense or 0)
    return value


def evaluate(state: GameState, perspective: int, weights: Weights = WEIGHTS) -> int:
    """How good `state` is for player `perspective`: bigger is better."""
    won_by = rules.winner(state)
    if won_by is not None:
        return WIN if won_by == perspective else -WIN
    me, them = state.players[perspective], state.players[1 - perspective]
    score = round(_value(me, weights) - _value(them, weights))
    return max(-WIN + 1, min(WIN - 1, score))
