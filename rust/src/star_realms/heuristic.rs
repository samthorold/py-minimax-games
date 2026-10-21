//! The heuristic baseline (`bots.HeuristicBot`), and a whole
//! heuristic-vs-heuristic game.
//!
//! It chooses from a view, as the Python bot does, and the differential
//! tests check it chooses the same action at every decision. Its scores
//! (`worth`, `keep`) are floats, computed with the Python operations in the
//! Python order; every term is a multiple of 0.5 well inside `f64`'s exact
//! range, so each sum is exact and only `deny`'s one division rounds, the
//! same IEEE division as Python's. Ties go to the first minimum, as
//! Python's `min` does (`first_min`).

use std::cmp::Ordering;

use super::bots::GameResult;
use super::card::{Card, CardType, Effect};
use super::rules::{Action, advance_legal, auto_play, legal_actions, new_game, winner};
use super::view::{PlayerView, View, observe};

/// What each card of its faction the bot owns adds to a buy.
pub const SYNERGY: f64 = 0.5;
/// For this many of its own first turns, the bot buys for trade; after
/// them, for combat.
pub const ECONOMY_TURNS: u32 = 2;
/// What each point of primary trade adds to a buy in the economy turns.
pub const ECONOMY: f64 = 1.0;
/// What each point of primary combat adds to a buy after the economy turns.
pub const AGGRESSION: f64 = 1.5;
/// What each point of defense adds to buying a base that isn't an Outpost.
pub const BASE_DEFENSE: f64 = 0.5;
/// The bot draws with Blob World instead of taking 5 combat once that
/// draws at least this many cards.
pub const BLOB_WORLD_DRAWS: u32 = 3;
/// At or below this much Authority, the bot's "or" choices take Authority
/// first.
pub const LOW_AUTHORITY: i32 = 10;

fn amount(effects: &[Effect], pick: fn(&Effect) -> Option<i32>) -> i32 {
    effects.iter().filter_map(pick).sum()
}

fn trade(effects: &[Effect]) -> i32 {
    amount(effects, |e| match e {
        Effect::Trade(n) => Some(*n),
        _ => None,
    })
}

fn combat(effects: &[Effect]) -> i32 {
    amount(effects, |e| match e {
        Effect::Combat(n) => Some(*n),
        _ => None,
    })
}

fn authority(effects: &[Effect]) -> i32 {
    amount(effects, |e| match e {
        Effect::Authority(n) => Some(*n),
        _ => None,
    })
}

/// Python's `min(items, key=key)`: the first item with the least key. Keys
/// are compared with `PartialOrd` (floats among them, never NaN here).
fn first_min<T, K: PartialOrd>(
    items: impl IntoIterator<Item = T>,
    key: impl Fn(&T) -> K,
) -> Option<T> {
    let mut best: Option<(T, K)> = None;
    for item in items {
        let k = key(&item);
        let better = match &best {
            None => true,
            Some((_, b)) => k.partial_cmp(b) == Some(Ordering::Less),
        };
        if better {
            best = Some((item, k));
        }
    }
    best.map(|(item, _)| item)
}

fn is_starter(card: Card) -> bool {
    matches!(card, Card::Scout | Card::Viper)
}

/// The player's own turn number, `(view.turn + 1) // 2`.
fn own_turn(view: &View) -> u32 {
    view.turn.div_ceil(2)
}

/// Every card the observer owns, in whichever pile (`bots._owned`).
fn owned(view: &View) -> Vec<Card> {
    let me = &view.players[view.observer];
    [&me.hand, &me.unseen, &me.in_play, &me.discard, &me.bases]
        .into_iter()
        .flatten()
        .copied()
        .collect()
}

/// The total defense of `player`'s Outposts (`bots._guard`).
fn guard(player: &PlayerView) -> i32 {
    player
        .bases
        .iter()
        .filter(|c| c.is_outpost())
        .map(|c| c.data().defense)
        .sum()
}

/// How much the bot wants to acquire `card` in its `own_turn`th turn,
/// owning `owned` (`bots._worth`), summed in Python's order.
fn worth(card: Card, own_turn: u32, owned: &[Card]) -> f64 {
    let data = card.data();
    let mut worth = f64::from(data.cost);
    if let Some(faction) = data.faction {
        let same = owned
            .iter()
            .filter(|c| c.data().faction == Some(faction))
            .count();
        worth += SYNERGY * same as f64;
    }
    if data.card_type == CardType::Base {
        worth += BASE_DEFENSE * f64::from(data.defense);
    }
    if own_turn <= ECONOMY_TURNS {
        worth += ECONOMY * f64::from(trade(data.primary));
    } else {
        worth += AGGRESSION * f64::from(combat(data.primary));
    }
    worth
}

/// How much the bot wants to keep `card` in hand rather than discard it
/// (`bots._keep`).
fn keep(card: Card, view: &View) -> f64 {
    let data = card.data();
    if own_turn(view) <= ECONOMY_TURNS {
        f64::from(data.cost) + ECONOMY * f64::from(trade(data.primary))
    } else {
        f64::from(data.cost) + AGGRESSION * f64::from(combat(data.primary))
    }
}

/// Python's `sum(keep(card, view) for card in cards)`: from the int 0, in
/// order.
fn keep_sum(cards: &[Card], view: &View) -> f64 {
    cards.iter().fold(0.0, |sum, &card| sum + keep(card, view))
}

/// Of `cards`, the one the bot least wants to play, ties by name
/// (`bots._least_wanted`).
fn least_wanted(cards: impl IntoIterator<Item = Card>, view: &View) -> Option<Card> {
    first_min(cards, |&c| (keep(c, view), c.name()))
}

fn cards_of(actions: &[Action], pick: fn(Action) -> Option<Card>) -> Vec<Card> {
    actions.iter().filter_map(|&a| pick(a)).collect()
}

/// `HeuristicBot.choose`: the heuristic's choice among `actions`, the legal
/// actions in the game `view` shows. See the Python bot's docstring for the
/// priority order.
pub fn heuristic_move(view: &View, actions: &[Action]) -> Action {
    // 0. Forced decisions.
    let discards = cards_of(actions, |a| match a {
        Action::Discard(c) => Some(c),
        _ => None,
    });
    if !discards.is_empty() {
        return Action::Discard(least_wanted(discards, view).expect("a discard"));
    }
    if view.must_scrap > 0 {
        let scraps = cards_of(actions, |a| match a {
            Action::ScrapHandCard(c) => Some(c),
            _ => None,
        });
        return Action::ScrapHandCard(least_wanted(scraps, view).expect("a scrap from hand"));
    }
    let options: Vec<(Card, u8)> = actions
        .iter()
        .filter_map(|&a| match a {
            Action::Choose(card, option) => Some((card, option)),
            _ => None,
        })
        .collect();
    if !options.is_empty() {
        return choose(view, &options);
    }
    let recycles: Vec<&[Card]> = actions
        .iter()
        .filter_map(|a| match a {
            Action::Recycle(cards) => Some(cards.as_slice()),
            _ => None,
        })
        .collect();
    if !recycles.is_empty() {
        return recycle(view, &recycles);
    }
    let frees = cards_of(actions, |a| match a {
        Action::AcquireFreeShip(c) => Some(c),
        _ => None,
    });
    if !frees.is_empty() {
        return Action::AcquireFreeShip(most_wanted(view, frees));
    }
    let culls: Vec<Action> = actions
        .iter()
        .copied()
        .filter(|a| matches!(a, Action::ScrapAndDraw(..)))
        .collect();
    if !culls.is_empty() {
        return scrap_to_draw(view, culls);
    }
    let copies = cards_of(actions, |a| match a {
        Action::Copy(c) => Some(c),
        _ => None,
    });
    if let Some(card) = first_min(copies, |&c| (-keep(c, view), c.name())) {
        return Action::Copy(card);
    }
    // 1. Thin the deck of Vipers and Scouts.
    if let Some(thin) = thin(view, actions) {
        return thin;
    }
    // 2. Play everything.
    if let Some(&play) = actions.iter().find(|a| {
        matches!(
            a,
            Action::PlayCard(_) | Action::UseAlly(_) | Action::UseBase(_)
        )
    }) {
        return play;
    }
    // 3. Destroy the toughest base for free.
    let destroys = cards_of(actions, |a| match a {
        Action::DestroyTarget(c) => Some(c),
        _ => None,
    });
    if let Some(card) = first_min(destroys, |&c| {
        let data = c.data();
        (-data.defense, -data.cost, c.name())
    }) {
        return Action::DestroyTarget(card);
    }
    // 4. Buy.
    let buys = cards_of(actions, |a| match a {
        Action::Acquire(c) => Some(c),
        _ => None,
    });
    if !buys.is_empty() {
        let card = most_wanted(view, buys);
        let on_top = Action::AcquireOnTop(card);
        return if actions.contains(&on_top) {
            on_top
        } else {
            Action::Acquire(card)
        };
    }
    // 5. Deny the opponent a Trade Row card.
    if let Some(deny) = deny(view, actions) {
        return deny;
    }
    // 6. Scrap for combat.
    if let Some(scrap) = scrap(view, actions) {
        return scrap;
    }
    // 7. Attack.
    attack(view, actions).unwrap_or(Action::EndTurn)
}

/// An "or" choice (`HeuristicBot._choose`).
fn choose(view: &View, options: &[(Card, u8)]) -> Action {
    let hand = &view.players[view.observer].hand;
    let mut gains = Vec::new();
    for &(card, option) in options {
        let effects = card.options()[usize::from(option)];
        if effects.contains(&Effect::DrawPerFactionCard) {
            if let Some(faction) = card.data().faction
                && view.factions_played[faction.index()] >= BLOB_WORLD_DRAWS
            {
                return Action::Choose(card, option);
            }
        } else if effects
            .iter()
            .any(|e| matches!(e, Effect::DiscardThenDraw(_)))
        {
            if hand.iter().any(|&c| is_starter(c)) {
                return Action::Choose(card, option);
            }
        } else {
            gains.push((card, option));
        }
    }
    best_gain(view, if gains.is_empty() { options } else { &gains })
}

/// Between trade, combat and Authority (`HeuristicBot._best_gain`).
fn best_gain(view: &View, options: &[(Card, u8)]) -> Action {
    let low = view.players[view.observer].authority <= LOW_AUTHORITY;
    let economy = own_turn(view) <= ECONOMY_TURNS;
    let (card, option) = first_min(options.iter().copied(), |&(card, option)| {
        let effects = card.options()[usize::from(option)];
        let (trade, combat, authority) = (trade(effects), combat(effects), authority(effects));
        let first = if low {
            authority
        } else if economy {
            trade
        } else {
            combat
        };
        ((-first, -trade, -combat, -authority), option)
    })
    .expect("an option");
    Action::Choose(card, option)
}

/// Recycling Station's discard (`HeuristicBot._recycle`).
fn recycle(view: &View, recycles: &[&[Card]]) -> Action {
    let starters: Vec<&[Card]> = recycles
        .iter()
        .copied()
        .filter(|cards| cards.iter().all(|&c| is_starter(c)))
        .collect();
    let only = if starters.is_empty() {
        recycles
            .iter()
            .copied()
            .filter(|cards| cards.len() == 1)
            .collect()
    } else {
        starters
    };
    let cards = first_min(only, |cards| {
        let names: Vec<&str> = cards.iter().map(|c| c.name()).collect();
        (-(cards.len() as i64), keep_sum(cards, view), names)
    })
    .expect("a recycle");
    Action::Recycle(super::rules::Few::new(cards).expect("at most two cards"))
}

/// Brain World's scrap (`HeuristicBot._scrap_to_draw`).
fn scrap_to_draw(view: &View, culls: Vec<Action>) -> Action {
    let both = |a: &Action| match a {
        Action::ScrapAndDraw(hand, discard) => {
            let mut cards = hand.as_slice().to_vec();
            cards.extend_from_slice(discard.as_slice());
            (cards, discard.len())
        }
        _ => unreachable!("only ScrapAndDraw"),
    };
    let only = culls
        .into_iter()
        .filter(|a| both(a).0.iter().all(|&c| is_starter(c)));
    first_min(only, |a| {
        let (cards, discarded) = both(a);
        let names: Vec<&str> = cards.iter().map(|c| c.name()).collect();
        (
            -(cards.len() as i64),
            -(discarded as i64),
            keep_sum(&cards, view),
            names,
        )
    })
    .expect("a scrap to draw")
}

/// Scrap a Viper or Scout, from the discard pile first
/// (`HeuristicBot._thin`).
fn thin(view: &View, actions: &[Action]) -> Option<Action> {
    let from_discard = cards_of(actions, |a| match a {
        Action::ScrapDiscardCard(c) if is_starter(c) => Some(c),
        _ => None,
    });
    if !from_discard.is_empty() {
        return least_wanted(from_discard, view).map(Action::ScrapDiscardCard);
    }
    let from_hand = cards_of(actions, |a| match a {
        Action::ScrapHandCard(c) if is_starter(c) => Some(c),
        _ => None,
    });
    least_wanted(from_hand, view).map(Action::ScrapHandCard)
}

/// Of `cards`, the one worth most, ties by name
/// (`HeuristicBot._most_wanted`).
fn most_wanted(view: &View, cards: Vec<Card>) -> Card {
    let owned = owned(view);
    let turn = own_turn(view);
    first_min(cards, |&c| (-worth(c, turn, &owned), c.name())).expect("a card")
}

/// Scrap the Trade Row card the opponent would most want, unless the refill
/// would likely hand them better (`HeuristicBot._deny`).
fn deny(view: &View, actions: &[Action]) -> Option<Action> {
    let scraps = cards_of(actions, |a| match a {
        Action::ScrapTradeRowCard(c) => Some(c),
        _ => None,
    });
    if scraps.is_empty() {
        return None;
    }
    let opponent = &view.players[1 - view.observer];
    let owned: Vec<Card> = [
        &opponent.unseen,
        &opponent.in_play,
        &opponent.discard,
        &opponent.bases,
    ]
    .into_iter()
    .flatten()
    .copied()
    .collect();
    // The opponent's next turn is the game's next.
    let their_turn = (view.turn + 2) / 2;
    let value = |c: Card| worth(c, their_turn, &owned);
    let best = first_min(scraps, |&c| (-value(c), c.name()))?;
    if !view.trade_deck.is_empty() {
        let total = view.trade_deck.iter().fold(0.0, |sum, &c| sum + value(c));
        let refill = total / view.trade_deck.len() as f64;
        if value(best) < refill {
            return None;
        }
    }
    Some(Action::ScrapTradeRowCard(best))
}

/// Scrap cards for combat (`HeuristicBot._scrap`).
fn scrap(view: &View, actions: &[Action]) -> Option<Action> {
    let scraps = cards_of(actions, |a| match a {
        Action::Scrap(c) if combat(c.data().scrap) > 0 => Some(c),
        _ => None,
    });
    if scraps.is_empty() {
        return None;
    }
    let me = &view.players[view.observer];
    let opponent = &view.players[1 - view.observer];
    let reach = view.combat
        + scraps
            .iter()
            .map(|&card| {
                let copies = me.in_play.iter().chain(&me.bases).filter(|&&c| c == card);
                combat(card.data().scrap) * copies.count() as i32
            })
            .sum::<i32>();
    if reach >= opponent.authority + guard(opponent) {
        return first_min(scraps, |c| c.name()).map(Action::Scrap);
    }
    scraps
        .contains(&Card::Explorer)
        .then_some(Action::Scrap(Card::Explorer))
}

/// Attack (`HeuristicBot._attack`).
fn attack(view: &View, actions: &[Action]) -> Option<Action> {
    let opponent = &view.players[1 - view.observer];
    let bases = cards_of(actions, |a| match a {
        Action::AttackBase(c) => Some(c),
        _ => None,
    });
    let dearest = |c: &Card| (-c.data().cost, c.name());
    let outposts: Vec<Card> = bases.iter().copied().filter(|c| c.is_outpost()).collect();
    if !outposts.is_empty() {
        let target = if view.combat >= guard(opponent) {
            first_min(outposts, |c| (c.data().defense, c.name()))
        } else {
            first_min(outposts, dearest)
        };
        return target.map(Action::AttackBase);
    }
    let may_hit = actions.contains(&Action::AttackPlayer);
    if may_hit && view.combat >= opponent.authority {
        return Some(Action::AttackPlayer);
    }
    if let Some(card) = first_min(bases, dearest) {
        return Some(Action::AttackBase(card));
    }
    may_hit.then_some(Action::AttackPlayer)
}

/// Play a heuristic-vs-heuristic game from `seed`, as `match.play_match`
/// does, stopping unfinished once the turn passes `max_turns`.
pub fn play_heuristic(seed: u64, max_turns: u32) -> GameResult {
    let mut state = new_game(seed);
    auto_play(&mut state);
    let mut decisions = 0;
    while winner(&state).is_none() && state.turn <= max_turns {
        let view = observe(&state, state.current);
        let action = heuristic_move(&view, &legal_actions(&state));
        state = advance_legal(&state, action);
        decisions += 1;
    }
    GameResult {
        winner: winner(&state),
        turns: state.turn,
        decisions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_min_keeps_the_first_of_equal_keys() {
        assert_eq!(
            first_min([(1, 'a'), (0, 'b'), (0, 'c')], |p| p.0),
            Some((0, 'b'))
        );
        assert_eq!(first_min([0.5, 0.5], |&x| -x), Some(0.5));
    }

    #[test]
    fn heuristic_games_finish() {
        for seed in 0..20 {
            let result = play_heuristic(seed, 1000);
            assert!(result.winner.is_some(), "seed {seed}: {result:?}");
        }
    }
}
