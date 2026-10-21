//! The greedy baseline (`bots.GreedyBot`, `bots.greedy_move`), and a whole
//! greedy-vs-greedy game.

use super::card::{Card, Effect};
use super::rules::{Action, advance_legal, auto_play, legal_actions, new_game, winner};

fn trade(effects: &[Effect]) -> i32 {
    effects
        .iter()
        .map(|e| match e {
            Effect::Trade(amount) => *amount,
            _ => 0,
        })
        .sum()
}

fn combat(effects: &[Effect]) -> i32 {
    effects
        .iter()
        .map(|e| match e {
            Effect::Combat(amount) => *amount,
            _ => 0,
        })
        .sum()
}

/// A sort key for giving up cards: the cheapest first, ties by name.
fn cheapest(card: Card) -> (i32, &'static str) {
    (card.data().cost, card.name())
}

/// A sort key for buying cards: the dearest first, ties by name.
fn dearest(card: Card) -> (i32, &'static str) {
    (-card.data().cost, card.name())
}

/// `bots.greedy_move` among `actions`, the legal actions, when the player
/// owes `must_scrap` scraps from hand. In priority order: a forced discard
/// or scrap from hand gives up the cheapest card; an "or" choice takes the
/// option with the most trade plus combat (ties by order); recycling gives
/// up the fewest and cheapest cards; a free ship is the dearest; Brain
/// World scraps nothing; Stealth Needle copies the ship with the most
/// primary trade plus combat; then the first play, ally or base ability;
/// else the dearest affordable card (on top of the deck when it may); else
/// an attack on the player; else the weakest Outpost; else the end of the
/// turn.
pub fn greedy_move(must_scrap: u32, actions: &[Action]) -> Action {
    let discards = actions.iter().filter_map(|&a| match a {
        Action::Discard(card) => Some(card),
        _ => None,
    });
    if let Some(card) = discards.min_by_key(|&c| cheapest(c)) {
        return Action::Discard(card);
    }
    if must_scrap > 0 {
        let scraps = actions.iter().filter_map(|&a| match a {
            Action::ScrapHandCard(card) => Some(card),
            _ => None,
        });
        if let Some(card) = scraps.min_by_key(|&c| cheapest(c)) {
            return Action::ScrapHandCard(card);
        }
    }
    let choice = actions
        .iter()
        .filter_map(|&a| match a {
            Action::Choose(card, option) => Some((card, option)),
            _ => None,
        })
        .min_by_key(|&(card, option)| {
            let effects = card.options()[usize::from(option)];
            (-trade(effects) - combat(effects), option)
        });
    if let Some((card, option)) = choice {
        return Action::Choose(card, option);
    }
    let recycle = actions
        .iter()
        .filter_map(|&a| match a {
            Action::Recycle(cards) => Some(cards),
            _ => None,
        })
        .min_by_key(|cards| {
            let cost: i32 = cards.as_slice().iter().map(|c| c.data().cost).sum();
            let names: Vec<&str> = cards.as_slice().iter().map(|c| c.name()).collect();
            (cards.len(), cost, names)
        });
    if let Some(cards) = recycle {
        return Action::Recycle(cards);
    }
    let free = actions
        .iter()
        .filter_map(|&a| match a {
            Action::AcquireFreeShip(card) => Some(card),
            _ => None,
        })
        .min_by_key(|&c| dearest(c));
    if let Some(card) = free {
        return Action::AcquireFreeShip(card);
    }
    let none = Action::ScrapAndDraw(super::rules::Few::EMPTY, super::rules::Few::EMPTY);
    if actions.contains(&none) {
        return none;
    }
    let copy = actions
        .iter()
        .filter_map(|&a| match a {
            Action::Copy(card) => Some(card),
            _ => None,
        })
        .min_by_key(|&card| {
            let primary = card.data().primary;
            (-trade(primary) - combat(primary), card.name())
        });
    if let Some(card) = copy {
        return Action::Copy(card);
    }
    if let Some(&play) = actions.iter().find(|a| {
        matches!(
            a,
            Action::PlayCard(_) | Action::UseAlly(_) | Action::UseBase(_)
        )
    }) {
        return play;
    }
    let buy = actions
        .iter()
        .filter_map(|&a| match a {
            Action::Acquire(card) => Some(card),
            _ => None,
        })
        .min_by_key(|&c| dearest(c));
    if let Some(card) = buy {
        let on_top = Action::AcquireOnTop(card);
        return if actions.contains(&on_top) {
            on_top
        } else {
            Action::Acquire(card)
        };
    }
    if actions.contains(&Action::AttackPlayer) {
        return Action::AttackPlayer;
    }
    let outpost = actions
        .iter()
        .filter_map(|&a| match a {
            Action::AttackBase(card) if card.is_outpost() => Some(card),
            _ => None,
        })
        .min_by_key(|card| (card.data().defense, card.name()));
    outpost.map_or(Action::EndTurn, Action::AttackBase)
}

/// How a whole game went.
#[derive(Debug, PartialEq, Eq)]
pub struct GameResult {
    /// The winner, or `None` if the game passed `max_turns`.
    pub winner: Option<usize>,
    pub turns: u32,
    /// The decisions taken: one `advance` each.
    pub decisions: u64,
}

/// Play a greedy-vs-greedy game from `seed`, as `match.play_match` does,
/// stopping unfinished once the turn passes `max_turns`.
pub fn play_greedy(seed: u64, max_turns: u32) -> GameResult {
    let mut state = new_game(seed);
    auto_play(&mut state);
    let mut decisions = 0;
    while winner(&state).is_none() && state.turn <= max_turns {
        let action = greedy_move(state.must_scrap, &legal_actions(&state));
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
    use super::super::card::Card;
    use super::*;

    #[test]
    fn greedy_buys_the_dearest_card_ties_by_name() {
        let actions = [
            Action::Acquire(Card::Cutter),
            Action::Acquire(Card::TradePod),
            Action::Acquire(Card::Explorer),
            Action::EndTurn,
        ];
        assert_eq!(greedy_move(0, &actions), Action::Acquire(Card::Cutter));
    }

    #[test]
    fn greedy_attacks_the_weakest_outpost() {
        let actions = [
            Action::AttackBase(Card::BattleStation),
            Action::AttackBase(Card::SpaceStation),
            Action::EndTurn,
        ];
        assert_eq!(
            greedy_move(0, &actions),
            Action::AttackBase(Card::SpaceStation)
        );
    }

    #[test]
    fn greedy_discards_its_cheapest_card() {
        let actions = [
            Action::Discard(Card::Cutter),
            Action::Discard(Card::Viper),
            Action::Discard(Card::Scout),
        ];
        assert_eq!(greedy_move(0, &actions), Action::Discard(Card::Scout));
    }

    #[test]
    fn greedy_games_finish() {
        for seed in 0..20 {
            let result = play_greedy(seed, 1000);
            assert!(result.winner.is_some(), "seed {seed}: {result:?}");
        }
    }
}
