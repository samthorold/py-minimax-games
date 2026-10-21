//! The greedy baseline (`bots.GreedyBot`), for the first slice's actions, and
//! a whole greedy-vs-greedy game.

use super::rules::{Action, advance_legal, auto_play, legal_actions, new_game, winner};

/// `bots.greedy_move` among `actions`, the legal actions: the first play,
/// ally or base ability; else the dearest affordable card (ties by name);
/// else an attack on the player; else the weakest Outpost (ties by name);
/// else the end of the turn. The first slice has no forced decisions.
pub fn greedy_move(actions: &[Action]) -> Action {
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
        .min_by_key(|card| (-card.data().cost, card.name()));
    if let Some(card) = buy {
        return Action::Acquire(card);
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

/// Play a greedy-vs-greedy first-slice game from `seed`, as `match.play_match`
/// does, stopping unfinished once the turn passes `max_turns`.
pub fn play_greedy(seed: u64, max_turns: u32) -> GameResult {
    let mut state = new_game(seed);
    auto_play(&mut state);
    let mut decisions = 0;
    while winner(&state).is_none() && state.turn <= max_turns {
        let action = greedy_move(&legal_actions(&state));
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
        assert_eq!(greedy_move(&actions), Action::Acquire(Card::Cutter));
    }

    #[test]
    fn greedy_attacks_the_weakest_outpost() {
        let actions = [
            Action::AttackBase(Card::BattleStation),
            Action::AttackBase(Card::SpaceStation),
            Action::EndTurn,
        ];
        assert_eq!(
            greedy_move(&actions),
            Action::AttackBase(Card::SpaceStation)
        );
    }

    #[test]
    fn greedy_games_finish() {
        for seed in 0..20 {
            let result = play_greedy(seed, 1000);
            assert!(result.winner.is_some(), "seed {seed}: {result:?}");
        }
    }
}
