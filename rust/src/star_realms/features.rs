//! The feature encoder a learned value reads, a port of `features.py`: one
//! row of numbers per turn-boundary state, in the same columns, so the rows
//! are bit-identical to Python's (a differential test checks them).
//!
//! Per player, the one who has just ended their turn first, then the one
//! about to move: how many of each card they own in their draw pool (deck,
//! hand, discard pile and cards in play, a Stealth Needle's copy counted as
//! the Needle), how many of each base they have in play, their Authority,
//! their bases' defense and the discards they owe. Then the turn, and how
//! many of each card the Trade Row shows. Nothing depends on a pile's order.

use std::sync::OnceLock;

use super::card::Card;
use super::state::{GameState, PlayerState};

/// The bases and Outposts, in catalogue order (`features.BASES`).
fn bases() -> &'static [Card] {
    static BASES: OnceLock<Vec<Card>> = OnceLock::new();
    BASES.get_or_init(|| Card::ALL.into_iter().filter(|c| !c.is_ship()).collect())
}

const CARDS: usize = Card::ALL.len();

/// A player's columns: a pool count per card, a count per base, then
/// Authority, base defense and discards owed.
fn player_columns() -> usize {
    CARDS + bases().len() + 3
}

/// The row's length (`len(features.NAMES)`).
pub fn columns() -> usize {
    2 * player_columns() + 1 + CARDS
}

/// A catalogue card's index in `Card::ALL`, which lists the cards in the
/// enum's (derived) order, so a binary search finds it.
fn index(card: Card) -> usize {
    Card::ALL
        .binary_search(&card)
        .expect("a catalogue card, not a copy")
}

fn base_index(card: Card) -> usize {
    bases()
        .binary_search(&card)
        .expect("a base of the catalogue")
}

fn player(row: &mut [f64], player: &PlayerState) {
    let pool = [&player.deck, &player.hand, &player.discard, &player.in_play];
    for card in pool.into_iter().flatten() {
        row[index(card.original())] += 1.0;
    }
    for &base in &player.bases {
        row[CARDS + base_index(base)] += 1.0;
    }
    let authority = CARDS + bases().len();
    row[authority] = f64::from(player.authority);
    row[authority + 1] = f64::from(player.bases.iter().map(|b| b.data().defense).sum::<i32>());
    row[authority + 2] = f64::from(player.to_discard);
}

/// Write `state`'s row (`features.encode`) into `row`, of `columns()` zeros.
pub fn encode_into(row: &mut [f64], state: &GameState) {
    let width = player_columns();
    let ended = 1 - state.current;
    player(&mut row[..width], &state.players[ended]);
    player(&mut row[width..2 * width], &state.players[state.current]);
    row[2 * width] = f64::from(state.turn);
    let trade_row = 2 * width + 1;
    for &card in &state.trade_row {
        row[trade_row + index(card)] += 1.0;
    }
}

/// `state`'s row (`features.encode`).
pub fn encode(state: &GameState) -> Vec<f64> {
    let mut row = vec![0.0; columns()];
    encode_into(&mut row, state);
    row
}

#[cfg(test)]
mod tests {
    use super::super::rules::{auto_play, new_game};
    use super::*;

    #[test]
    fn the_catalogue_is_in_the_enum_s_order() {
        assert!(Card::ALL.is_sorted());
        assert!(bases().is_sorted());
    }

    #[test]
    fn a_new_game_counts_the_next_player_s_starting_deck() {
        let mut state = new_game(0);
        auto_play(&mut state);
        let row = encode(&state);
        let width = player_columns();
        // the player about to move owns 8 Scouts and 2 Vipers
        assert_eq!(row[width + index(Card::Scout)], 8.0);
        assert_eq!(row[width + index(Card::Viper)], 2.0);
        assert_eq!(row[width + CARDS + bases().len()], 50.0);
        assert_eq!(row[2 * width], 1.0);
        assert_eq!(row[2 * width + 1..].iter().sum::<f64>(), 5.0);
    }
}
