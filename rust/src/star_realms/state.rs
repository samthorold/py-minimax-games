//! The game state, as `state.GameState` holds it for the first slice.
//!
//! Every pile is a vector in the Python tuple's order (the top of a deck is
//! `deck[0]`), so the differential tests compare piles exactly, and a shuffle
//! of the same discard pile gives the same deck. The Python fields the first
//! slice never sets (pending discards, scraps, choices, ...) are left out;
//! the tests check they stay at their defaults in Python.

use super::card::Card;

pub const STARTING_AUTHORITY: i32 = 50;
pub const HAND_SIZE: usize = 5;
pub const FIRST_HAND_SIZE: usize = 3;
pub const TRADE_ROW_SIZE: usize = 5;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PlayerState {
    pub authority: i32,
    pub deck: Vec<Card>,
    pub hand: Vec<Card>,
    pub in_play: Vec<Card>,
    pub discard: Vec<Card>,
    pub bases: Vec<Card>,
}

impl PlayerState {
    pub fn new(deck: Vec<Card>) -> Self {
        PlayerState {
            authority: STARTING_AUTHORITY,
            deck,
            hand: Vec::new(),
            in_play: Vec::new(),
            discard: Vec::new(),
            bases: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GameState {
    pub players: [PlayerState; 2],
    pub rng_seed: u64,
    pub trade_deck: Vec<Card>,
    pub trade_row: Vec<Card>,
    pub explorers: u32,
    pub scrap_heap: Vec<Card>,
    pub current: usize,
    pub turn: u32,
    pub trade: i32,
    pub combat: i32,
    pub allies_used: Vec<Card>,
    pub bases_used: Vec<Card>,
    pub factions_played: [u32; 4],
}
