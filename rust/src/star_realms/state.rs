//! The game state, as `state.GameState` holds it, field for field.
//!
//! Every pile is a vector in the Python tuple's order (the top of a deck is
//! `deck[0]`), so the differential tests compare piles exactly, and a shuffle
//! of the same discard pile gives the same deck. The fields' meanings are
//! `state.py`'s.

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
    /// The discards an opponent's `OpponentDiscards` effects have left the
    /// player to make at the start of their next turn.
    pub to_discard: u32,
    /// How many cards on top of the deck both players know (`deck[..known_top]`).
    pub known_top: usize,
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
            to_discard: 0,
            known_top: 0,
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
    /// Optional scraps from hand or discard pile left this turn.
    pub may_scrap: u32,
    /// Scraps from hand owed right now (Machine Base).
    pub must_scrap: u32,
    /// Opponent's bases the player may still destroy this turn.
    pub may_destroy: u32,
    /// Trade Row scraps left this turn.
    pub may_scrap_row: u32,
    /// The cards whose "or" choice must be made right now, first first.
    pub choosing: Vec<Card>,
    /// How many cards may be discarded right now to draw as many (Recycling
    /// Station).
    pub recycling: u32,
    pub factions_played: [u32; 4],
    /// Ships the player may still put on top of their deck as acquired.
    pub may_on_top: u32,
    /// Free ships owed right now (Blob Carrier).
    pub free_ships: u32,
    /// How many cards may be scrapped right now to draw as many (Brain World).
    pub scrap_to_draw: u32,
    /// Stealth Needles just played that must copy a ship right now.
    pub copying: u32,
}
