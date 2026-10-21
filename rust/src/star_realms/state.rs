//! The game state, as `state.GameState` holds it, field for field.
//!
//! Every pile is a list in the Python tuple's order (the top of a deck is
//! `deck[0]`), so the differential tests compare piles exactly, and a shuffle
//! of the same discard pile gives the same deck. The fields' meanings are
//! `state.py`'s.
//!
//! Piles are held inline, in `Pile`s of fixed capacity, each bounded by the
//! card counts (`capacity`), so a state is one block of plain data: copying
//! one allocates nothing (#156). The planner copies a state for every play it
//! tries, and with a `Vec` per pile that was 16 allocations and frees a copy.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut};

use super::card::Card;

pub const STARTING_AUTHORITY: i32 = 50;
pub const HAND_SIZE: usize = 5;
pub const FIRST_HAND_SIZE: usize = 3;
pub const TRADE_ROW_SIZE: usize = 5;

/// The card counts the piles' capacities are bounded by. A test checks them
/// against the card data (`card::STARTING_DECK`, `card::trade_deck`,
/// `card::EXPLORER_PILE`).
pub mod capacity {
    /// Cards in a starting deck.
    pub const STARTING: usize = 10;
    /// Cards in the Trade Deck.
    pub const TRADE_DECK: usize = 80;
    /// Bases and Outposts in the Trade Deck.
    pub const TRADE_DECK_BASES: usize = 25;
    /// Explorers.
    pub const EXPLORERS: usize = 10;

    /// The most cards one player can own at once: their starting deck, the
    /// whole Trade Deck and every Explorer. A deck, hand or discard pile
    /// holds only cards its player owns.
    pub const OWNED: usize = STARTING + TRADE_DECK + EXPLORERS;
    /// The most ships one player can own (ships in play are theirs).
    pub const SHIPS: usize = OWNED - TRADE_DECK_BASES;
    /// The most bases one player can own (bases in play are theirs).
    pub const BASES: usize = TRADE_DECK_BASES;
    /// The scrap heap: every card but the Explorers, which go back to their
    /// pile when scrapped.
    pub const SCRAP_HEAP: usize = 2 * STARTING + TRADE_DECK;
    /// Allies used this turn, and the "or" choices pending: at most one per
    /// card the current player has in play (allies used are dropped as their
    /// card is scrapped; a choice is pending only once its card is played).
    pub const IN_PLAY: usize = OWNED;
}

/// A pile of at most `N` cards, in order, held inline: a list with a fixed
/// capacity, `Copy`, so copying a state is one copy of plain data. It reads
/// as a slice (`Deref`), and only the first `len` places count: equality,
/// hashing and `Debug` see `self[..]`, never the places left over. Pushing
/// past the capacity panics; the capacities are bounded by the card counts
/// (`capacity`), so it never happens in a game.
#[derive(Clone, Copy)]
pub struct Pile<const N: usize> {
    len: u8,
    cards: [Card; N],
}

impl<const N: usize> Pile<N> {
    /// The capacity, which fits `len`.
    pub const CAPACITY: usize = {
        assert!(N <= u8::MAX as usize, "a pile's length is a byte");
        N
    };

    pub const fn new() -> Self {
        let _ = Self::CAPACITY;
        Pile {
            len: 0,
            cards: [Card::Scout; N],
        }
    }

    /// `cards` as a pile, or `None` if there are more than `N`.
    pub fn from_slice(cards: &[Card]) -> Option<Self> {
        if cards.len() > Self::CAPACITY {
            return None;
        }
        let mut pile = Pile::new();
        pile.cards[..cards.len()].copy_from_slice(cards);
        pile.len = cards.len() as u8;
        Some(pile)
    }

    #[track_caller]
    fn grow(&mut self, by: usize) -> usize {
        let len = self.len();
        assert!(
            len + by <= Self::CAPACITY,
            "a pile of capacity {N} can't hold {} cards",
            len + by
        );
        self.len = (len + by) as u8;
        len
    }

    #[track_caller]
    pub fn push(&mut self, card: Card) {
        let at = self.grow(1);
        self.cards[at] = card;
    }

    /// Put `card` at `index`, moving the cards from there along one.
    #[track_caller]
    pub fn insert(&mut self, index: usize, card: Card) {
        let len = self.len();
        assert!(index <= len, "insertion index {index} past length {len}");
        self.grow(1);
        self.cards.copy_within(index..len, index + 1);
        self.cards[index] = card;
    }

    /// Take the card at `index`, moving the cards after it back one.
    #[track_caller]
    pub fn remove(&mut self, index: usize) -> Card {
        let len = self.len();
        assert!(index < len, "removal index {index} past length {len}");
        let card = self.cards[index];
        self.cards.copy_within(index + 1..len, index);
        self.len -= 1;
        card
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    #[track_caller]
    pub fn extend_from_slice(&mut self, cards: &[Card]) {
        let at = self.grow(cards.len());
        self.cards[at..at + cards.len()].copy_from_slice(cards);
    }
}

impl<const N: usize> Default for Pile<N> {
    fn default() -> Self {
        Pile::new()
    }
}

impl<const N: usize> Deref for Pile<N> {
    type Target = [Card];

    fn deref(&self) -> &[Card] {
        &self.cards[..usize::from(self.len)]
    }
}

impl<const N: usize> DerefMut for Pile<N> {
    fn deref_mut(&mut self) -> &mut [Card] {
        &mut self.cards[..usize::from(self.len)]
    }
}

impl<const N: usize> PartialEq for Pile<N> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<const N: usize> Eq for Pile<N> {}

impl<const N: usize> Hash for Pile<N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl<const N: usize> fmt::Debug for Pile<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<const N: usize> Extend<Card> for Pile<N> {
    #[track_caller]
    fn extend<I: IntoIterator<Item = Card>>(&mut self, cards: I) {
        for card in cards {
            self.push(card);
        }
    }
}

impl<const N: usize> FromIterator<Card> for Pile<N> {
    #[track_caller]
    fn from_iter<I: IntoIterator<Item = Card>>(cards: I) -> Self {
        let mut pile = Pile::new();
        pile.extend(cards);
        pile
    }
}

impl<'a, const N: usize> IntoIterator for &'a Pile<N> {
    type Item = &'a Card;
    type IntoIter = std::slice::Iter<'a, Card>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<const N: usize> IntoIterator for Pile<N> {
    type Item = Card;
    type IntoIter = std::iter::Take<std::array::IntoIter<Card, N>>;

    fn into_iter(self) -> Self::IntoIter {
        self.cards.into_iter().take(usize::from(self.len))
    }
}

/// A player's deck, hand or discard pile.
pub type Owned = Pile<{ capacity::OWNED }>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PlayerState {
    pub authority: i32,
    pub deck: Owned,
    pub hand: Owned,
    pub in_play: Pile<{ capacity::SHIPS }>,
    pub discard: Owned,
    pub bases: Pile<{ capacity::BASES }>,
    /// The discards an opponent's `OpponentDiscards` effects have left the
    /// player to make at the start of their next turn.
    pub to_discard: u32,
    /// How many cards on top of the deck both players know (`deck[..known_top]`).
    pub known_top: usize,
}

impl PlayerState {
    pub fn new(deck: Owned) -> Self {
        PlayerState {
            authority: STARTING_AUTHORITY,
            deck,
            hand: Pile::new(),
            in_play: Pile::new(),
            discard: Pile::new(),
            bases: Pile::new(),
            to_discard: 0,
            known_top: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GameState {
    pub players: [PlayerState; 2],
    pub rng_seed: u64,
    pub trade_deck: Pile<{ capacity::TRADE_DECK }>,
    pub trade_row: Pile<TRADE_ROW_SIZE>,
    pub explorers: u32,
    pub scrap_heap: Pile<{ capacity::SCRAP_HEAP }>,
    pub current: usize,
    pub turn: u32,
    pub trade: i32,
    pub combat: i32,
    pub allies_used: Pile<{ capacity::IN_PLAY }>,
    /// The current player's bases whose primary ability is used this turn,
    /// at most one entry per base in play.
    pub bases_used: Pile<{ capacity::BASES }>,
    /// Optional scraps from hand or discard pile left this turn.
    pub may_scrap: u32,
    /// Scraps from hand owed right now (Machine Base).
    pub must_scrap: u32,
    /// Opponent's bases the player may still destroy this turn.
    pub may_destroy: u32,
    /// Trade Row scraps left this turn.
    pub may_scrap_row: u32,
    /// The cards whose "or" choice must be made right now, first first.
    pub choosing: Pile<{ capacity::IN_PLAY }>,
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

#[cfg(test)]
mod tests {
    use super::super::card::{EXPLORER_PILE, STARTING_DECK, trade_deck};
    use super::*;

    #[test]
    fn the_capacities_are_the_card_counts() {
        let deck = trade_deck();
        assert_eq!(capacity::STARTING, STARTING_DECK.len());
        assert_eq!(capacity::TRADE_DECK, deck.len());
        assert_eq!(
            capacity::TRADE_DECK_BASES,
            deck.iter().filter(|c| !c.is_ship()).count()
        );
        assert_eq!(capacity::EXPLORERS, EXPLORER_PILE as usize);
        assert!(STARTING_DECK.iter().all(|c| c.is_ship()));
    }

    #[test]
    fn a_state_is_a_few_kilobytes_of_plain_data() {
        // Copy, so a clone allocates nothing.
        fn plain<T: Copy>() {}
        plain::<Pile<{ capacity::OWNED }>>();
        assert!(std::mem::size_of::<GameState>() < 4096);
    }

    #[test]
    fn a_pile_is_a_list() {
        let mut pile: Pile<4> = Pile::new();
        pile.push(Card::Viper);
        pile.insert(0, Card::Scout);
        pile.extend_from_slice(&[Card::Ram, Card::Cutter]);
        assert_eq!(
            &pile[..],
            [Card::Scout, Card::Viper, Card::Ram, Card::Cutter]
        );
        assert_eq!(pile.remove(1), Card::Viper);
        assert_eq!(&pile[..], [Card::Scout, Card::Ram, Card::Cutter]);
        let other = Pile::<4>::from_slice(&[Card::Scout, Card::Ram, Card::Cutter]).unwrap();
        assert_eq!(pile, other);
        assert!(Pile::<2>::from_slice(&[Card::Scout; 3]).is_none());
        pile.clear();
        assert!(pile.is_empty());
        // Places past the length never count.
        assert_eq!(pile, Pile::new());
    }

    #[test]
    #[should_panic(expected = "can't hold")]
    fn a_pile_never_passes_its_capacity() {
        let mut pile: Pile<2> = Pile::new();
        pile.extend([Card::Scout; 3]);
    }
}
