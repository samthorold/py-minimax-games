//! Observation views, a port of `view.py`: what one player knows about a
//! game (`observe`), and a full state consistent with it (`sample_world`).
//!
//! Hidden piles are multisets, held sorted by card name as in Python. Given a
//! `Random` seeded as Python's `random.Random`, `sample_world` draws exactly
//! what Python's does, so the differential tests compare sampled worlds
//! exactly.

use super::card::Card;
use super::rng::Random;
use super::state::{GameState, PlayerState};

/// One player as the observer sees them (`view.PlayerView`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PlayerView {
    pub authority: i32,
    pub hand_size: usize,
    pub deck_size: usize,
    pub hand: Vec<Card>,
    pub unseen: Vec<Card>,
    pub in_play: Vec<Card>,
    pub discard: Vec<Card>,
    pub bases: Vec<Card>,
    pub to_discard: u32,
    pub deck_top: Vec<Card>,
}

/// What `observer` knows about the game (`view.View`): the turn state is
/// all public, as `GameState` holds it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct View {
    pub observer: usize,
    pub players: [PlayerView; 2],
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
    pub may_scrap: u32,
    pub must_scrap: u32,
    pub may_destroy: u32,
    pub may_scrap_row: u32,
    pub choosing: Vec<Card>,
    pub recycling: u32,
    pub factions_played: [u32; 4],
    pub may_on_top: u32,
    pub free_ships: u32,
    pub scrap_to_draw: u32,
    pub copying: u32,
}

/// `cards` in a canonical order, forgetting their own (`view._multiset`).
fn multiset(cards: impl IntoIterator<Item = Card>) -> Vec<Card> {
    let mut cards: Vec<Card> = cards.into_iter().collect();
    cards.sort_by_key(|card| card.name());
    cards
}

fn player_view(player: &PlayerState, hand_seen: bool) -> PlayerView {
    let (deck_top, deck) = player.deck.split_at(player.known_top);
    let (hand, unseen) = if hand_seen {
        (
            multiset(player.hand.iter().copied()),
            multiset(deck.iter().copied()),
        )
    } else {
        (
            Vec::new(),
            multiset(player.hand.iter().chain(deck).copied()),
        )
    };
    PlayerView {
        authority: player.authority,
        hand_size: player.hand.len(),
        deck_size: player.deck.len(),
        hand,
        unseen,
        in_play: player.in_play.clone(),
        discard: player.discard.clone(),
        bases: player.bases.clone(),
        to_discard: player.to_discard,
        deck_top: deck_top.to_vec(),
    }
}

/// What `observer` (0 or 1) knows about `state` (`view.observe`).
pub fn observe(state: &GameState, observer: usize) -> View {
    View {
        observer,
        players: [
            player_view(&state.players[0], observer == 0),
            player_view(&state.players[1], observer == 1),
        ],
        trade_deck: multiset(state.trade_deck.iter().copied()),
        trade_row: state.trade_row.clone(),
        explorers: state.explorers,
        scrap_heap: state.scrap_heap.clone(),
        current: state.current,
        turn: state.turn,
        trade: state.trade,
        combat: state.combat,
        allies_used: state.allies_used.clone(),
        bases_used: state.bases_used.clone(),
        may_scrap: state.may_scrap,
        must_scrap: state.must_scrap,
        may_destroy: state.may_destroy,
        may_scrap_row: state.may_scrap_row,
        choosing: state.choosing.clone(),
        recycling: state.recycling,
        factions_played: state.factions_played,
        may_on_top: state.may_on_top,
        free_ships: state.free_ships,
        scrap_to_draw: state.scrap_to_draw,
        copying: state.copying,
    }
}

fn sample_player(player: &PlayerView, rng: &mut Random) -> PlayerState {
    let unseen = rng.sample(&player.unseen);
    let dealt = player.hand_size - player.hand.len();
    let mut hand = player.hand.clone();
    hand.extend_from_slice(&unseen[..dealt]);
    let mut deck = player.deck_top.clone();
    deck.extend_from_slice(&unseen[dealt..]);
    PlayerState {
        authority: player.authority,
        deck,
        hand,
        in_play: player.in_play.clone(),
        discard: player.discard.clone(),
        bases: player.bases.clone(),
        to_discard: player.to_discard,
        known_top: player.deck_top.len(),
    }
}

/// A full state consistent with `view`, its hidden information drawn from
/// `rng` (`view.sample_world`): `observe(&sample_world(v, rng), v.observer)
/// == v`. The draws are Python's, in its order: each player's unseen cards,
/// the seed for future shuffles, then the Trade Deck.
pub fn sample_world(view: &View, rng: &mut Random) -> GameState {
    let first = sample_player(&view.players[0], rng);
    let second = sample_player(&view.players[1], rng);
    let rng_seed = rng.next_u64();
    let trade_deck = rng.sample(&view.trade_deck);
    GameState {
        players: [first, second],
        rng_seed,
        trade_deck,
        trade_row: view.trade_row.clone(),
        explorers: view.explorers,
        scrap_heap: view.scrap_heap.clone(),
        current: view.current,
        turn: view.turn,
        trade: view.trade,
        combat: view.combat,
        allies_used: view.allies_used.clone(),
        bases_used: view.bases_used.clone(),
        may_scrap: view.may_scrap,
        must_scrap: view.must_scrap,
        may_destroy: view.may_destroy,
        may_scrap_row: view.may_scrap_row,
        choosing: view.choosing.clone(),
        recycling: view.recycling,
        factions_played: view.factions_played,
        may_on_top: view.may_on_top,
        free_ships: view.free_ships,
        scrap_to_draw: view.scrap_to_draw,
        copying: view.copying,
    }
}

#[cfg(test)]
mod tests {
    use super::super::bots::greedy_move;
    use super::super::rules::{advance_legal, auto_play, legal_actions, new_game, winner};
    use super::*;

    #[test]
    fn a_sampled_world_has_the_view_it_was_sampled_from() {
        let mut rng = Random::new(7);
        for seed in 0..10 {
            let mut state = new_game(seed);
            auto_play(&mut state);
            while winner(&state).is_none() && state.turn <= 60 {
                for observer in 0..2 {
                    let view = observe(&state, observer);
                    let world = sample_world(&view, &mut rng);
                    assert_eq!(observe(&world, observer), view, "seed {seed}");
                }
                let action = greedy_move(state.must_scrap, &legal_actions(&state));
                state = advance_legal(&state, action);
            }
        }
    }

    #[test]
    fn the_opponent_s_hand_is_hidden() {
        let mut state = new_game(0);
        auto_play(&mut state);
        let view = observe(&state, 0);
        assert!(view.players[1].hand.is_empty());
        assert_eq!(view.players[1].hand_size, 5);
        assert_eq!(view.players[1].unseen.len(), 10);
    }
}
