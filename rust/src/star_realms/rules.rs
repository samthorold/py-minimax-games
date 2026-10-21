//! The rules for the first slice: dealing, legal actions, applying them, and
//! automatic play, a port of `rules.py` restricted to the first slice's
//! actions and effects. Python's `rules` is the reference: the differential
//! tests (`tests/star_realms/test_native.py`) compare states and legal actions
//! after every action.

use std::fmt;

use super::card::{Card, CardType, EXPLORER_PILE, Effect, STARTING_DECK, trade_deck};
use super::rng::Random;
use super::state::{FIRST_HAND_SIZE, GameState, HAND_SIZE, PlayerState, TRADE_ROW_SIZE};

/// What a player can do: the first slice's variants of `action.Action`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    PlayCard(Card),
    UseAlly(Card),
    UseBase(Card),
    Scrap(Card),
    Acquire(Card),
    AttackPlayer,
    AttackBase(Card),
    EndTurn,
}

impl Action {
    /// The Python variant's class name.
    pub fn kind(self) -> &'static str {
        match self {
            Action::PlayCard(_) => "PlayCard",
            Action::UseAlly(_) => "UseAlly",
            Action::UseBase(_) => "UseBase",
            Action::Scrap(_) => "Scrap",
            Action::Acquire(_) => "Acquire",
            Action::AttackPlayer => "AttackPlayer",
            Action::AttackBase(_) => "AttackBase",
            Action::EndTurn => "EndTurn",
        }
    }

    /// The card the action names, if any.
    pub fn card(self) -> Option<Card> {
        match self {
            Action::PlayCard(card)
            | Action::UseAlly(card)
            | Action::UseBase(card)
            | Action::Scrap(card)
            | Action::Acquire(card)
            | Action::AttackBase(card) => Some(card),
            Action::AttackPlayer | Action::EndTurn => None,
        }
    }

    /// The action of Python variant `kind`, naming `card`, if there is one.
    pub fn from_parts(kind: &str, card: Option<Card>) -> Option<Action> {
        match (kind, card) {
            ("PlayCard", Some(card)) => Some(Action::PlayCard(card)),
            ("UseAlly", Some(card)) => Some(Action::UseAlly(card)),
            ("UseBase", Some(card)) => Some(Action::UseBase(card)),
            ("Scrap", Some(card)) => Some(Action::Scrap(card)),
            ("Acquire", Some(card)) => Some(Action::Acquire(card)),
            ("AttackPlayer", None) => Some(Action::AttackPlayer),
            ("AttackBase", Some(card)) => Some(Action::AttackBase(card)),
            ("EndTurn", None) => Some(Action::EndTurn),
            _ => None,
        }
    }
}

/// An action that isn't legal in the state it was applied to.
#[derive(Debug, PartialEq, Eq)]
pub struct IllegalAction(pub Action);

impl fmt::Display for IllegalAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} is not legal now", self.0)
    }
}

impl std::error::Error for IllegalAction {}

/// `cards` shuffled from `seed`, and the seed for the next shuffle
/// (`rules._shuffle`).
fn shuffle(cards: &[Card], seed: u64) -> (Vec<Card>, u64) {
    let mut rng = Random::new(seed);
    let shuffled = rng.sample(cards);
    (shuffled, rng.next_u64())
}

/// Draw `n` cards, shuffling the discard pile into the deck if it runs out
/// (`rules._draw`). Returns the next seed.
fn draw(player: &mut PlayerState, n: usize, mut seed: u64) -> u64 {
    for _ in 0..n {
        if player.deck.is_empty() {
            if player.discard.is_empty() {
                break;
            }
            let (deck, next) = shuffle(&player.discard, seed);
            player.deck = deck;
            player.discard.clear();
            seed = next;
        }
        let card = player.deck.remove(0);
        player.hand.push(card);
    }
    seed
}

/// Deal a first-slice game from `seed` (`rules.new_game` with the first
/// slice's Trade Deck), before automatic play.
pub fn new_game(seed: u64) -> GameState {
    let (shuffled, mut seed) = shuffle(&trade_deck(), seed);
    let trade_row = shuffled[..TRADE_ROW_SIZE].to_vec();
    let rest = shuffled[TRADE_ROW_SIZE..].to_vec();
    let mut deal = |hand_size: usize| {
        let (deck, next) = shuffle(&STARTING_DECK, seed);
        let mut player = PlayerState::new(deck);
        seed = draw(&mut player, hand_size, next);
        player
    };
    let first = deal(FIRST_HAND_SIZE);
    let second = deal(HAND_SIZE);
    GameState {
        players: [first, second],
        rng_seed: seed,
        trade_deck: rest,
        trade_row,
        explorers: EXPLORER_PILE,
        scrap_heap: Vec::new(),
        current: 0,
        turn: 1,
        trade: 0,
        combat: 0,
        allies_used: Vec::new(),
        bases_used: Vec::new(),
        factions_played: [0; 4],
    }
}

/// The index of the player who has won, if any (`rules.winner`).
pub fn winner(state: &GameState) -> Option<usize> {
    state
        .players
        .iter()
        .position(|player| player.authority <= 0)
        .map(|i| 1 - i)
}

/// Remove one copy of `card`, which `cards` holds.
fn remove_one(cards: &mut Vec<Card>, card: Card) {
    let i = cards
        .iter()
        .position(|&c| c == card)
        .expect("the card is in the pile");
    cards.remove(i);
}

fn count(cards: &[Card], card: Card) -> usize {
    cards.iter().filter(|&&c| c == card).count()
}

/// The distinct cards among `cards`, in order of first appearance, as
/// Python's `dict.fromkeys(cards)`.
fn distinct(cards: impl IntoIterator<Item = Card>) -> Vec<Card> {
    let mut seen = Vec::new();
    for card in cards {
        if !seen.contains(&card) {
            seen.push(card);
        }
    }
    seen
}

/// The current player's cards in play: ships, then bases.
fn in_play(player: &PlayerState) -> impl Iterator<Item = Card> + '_ {
    player.in_play.iter().chain(&player.bases).copied()
}

/// Whether another card of `card`'s faction is in play (`rules._allied`;
/// the first slice has no card that allies other factions).
fn allied(card: Card, player: &PlayerState) -> bool {
    let Some(faction) = card.data().faction else {
        return false;
    };
    in_play(player)
        .filter(|c| c.data().faction == Some(faction))
        .count()
        >= 2
}

/// Whether `card` has a primary ability to use as a base
/// (`rules._primary_usable`): the first slice has no always-on effects.
fn primary_usable(card: Card) -> bool {
    !card.data().primary.is_empty()
}

fn pure_gain(effects: &[Effect]) -> bool {
    effects.iter().all(|e| e.pure_gain())
}

/// The legal `PlayCard`, `UseAlly` and `UseBase` actions, in
/// `legal_actions` order (`rules._maybe_automatic`).
fn maybe_automatic(state: &GameState, actions: &mut Vec<Action>) {
    let player = &state.players[state.current];
    actions.extend(
        distinct(player.hand.iter().copied())
            .into_iter()
            .map(Action::PlayCard),
    );
    let playing: Vec<Card> = in_play(player).collect();
    for card in distinct(playing.iter().copied()) {
        if !card.data().ally.is_empty()
            && allied(card, player)
            && count(&playing, card) > count(&state.allies_used, card)
        {
            actions.push(Action::UseAlly(card));
        }
    }
    for card in distinct(player.bases.iter().copied()) {
        if primary_usable(card) && count(&player.bases, card) > count(&state.bases_used, card) {
            actions.push(Action::UseBase(card));
        }
    }
}

/// The actions open to the current player (`rules.legal_actions`): empty
/// once the game is over.
pub fn legal_actions(state: &GameState) -> Vec<Action> {
    let mut actions = Vec::new();
    if winner(state).is_some() {
        return actions;
    }
    maybe_automatic(state, &mut actions);
    let player = &state.players[state.current];
    for card in distinct(in_play(player)) {
        if !card.data().scrap.is_empty() {
            actions.push(Action::Scrap(card));
        }
    }
    let explorer = (state.explorers > 0).then_some(Card::Explorer);
    let market = state.trade_row.iter().copied().chain(explorer);
    for card in distinct(market) {
        if card.data().cost <= state.trade {
            actions.push(Action::Acquire(card));
        }
    }
    let targets = &state.players[1 - state.current].bases;
    let outposts: Vec<Card> = targets.iter().copied().filter(|c| c.is_outpost()).collect();
    let attackable = if outposts.is_empty() {
        targets
    } else {
        &outposts
    };
    for card in distinct(attackable.iter().copied()) {
        if card.data().defense <= state.combat {
            actions.push(Action::AttackBase(card));
        }
    }
    if state.combat > 0 && outposts.is_empty() {
        actions.push(Action::AttackPlayer);
    }
    actions.push(Action::EndTurn);
    actions
}

/// Whether `action`, legal in `state`, is applied automatically
/// (`rules.automatic`). In the first slice nothing scraps from hand and no
/// Fleet HQ boosts ships, so a ship whose primary effects are pure gain is
/// always played automatically.
pub fn automatic(action: Action) -> bool {
    match action {
        Action::PlayCard(card) => card.is_ship() && pure_gain(card.data().primary),
        Action::UseAlly(card) => pure_gain(card.data().ally),
        Action::UseBase(card) => pure_gain(card.data().primary),
        Action::Scrap(_)
        | Action::Acquire(_)
        | Action::AttackPlayer
        | Action::AttackBase(_)
        | Action::EndTurn => false,
    }
}

/// The first automatic action in `legal_actions(state)`, if any
/// (`rules._next_automatic`).
fn next_automatic(state: &GameState) -> Option<Action> {
    if winner(state).is_some() {
        return None;
    }
    let mut actions = Vec::new();
    maybe_automatic(state, &mut actions);
    actions.into_iter().find(|&a| automatic(a))
}

/// Apply automatic actions, in `legal_actions` order, until none is legal
/// (`rules.auto_play`).
pub fn auto_play(state: &mut GameState) {
    while let Some(action) = next_automatic(state) {
        apply_legal(state, action);
    }
}

/// `state` after the decision `action` and the automatic play it opens up
/// (`rules.advance`), or an error if `action` isn't legal.
pub fn advance(state: &GameState, action: Action) -> Result<GameState, IllegalAction> {
    if !legal_actions(state).contains(&action) {
        return Err(IllegalAction(action));
    }
    Ok(advance_legal(state, action))
}

/// `advance` for an action from `legal_actions(state)`, without the check
/// (`rules.advance_legal`).
pub fn advance_legal(state: &GameState, action: Action) -> GameState {
    let mut next = state.clone();
    apply_legal(&mut next, action);
    auto_play(&mut next);
    next
}

fn resolve(state: &mut GameState, effect: Effect) {
    match effect {
        Effect::Trade(amount) => state.trade += amount,
        Effect::Combat(amount) => state.combat += amount,
        Effect::Authority(amount) => state.players[state.current].authority += amount,
        Effect::Draw(cards) => {
            state.rng_seed = draw(&mut state.players[state.current], cards, state.rng_seed);
        }
    }
}

fn resolve_all(state: &mut GameState, effects: &[Effect]) {
    for &effect in effects {
        resolve(state, effect);
    }
}

fn played(state: &mut GameState, card: Card) {
    if let Some(faction) = card.data().faction {
        state.factions_played[faction.index()] += 1;
    }
}

/// Take `card` from the Trade Row, refilling its place from the top of the
/// Trade Deck if any is left (`rules._take_from_row`), or, for an Explorer,
/// from the Explorer pile.
fn take_from_market(state: &mut GameState, card: Card) {
    if card == Card::Explorer {
        state.explorers -= 1;
        return;
    }
    let i = state
        .trade_row
        .iter()
        .position(|&c| c == card)
        .expect("the card is in the Trade Row");
    if state.trade_deck.is_empty() {
        state.trade_row.remove(i);
    } else {
        state.trade_row[i] = state.trade_deck.remove(0);
    }
}

/// Apply `action`, already known to be legal, in place (`rules._apply`).
pub fn apply_legal(state: &mut GameState, action: Action) {
    let me = state.current;
    match action {
        Action::PlayCard(card) => {
            let player = &mut state.players[me];
            remove_one(&mut player.hand, card);
            match card.data().card_type {
                CardType::Ship => player.in_play.push(card),
                CardType::Base | CardType::Outpost => {
                    // Playing a base uses its primary ability for this turn.
                    player.bases.push(card);
                    state.bases_used.push(card);
                }
            }
            played(state, card);
            resolve_all(state, card.data().primary);
        }
        Action::UseAlly(card) => {
            state.allies_used.push(card);
            resolve_all(state, card.data().ally);
        }
        Action::UseBase(card) => {
            state.bases_used.push(card);
            resolve_all(state, card.data().primary);
        }
        Action::Scrap(card) => {
            let player = &mut state.players[me];
            if card.is_ship() {
                remove_one(&mut player.in_play, card);
            } else {
                remove_one(&mut player.bases, card);
            }
            // Of identical copies, scrap one whose abilities were used, so an
            // unused one keeps its abilities.
            if state.bases_used.contains(&card) {
                remove_one(&mut state.bases_used, card);
            }
            if state.allies_used.contains(&card) {
                remove_one(&mut state.allies_used, card);
            }
            if card == Card::Explorer {
                state.explorers += 1;
            } else {
                state.scrap_heap.push(card);
            }
            resolve_all(state, card.data().scrap);
        }
        Action::Acquire(card) => {
            state.players[me].discard.push(card);
            state.trade -= card.data().cost;
            take_from_market(state, card);
        }
        Action::AttackPlayer => {
            state.players[1 - me].authority -= state.combat;
            state.combat = 0;
        }
        Action::AttackBase(card) => {
            let opponent = &mut state.players[1 - me];
            remove_one(&mut opponent.bases, card);
            opponent.discard.push(card);
            state.combat -= card.data().defense;
        }
        Action::EndTurn => {
            let player = &mut state.players[me];
            let in_play = std::mem::take(&mut player.in_play);
            let hand = std::mem::take(&mut player.hand);
            player.discard.extend(in_play);
            player.discard.extend(hand);
            state.rng_seed = draw(player, HAND_SIZE, state.rng_seed);
            state.current = 1 - me;
            state.turn += 1;
            state.trade = 0;
            state.combat = 0;
            state.allies_used.clear();
            state.bases_used.clear();
            state.factions_played = [0; 4];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dealt(seed: u64) -> GameState {
        let mut state = new_game(seed);
        auto_play(&mut state);
        state
    }

    #[test]
    fn a_new_game_deals_3_and_5_cards_and_a_trade_row() {
        let state = new_game(0);
        assert_eq!(state.players[0].hand.len(), 3);
        assert_eq!(state.players[1].hand.len(), 5);
        assert_eq!(state.players[0].deck.len(), 7);
        assert_eq!(state.trade_row.len(), 5);
        assert_eq!(state.trade_deck.len(), 25);
    }

    #[test]
    fn automatic_play_plays_the_starting_hand() {
        let state = dealt(0);
        let player = &state.players[0];
        assert!(player.hand.is_empty());
        assert_eq!(player.in_play.len(), 3);
        let scouts = count(&player.in_play, Card::Scout) as i32;
        assert_eq!(state.trade, scouts);
        assert_eq!(state.combat, 3 - scouts);
    }

    #[test]
    fn an_illegal_action_is_refused() {
        let state = dealt(0);
        let action = Action::Acquire(Card::Mothership);
        assert_eq!(advance(&state, action), Err(IllegalAction(action)));
    }

    #[test]
    fn end_turn_passes_play_and_draws_five() {
        let state = advance(&dealt(0), Action::EndTurn).expect("EndTurn is legal");
        assert_eq!(state.current, 1);
        assert_eq!(state.turn, 2);
        assert_eq!(state.players[0].hand.len(), 5);
        assert_eq!(state.players[0].discard.len(), 3);
    }

    #[test]
    fn legal_actions_end_with_end_turn_and_offer_only_decisions() {
        let state = dealt(3);
        let actions = legal_actions(&state);
        assert_eq!(actions.last(), Some(&Action::EndTurn));
        assert!(actions.iter().all(|&a| !automatic(a)));
    }

    #[test]
    fn attacking_spends_all_the_combat() {
        let mut state = dealt(0);
        state.combat = 4;
        let state = advance(&state, Action::AttackPlayer).expect("legal with combat");
        assert_eq!(state.players[1].authority, 46);
        assert_eq!(state.combat, 0);
    }

    #[test]
    fn a_player_at_zero_authority_has_lost() {
        let mut state = dealt(0);
        state.players[1].authority = 0;
        assert_eq!(winner(&state), Some(0));
        assert!(legal_actions(&state).is_empty());
    }

    #[test]
    fn an_outpost_guards_the_player_and_the_other_bases() {
        let mut state = dealt(0);
        state.players[1].bases = vec![Card::BlobWheel, Card::SpaceStation];
        state.combat = 10;
        let actions = legal_actions(&state);
        assert!(actions.contains(&Action::AttackBase(Card::SpaceStation)));
        assert!(!actions.contains(&Action::AttackBase(Card::BlobWheel)));
        assert!(!actions.contains(&Action::AttackPlayer));
    }
}
