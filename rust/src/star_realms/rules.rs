//! The rules: dealing, legal actions, applying them, and automatic play, a
//! port of `rules.py` for the whole base set. Python's `rules` is the
//! reference: the differential tests (`tests/star_realms/test_native.py`)
//! compare states, legal actions and views after every action.

use std::cell::RefCell;
use std::fmt;

use rustc_hash::FxHashMap;

use super::card::{Card, CardType, EXPLORER_PILE, Effect, STARTING_DECK, pure_gain, trade_deck};
use super::rng::Random;
use super::state::{
    FIRST_HAND_SIZE, GameState, HAND_SIZE, Pile, PlayerState, TRADE_ROW_SIZE, capacity,
};

/// The most cards a `Recycle` or `ScrapAndDraw` names: Recycling Station
/// discards and Brain World scraps up to two.
pub const MAX_FEW: usize = 2;

/// Up to `MAX_FEW` cards, in order: the cards a `Recycle` or a pile of a
/// `ScrapAndDraw` names (a tuple in Python). Inline, so `Action` stays
/// `Copy`; the unused places hold Scouts, so equal lists compare equal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Few {
    len: u8,
    cards: [Card; MAX_FEW],
}

impl Few {
    pub const EMPTY: Few = Few {
        len: 0,
        cards: [Card::Scout; MAX_FEW],
    };

    /// `cards`, at most `MAX_FEW` of them.
    pub fn new(cards: &[Card]) -> Option<Few> {
        if cards.len() > MAX_FEW {
            return None;
        }
        let mut few = Few::EMPTY;
        few.cards[..cards.len()].copy_from_slice(cards);
        few.len = cards.len() as u8;
        Some(few)
    }

    pub fn as_slice(&self) -> &[Card] {
        &self.cards[..usize::from(self.len)]
    }

    pub fn len(&self) -> usize {
        usize::from(self.len)
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// What a player can do: `action.Action`, every variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    PlayCard(Card),
    UseAlly(Card),
    UseBase(Card),
    Scrap(Card),
    Acquire(Card),
    AcquireOnTop(Card),
    AcquireFreeShip(Card),
    AttackPlayer,
    AttackBase(Card),
    DestroyTarget(Card),
    Discard(Card),
    ScrapHandCard(Card),
    ScrapDiscardCard(Card),
    ScrapTradeRowCard(Card),
    /// The card's "or" choice, and the option: its index among the card's
    /// options (`Card::options`), where Python names its effects.
    Choose(Card, u8),
    Copy(Card),
    Recycle(Few),
    /// The cards scrapped from hand, then from the discard pile.
    ScrapAndDraw(Few, Few),
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
            Action::AcquireOnTop(_) => "AcquireOnTop",
            Action::AcquireFreeShip(_) => "AcquireFreeShip",
            Action::AttackPlayer => "AttackPlayer",
            Action::AttackBase(_) => "AttackBase",
            Action::DestroyTarget(_) => "DestroyTarget",
            Action::Discard(_) => "Discard",
            Action::ScrapHandCard(_) => "ScrapHandCard",
            Action::ScrapDiscardCard(_) => "ScrapDiscardCard",
            Action::ScrapTradeRowCard(_) => "ScrapTradeRowCard",
            Action::Choose(..) => "Choose",
            Action::Copy(_) => "Copy",
            Action::Recycle(_) => "Recycle",
            Action::ScrapAndDraw(..) => "ScrapAndDraw",
            Action::EndTurn => "EndTurn",
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
fn shuffle<const N: usize>(cards: &[Card], seed: u64) -> (Pile<N>, u64) {
    SHUFFLES.with_borrow_mut(|cache| {
        if cache.len() >= SHUFFLES_KEPT {
            cache.clear();
        }
        let (order, next) = cache.entry((seed, cards.len())).or_insert_with(|| {
            let mut rng = Random::new(seed);
            let positions: Vec<usize> = (0..cards.len()).collect();
            let order = rng.sample(&positions);
            (order, rng.next_u64())
        });
        (order.iter().map(|&i| cards[i]).collect(), *next)
    })
}

/// A shuffle: the positions it takes the cards from, and the next seed.
type Shuffle = (Vec<usize>, u64);

/// How many shuffles `SHUFFLES` keeps before it starts again.
const SHUFFLES_KEPT: usize = 256;

thread_local! {
    /// Shuffles made, by seed and pile size, as the positions they take the
    /// cards from and the next seed: a shuffle moves cards by positions drawn
    /// from the seed alone, never by what the cards are. The planner
    /// reshuffles a world's discard pile, from the same seed, at most ends of
    /// a turn, and seeding the Mersenne Twister was a fifth of its time.
    static SHUFFLES: RefCell<FxHashMap<(u64, usize), Shuffle>> =
        RefCell::new(FxHashMap::default());
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
    // The known top cards are drawn first.
    player.known_top = player.known_top.saturating_sub(n);
    seed
}

/// Deal a game from `seed` (`rules.new_game` with the whole Trade Deck),
/// before automatic play.
pub fn new_game(seed: u64) -> GameState {
    let (shuffled, mut seed) = shuffle::<{ capacity::TRADE_DECK }>(&trade_deck(), seed);
    let trade_row = Pile::from_slice(&shuffled[..TRADE_ROW_SIZE]).expect("a row of cards");
    let rest = Pile::from_slice(&shuffled[TRADE_ROW_SIZE..]).expect("the Trade Deck");
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
        scrap_heap: Pile::new(),
        current: 0,
        turn: 1,
        trade: 0,
        combat: 0,
        allies_used: Pile::new(),
        bases_used: Pile::new(),
        may_scrap: 0,
        must_scrap: 0,
        may_destroy: 0,
        may_scrap_row: 0,
        choosing: Pile::new(),
        recycling: 0,
        factions_played: [0; 4],
        may_on_top: 0,
        free_ships: 0,
        scrap_to_draw: 0,
        copying: 0,
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
fn remove_one<const N: usize>(cards: &mut Pile<N>, card: Card) {
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

/// Whether another card that is an ally for `card`'s faction is in play
/// (`rules._allied`), `card` among the cards in play.
fn allied(card: Card, player: &PlayerState) -> bool {
    let Some(faction) = card.data().faction else {
        return false;
    };
    in_play(player).filter(|c| c.ally_for(faction)).count() >= 2
}

/// Whether `card` has a primary ability to use as a base
/// (`rules._primary_usable`): not none, nor one that is always on.
fn primary_usable(card: Card) -> bool {
    let primary = card.data().primary;
    !primary.is_empty() && !primary.iter().all(|e| e.always_on())
}

/// The distinct ships among `in_play` that Stealth Needle may copy
/// (`rules._copy_targets`).
fn copy_targets(in_play: &[Card]) -> Vec<Card> {
    distinct(in_play.iter().copied().filter(|&c| Card::copy_target(c)))
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
/// once the game is over, and only a forced decision's while one is pending
/// (`forced`).
pub fn legal_actions(state: &GameState) -> Vec<Action> {
    if winner(state).is_some() {
        return Vec::new();
    }
    if let Some(forced) = forced(state) {
        return forced;
    }
    let mut actions = Vec::new();
    maybe_automatic(state, &mut actions);
    let player = &state.players[state.current];
    for card in distinct(in_play(player)) {
        if !card.data().scrap.is_empty() {
            actions.push(Action::Scrap(card));
        }
    }
    if state.may_scrap > 0 {
        actions.extend(
            distinct(player.hand.iter().copied())
                .into_iter()
                .map(Action::ScrapHandCard),
        );
        actions.extend(
            distinct(player.discard.iter().copied())
                .into_iter()
                .map(Action::ScrapDiscardCard),
        );
    }
    if state.may_scrap_row > 0 {
        actions.extend(
            distinct(state.trade_row.iter().copied())
                .into_iter()
                .map(Action::ScrapTradeRowCard),
        );
    }
    let explorer = (state.explorers > 0).then_some(Card::Explorer);
    let market = state.trade_row.iter().copied().chain(explorer);
    let affordable: Vec<Card> = distinct(market)
        .into_iter()
        .filter(|card| card.data().cost <= state.trade)
        .collect();
    actions.extend(affordable.iter().map(|&card| Action::Acquire(card)));
    if state.may_on_top > 0 {
        actions.extend(
            affordable
                .iter()
                .filter(|card| card.is_ship())
                .map(|&card| Action::AcquireOnTop(card)),
        );
    }
    let targets: &[Card] = &state.players[1 - state.current].bases;
    if state.may_destroy > 0 {
        actions.extend(
            distinct(targets.iter().copied())
                .into_iter()
                .map(Action::DestroyTarget),
        );
    }
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

/// The actions of a decision the current player must make before any other,
/// if one is pending (`rules._forced`), in its order of precedence: pending
/// discards, an owed scrap from hand, recycling, Brain World's scraps,
/// Stealth Needle's copy, a free ship, and an "or" choice.
fn forced(state: &GameState) -> Option<Vec<Action>> {
    let player = &state.players[state.current];
    let hand = || distinct(player.hand.iter().copied()).into_iter();
    if player.to_discard > 0 {
        return Some(hand().map(Action::Discard).collect());
    }
    if state.must_scrap > 0 {
        return Some(hand().map(Action::ScrapHandCard).collect());
    }
    if state.recycling > 0 {
        let sets = discard_sets(&player.hand, state.recycling as usize);
        return Some(sets.into_iter().map(Action::Recycle).collect());
    }
    if state.scrap_to_draw > 0 {
        let sets = scrap_sets(&player.hand, &player.discard, state.scrap_to_draw as usize);
        return Some(
            sets.into_iter()
                .map(|(hand, discard)| Action::ScrapAndDraw(hand, discard))
                .collect(),
        );
    }
    if state.copying > 0 {
        return Some(
            copy_targets(&player.in_play)
                .into_iter()
                .map(Action::Copy)
                .collect(),
        );
    }
    if state.free_ships > 0 {
        let ships = state.trade_row.iter().copied().filter(|c| c.is_ship());
        return Some(
            distinct(ships)
                .into_iter()
                .map(Action::AcquireFreeShip)
                .collect(),
        );
    }
    if let Some(&card) = state.choosing.first() {
        let options = card.options().len();
        return Some(
            (0..options)
                .map(|i| Action::Choose(card, i as u8))
                .collect(),
        );
    }
    None
}

/// Each combination of `size` of `cards`, in `itertools.combinations` order.
fn combinations(cards: &[Card], size: usize) -> Vec<Few> {
    fn extend(
        cards: &[Card],
        size: usize,
        start: usize,
        chosen: &mut Vec<Card>,
        out: &mut Vec<Few>,
    ) {
        if chosen.len() == size {
            out.push(Few::new(chosen).expect("at most MAX_FEW cards"));
            return;
        }
        for i in start..cards.len() {
            chosen.push(cards[i]);
            extend(cards, size, i + 1, chosen, out);
            chosen.pop();
        }
    }
    let mut out = Vec::new();
    if size <= cards.len() {
        extend(cards, size, 0, &mut Vec::new(), &mut out);
    }
    out
}

/// The distinct combinations of `size` of `cards`, sorted by name first, in
/// order of first appearance.
fn subsets(cards: &[Card], size: usize) -> Vec<Few> {
    let mut sorted = cards.to_vec();
    sorted.sort_by_key(|card| card.name());
    let mut seen: Vec<Few> = Vec::new();
    for combo in combinations(&sorted, size) {
        if !seen.contains(&combo) {
            seen.push(combo);
        }
    }
    seen
}

/// Each distinct set of one card up to `most` cards from `hand`, its cards
/// sorted by name (`rules._discard_sets`).
fn discard_sets(hand: &[Card], most: usize) -> Vec<Few> {
    (1..=most.min(hand.len()))
        .flat_map(|size| subsets(hand, size))
        .collect()
}

/// Each distinct choice of up to `most` cards from `hand` and `discard`
/// together, none included: the cards from each, sorted by name
/// (`rules._scrap_sets`).
fn scrap_sets(hand: &[Card], discard: &[Card], most: usize) -> Vec<(Few, Few)> {
    let mut sets = Vec::new();
    for total in 0..=most.min(hand.len() + discard.len()) {
        for in_hand in 0..=total {
            for from_hand in subsets(hand, in_hand) {
                for from_discard in subsets(discard, total - in_hand) {
                    sets.push((from_hand, from_discard));
                }
            }
        }
    }
    sets
}

/// Whether `card` is a ship whose primary effects are pure gain
/// (`rules._pure_gain_ship`).
fn pure_gain_ship(card: Card) -> bool {
    card.is_ship() && pure_gain(card.data().primary)
}

/// For a ship whose primary effects are pure gain but for drawing with
/// enough bases in play (Embassy Yacht), the fewest bases that draw
/// (`rules._bases_to_draw`).
fn bases_to_draw(card: Card) -> Option<usize> {
    if !card.is_ship() {
        return None;
    }
    let primary = card.data().primary;
    let mut thresholds = primary.iter().filter_map(|e| match e {
        Effect::DrawIfBases { bases, .. } => Some(*bases),
        _ => None,
    });
    let fewest = thresholds.next()?;
    let fewest = thresholds.fold(fewest, usize::min);
    let rest_pure = primary
        .iter()
        .filter(|e| !matches!(e, Effect::DrawIfBases { .. }))
        .all(|e| e.pure_gain());
    rest_pure.then_some(fewest)
}

/// Whether `card` would be pure gain but for a draw with enough bases, and
/// the current player can't have that many this turn (`rules._cannot_draw`).
fn cannot_draw(state: &GameState, card: Card) -> bool {
    let Some(bases) = bases_to_draw(card) else {
        return false;
    };
    let player = &state.players[state.current];
    let in_hand = player.hand.iter().filter(|c| !c.is_ship()).count();
    player.bases.len() + in_hand < bases
}

/// Whether a Fleet HQ is in hand (`rules._bonus_ahead`).
fn bonus_ahead(state: &GameState) -> bool {
    state.players[state.current]
        .hand
        .iter()
        .any(|card| card.ship_bonus() != 0)
}

fn primary_scraps_hand(card: Card) -> bool {
    card.data().primary.iter().any(|e| e.scraps_hand())
}

fn ally_scraps_hand(card: Card) -> bool {
    card.data().ally.iter().any(|e| e.scraps_hand())
}

/// Whether the current player could scrap `card`, in hand, from hand this
/// turn if they didn't play it (`rules._hand_scrap_ahead`).
fn hand_scrap_ahead(state: &GameState, card: Card) -> bool {
    if state.may_scrap > 0 {
        return true;
    }
    let player = &state.players[state.current];
    let scrappers = player
        .hand
        .iter()
        .filter(|&&c| primary_scraps_hand(c))
        .count();
    if scrappers > usize::from(primary_scraps_hand(card)) {
        return true;
    }
    for &base in &player.bases {
        if primary_scraps_hand(base) && count(&player.bases, base) > count(&state.bases_used, base)
        {
            return true;
        }
    }
    let playing: Vec<Card> = in_play(player).collect();
    for &other in &playing {
        if ally_scraps_hand(other) && count(&playing, other) > count(&state.allies_used, other) {
            return true;
        }
    }
    if player.hand.iter().any(|c| c.copies_ships()) {
        return player
            .in_play
            .iter()
            .any(|&ship| primary_scraps_hand(ship) || ally_scraps_hand(ship));
    }
    false
}

/// Whether `action`, legal in `state`, is applied automatically
/// (`rules.automatic`, whose docstring gives the reasoning).
pub fn automatic(state: &GameState, action: Action) -> bool {
    match action {
        Action::PlayCard(card) => {
            if !pure_gain_ship(card) && !cannot_draw(state, card) {
                return false;
            }
            !bonus_ahead(state) && !hand_scrap_ahead(state, card)
        }
        Action::UseAlly(card) => pure_gain(card.data().ally),
        Action::UseBase(card) => pure_gain(card.data().primary),
        Action::Scrap(_)
        | Action::Acquire(_)
        | Action::AcquireOnTop(_)
        | Action::AcquireFreeShip(_)
        | Action::AttackPlayer
        | Action::AttackBase(_)
        | Action::DestroyTarget(_)
        | Action::Discard(_)
        | Action::ScrapHandCard(_)
        | Action::ScrapDiscardCard(_)
        | Action::ScrapTradeRowCard(_)
        | Action::Choose(..)
        | Action::Copy(_)
        | Action::Recycle(_)
        | Action::ScrapAndDraw(..)
        | Action::EndTurn => false,
    }
}

/// The first automatic action in `legal_actions(state)`, if any
/// (`rules._next_automatic`).
fn next_automatic(state: &GameState) -> Option<Action> {
    if winner(state).is_some() || forced(state).is_some() {
        return None;
    }
    let mut actions = Vec::new();
    maybe_automatic(state, &mut actions);
    actions.into_iter().find(|&a| automatic(state, a))
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

fn draw_current(state: &mut GameState, cards: usize) {
    state.rng_seed = draw(&mut state.players[state.current], cards, state.rng_seed);
}

/// Apply `effect`, one of `card`'s (`rules._resolve`).
fn resolve(state: &mut GameState, effect: Effect, card: Card) {
    let me = state.current;
    match effect {
        Effect::Trade(amount) => state.trade += amount,
        Effect::Combat(amount) => state.combat += amount,
        Effect::Authority(amount) => state.players[me].authority += amount,
        Effect::Draw(cards) => draw_current(state, cards),
        Effect::OpponentDiscards(cards) => state.players[1 - me].to_discard += cards,
        Effect::ScrapHandOrDiscard(cards) => state.may_scrap += cards,
        Effect::DestroyBase => {
            // The opponent gains no bases during this turn, so with none now
            // there is nothing to destroy.
            if !state.players[1 - me].bases.is_empty() {
                state.may_destroy += 1;
            }
        }
        Effect::ScrapTradeRow(cards) => {
            // An empty row never refills: there is nothing to scrap.
            if !state.trade_row.is_empty() {
                state.may_scrap_row += cards;
            }
        }
        Effect::ScrapFromHand => {
            if !state.players[me].hand.is_empty() {
                state.must_scrap += 1;
            }
        }
        Effect::DrawPerFactionCard => {
            if let Some(faction) = card.data().faction {
                draw_current(state, state.factions_played[faction.index()] as usize);
            }
        }
        Effect::DiscardThenDraw(cards) => {
            if !state.players[me].hand.is_empty() {
                state.recycling = cards;
            }
        }
        Effect::NextShipOnTop => state.may_on_top += 1,
        Effect::Choice(_) => state.choosing.push(card),
        Effect::AcquireShipFreeOnTop => {
            if state.trade_row.iter().any(|c| c.is_ship()) {
                state.free_ships += 1;
            }
        }
        Effect::DrawIfBases { bases, cards } => {
            if state.players[me].bases.len() >= bases {
                draw_current(state, cards);
            }
        }
        Effect::AllyForAllFactions | Effect::ShipsGainCombat(_) => {
            // Always on while the base is in play: nothing happens when it is
            // played.
        }
        Effect::ScrapThenDraw(cards) => {
            let player = &state.players[me];
            if !player.hand.is_empty() || !player.discard.is_empty() {
                state.scrap_to_draw = cards;
            }
        }
        Effect::CopyShip => {
            if !copy_targets(&state.players[me].in_play).is_empty() {
                state.copying += 1;
            }
        }
    }
}

fn resolve_all(state: &mut GameState, effects: &[Effect], card: Card) {
    for &effect in effects {
        resolve(state, effect, card);
    }
}

fn played(state: &mut GameState, card: Card) {
    if let Some(faction) = card.data().faction {
        state.factions_played[faction.index()] += 1;
    }
}

/// `card`, taken from wherever it was, scrapped: on the scrap heap, or, for
/// an Explorer, back on the Explorer pile (`rules._to_scrap_heap`).
fn to_scrap_heap(state: &mut GameState, card: Card) {
    if card == Card::Explorer {
        state.explorers += 1;
    } else {
        state.scrap_heap.push(card);
    }
}

/// Take `card` from the Trade Row, refilling its place from the top of the
/// Trade Deck if any is left (`rules._take_from_row`).
fn take_from_row(state: &mut GameState, card: Card) {
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

/// Take `card` from the Trade Row or, for an Explorer, from the Explorer
/// pile (`rules._take_from_market`).
fn take_from_market(state: &mut GameState, card: Card) {
    if card == Card::Explorer {
        state.explorers -= 1;
    } else {
        take_from_row(state, card);
    }
}

/// Put `card`, which the current player acquires, on top of their deck,
/// where both players know it (`rules._on_top`).
fn on_top(state: &mut GameState, card: Card) {
    let player = &mut state.players[state.current];
    player.deck.insert(0, card);
    player.known_top += 1;
}

/// Destroy `card`, one of the opponent's bases: it goes to its owner's
/// discard pile, and destroy-base abilities left over once the opponent has
/// no bases are dropped (`rules._destroy`).
fn destroy(state: &mut GameState, card: Card) {
    let opponent = &mut state.players[1 - state.current];
    remove_one(&mut opponent.bases, card);
    opponent.discard.push(card);
    if opponent.bases.is_empty() {
        state.may_destroy = 0;
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
                CardType::Ship => {
                    player.in_play.push(card);
                    let bonus: i32 = player.bases.iter().map(|b| b.ship_bonus()).sum();
                    state.combat += bonus;
                }
                CardType::Base | CardType::Outpost => {
                    // Playing a base uses its primary ability for this turn.
                    player.bases.push(card);
                    state.bases_used.push(card);
                }
            }
            played(state, card);
            resolve_all(state, card.data().primary, card);
        }
        Action::UseAlly(card) => {
            state.allies_used.push(card);
            resolve_all(state, card.data().ally, card);
        }
        Action::UseBase(card) => {
            state.bases_used.push(card);
            resolve_all(state, card.data().primary, card);
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
            to_scrap_heap(state, card.original());
            resolve_all(state, card.data().scrap, card);
        }
        Action::Acquire(card) => {
            state.players[me].discard.push(card);
            state.trade -= card.data().cost;
            take_from_market(state, card);
        }
        Action::AcquireOnTop(card) => {
            on_top(state, card);
            state.trade -= card.data().cost;
            state.may_on_top -= 1;
            take_from_market(state, card);
        }
        Action::AcquireFreeShip(card) => {
            on_top(state, card);
            state.free_ships -= 1;
            take_from_row(state, card);
        }
        Action::AttackPlayer => {
            state.players[1 - me].authority -= state.combat;
            state.combat = 0;
        }
        Action::AttackBase(card) => {
            destroy(state, card);
            state.combat -= card.data().defense;
        }
        Action::DestroyTarget(card) => {
            destroy(state, card);
            state.may_destroy = state.may_destroy.saturating_sub(1);
        }
        Action::Discard(card) => {
            let player = &mut state.players[me];
            remove_one(&mut player.hand, card);
            player.discard.push(card);
            player.to_discard -= 1;
        }
        Action::ScrapHandCard(card) => {
            remove_one(&mut state.players[me].hand, card);
            if state.must_scrap > 0 {
                state.must_scrap -= 1;
            } else {
                state.may_scrap -= 1;
            }
            to_scrap_heap(state, card);
        }
        Action::ScrapDiscardCard(card) => {
            remove_one(&mut state.players[me].discard, card);
            state.may_scrap -= 1;
            to_scrap_heap(state, card);
        }
        Action::ScrapTradeRowCard(card) => {
            take_from_row(state, card);
            to_scrap_heap(state, card);
            if state.trade_row.is_empty() {
                state.may_scrap_row = 0;
            } else {
                state.may_scrap_row -= 1;
            }
        }
        Action::Choose(card, option) => {
            state.choosing.remove(0);
            resolve_all(state, card.options()[usize::from(option)], card);
        }
        Action::Copy(target) => {
            let player = &mut state.players[me];
            let i = player
                .in_play
                .iter()
                .position(|c| c.copies_ships())
                .expect("a Stealth Needle is in play");
            let needle = player.in_play[i];
            let copy = Card::needle_as(target);
            player.in_play[i] = copy;
            let faction = target.data().faction;
            if faction.is_some() && faction != needle.data().faction {
                // A card of the target's faction too (Blob World counts it).
                played(state, target);
            }
            state.copying -= 1;
            resolve_all(state, copy.data().primary, copy);
        }
        Action::Recycle(cards) => {
            let player = &mut state.players[me];
            for &card in cards.as_slice() {
                remove_one(&mut player.hand, card);
            }
            player.discard.extend_from_slice(cards.as_slice());
            draw_current(state, cards.len());
            state.recycling = 0;
        }
        Action::ScrapAndDraw(from_hand, from_discard) => {
            let player = &mut state.players[me];
            for &card in from_hand.as_slice() {
                remove_one(&mut player.hand, card);
            }
            for &card in from_discard.as_slice() {
                remove_one(&mut player.discard, card);
            }
            draw_current(state, from_hand.len() + from_discard.len());
            state.scrap_to_draw = 0;
            for &card in from_hand.as_slice().iter().chain(from_discard.as_slice()) {
                to_scrap_heap(state, card);
            }
        }
        Action::EndTurn => {
            let player = &mut state.players[me];
            let in_play = std::mem::take(&mut player.in_play);
            let hand = std::mem::take(&mut player.hand);
            player
                .discard
                .extend(in_play.into_iter().map(Card::original));
            player.discard.extend(hand);
            state.rng_seed = draw(player, HAND_SIZE, state.rng_seed);
            // The next player discards no more cards than they hold.
            let opponent = &mut state.players[1 - me];
            opponent.to_discard = opponent.to_discard.min(opponent.hand.len() as u32);
            state.current = 1 - me;
            state.turn += 1;
            state.trade = 0;
            state.combat = 0;
            state.allies_used.clear();
            state.bases_used.clear();
            state.may_scrap = 0;
            state.may_destroy = 0;
            state.may_scrap_row = 0;
            state.factions_played = [0; 4];
            state.may_on_top = 0;
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
        assert_eq!(state.trade_deck.len(), 75);
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
        let action = Action::Acquire(Card::FleetHq);
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
        assert!(actions.iter().all(|&a| !automatic(&state, a)));
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
        state.players[1].bases = [Card::BlobWheel, Card::SpaceStation].into_iter().collect();
        state.combat = 10;
        let actions = legal_actions(&state);
        assert!(actions.contains(&Action::AttackBase(Card::SpaceStation)));
        assert!(!actions.contains(&Action::AttackBase(Card::BlobWheel)));
        assert!(!actions.contains(&Action::AttackPlayer));
    }

    #[test]
    fn pending_discards_come_first() {
        let mut state = dealt(0);
        state.players[1].to_discard = 1;
        apply_legal(&mut state, Action::EndTurn);
        auto_play(&mut state);
        let actions = legal_actions(&state);
        assert!(!actions.is_empty());
        assert!(actions.iter().all(|a| matches!(a, Action::Discard(_))));
    }

    #[test]
    fn an_or_choice_is_forced_and_names_each_option() {
        let mut state = dealt(0);
        state.players[0].hand.push(Card::PatrolMech);
        auto_play(&mut state);
        assert_eq!(state.choosing[..], [Card::PatrolMech]);
        assert_eq!(
            legal_actions(&state),
            [
                Action::Choose(Card::PatrolMech, 0),
                Action::Choose(Card::PatrolMech, 1)
            ]
        );
        let trade = state.trade;
        let state = advance(&state, Action::Choose(Card::PatrolMech, 0)).expect("legal");
        assert_eq!(state.trade, trade + 3);
    }

    #[test]
    fn stealth_needle_copies_a_ship_in_play() {
        let mut state = dealt(0);
        state.players[0].hand.push(Card::StealthNeedle);
        let state = advance(&state, Action::PlayCard(Card::StealthNeedle)).expect("legal");
        let actions = legal_actions(&state);
        assert!(actions.iter().all(|a| matches!(a, Action::Copy(_))));
        let target = state.players[0].in_play[0];
        let state = advance(&state, Action::Copy(target)).expect("legal");
        assert!(state.players[0].in_play.contains(&Card::needle_as(target)));
        let state = advance(&state, Action::EndTurn).expect("legal");
        assert!(state.players[0].discard.contains(&Card::StealthNeedle));
    }

    #[test]
    fn scrap_sets_include_none_and_mix_the_piles() {
        let sets = scrap_sets(&[Card::Viper, Card::Scout], &[Card::Scout], 2);
        let names: Vec<(Vec<Card>, Vec<Card>)> = sets
            .iter()
            .map(|(h, d)| (h.as_slice().to_vec(), d.as_slice().to_vec()))
            .collect();
        assert_eq!(names[0], (vec![], vec![]));
        assert_eq!(names[1], (vec![], vec![Card::Scout]));
        assert!(names.contains(&(vec![Card::Scout], vec![Card::Scout])));
        assert!(names.contains(&(vec![Card::Scout, Card::Viper], vec![])));
        assert_eq!(names.len(), 7);
    }
}
