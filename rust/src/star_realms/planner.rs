//! The whole-turn planner's enumeration, a port of `planner.plan` up to the
//! scoring: for worlds sampled from one view, the distinct ends of the
//! current turn each world reaches, and which ends each candidate line
//! reaches in each world, so that a value scored on the ends' rows
//! (`features.rs`) gives every candidate the value `planner.plan` gives it.
//!
//! It makes the Python planner's walk exactly: the shared part before the
//! first chance event listed once, in the first world, with at most half the
//! budget; the rest split evenly among the worlds; each world listing the
//! ends through each candidate in turn, depth first, the greedy policy's move
//! first; states merged up to the order of every pile but the decks
//! (`merged`, `planner.merge_key`); and a world stopping at its share, its last candidate worth
//! the best end found through it so far. The enumeration never reads a
//! value, so it can run before the scoring, and the ends come out in the
//! order the Python planner scores them.
//!
//! Then the choice (`plan`, `planner._choose`), scoring the ends with a
//! learned value's forward pass (`value.rs`), and the bot that plays the
//! plans (`PlannerBot`, `planner.PlannerBot`), for self-play in Rust.
//!
//! The states it lists are kept once each, in lists, and its tables hold
//! their indices (#156): a play's state is made in place, a copy of its
//! parent's, and dropped again if it merges with one listed before.

use std::collections::VecDeque;
use std::collections::hash_map::Entry;
use std::hash::{Hash, Hasher};

use rustc_hash::{FxHashMap, FxHasher};

use super::bots::greedy_move;
use super::card::Card;
use super::features;
use super::rng::Random;
use super::rules::{Action, advance_legal, apply_legal, auto_play, legal_actions, winner};
use super::state::{GameState, PlayerState, capacity};
use super::value::Value;
use super::view::{PlayerView, View, observe, sample_world};

/// Whether a play from `before` to `after`, within the mover's turn, is a
/// chance event (`planner.reveals`): it draws a card no one knows, shuffles
/// the discard pile into the deck, or reveals a card from the Trade Deck.
pub fn reveals(before: &GameState, after: &GameState) -> bool {
    let was = &before.players[before.current];
    let now = &after.players[before.current];
    after.rng_seed != before.rng_seed
        || after.trade_deck.len() != before.trade_deck.len()
        || now.deck.len() - now.known_top != was.deck.len() - was.known_top
}

/// States are merged up to the order of every pile but the decks (and the
/// pending choices, which are ordered), as `planner.merge_key` merges them:
/// `merged` says whether two states merge, and `merge_hash` hashes a state
/// so that merged states hash alike. Neither copies a state or sorts a copy
/// of one, as the merge key used to: a pile whose order doesn't matter is
/// hashed as a sum over its cards, which no order changes, and compared as
/// a multiset.
mod merge {
    use super::*;

    /// A card's hash, before mixing.
    fn key(card: Card) -> u64 {
        let mut hasher = FxHasher::default();
        card.hash(&mut hasher);
        hasher.finish()
    }

    /// SplitMix64's finaliser: a bijection that spreads every input bit, so
    /// sums of mixed keys collide no more than chance.
    fn mix(mut x: u64) -> u64 {
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }

    /// A pile's hash, whatever its order.
    fn unordered(cards: &[Card]) -> u64 {
        cards
            .iter()
            .fold(0u64, |sum, &card| sum.wrapping_add(mix(key(card))))
    }

    /// A pile's hash, in its order.
    fn ordered(cards: &[Card]) -> u64 {
        cards.iter().enumerate().fold(0u64, |sum, (i, &card)| {
            let place = (i as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            sum.wrapping_add(mix(key(card) ^ place))
        })
    }

    /// How many cards of a deck its hash reads.
    const TOP: usize = 4;

    /// A deck's hash: its length and its top cards, in order. Merged states
    /// have equal decks, so equal hashes. Within a world's turn a deck only
    /// changes at the top (draws, and cards put on top; the Trade Deck as the
    /// Trade Row refills), so the cards below would cost more time than they
    /// save in comparisons; `merged` compares whole decks.
    fn deck(cards: &[Card]) -> u64 {
        ordered(&cards[..cards.len().min(TOP)]) ^ mix(cards.len() as u64)
    }

    fn hash_player(player: &PlayerState, h: &mut FxHasher) {
        let PlayerState {
            authority,
            deck,
            hand,
            in_play,
            discard,
            bases,
            to_discard,
            known_top,
        } = player;
        h.write_u64(self::deck(deck));
        h.write_u64(unordered(hand));
        h.write_u64(unordered(in_play));
        h.write_u64(unordered(discard));
        h.write_u64(unordered(bases));
        (authority, to_discard, known_top).hash(h);
    }

    /// `state`'s hash, the same for every state it merges with.
    pub fn merge_hash(state: &GameState) -> u64 {
        // Every field, by name, so a field added to `GameState` must be
        // placed here (and in `merged`) before this compiles.
        let GameState {
            players,
            rng_seed,
            trade_deck,
            trade_row,
            explorers,
            scrap_heap,
            current,
            turn,
            trade,
            combat,
            allies_used,
            bases_used,
            may_scrap,
            must_scrap,
            may_destroy,
            may_scrap_row,
            choosing,
            recycling,
            factions_played,
            may_on_top,
            free_ships,
            scrap_to_draw,
            copying,
        } = state;
        let mut h = FxHasher::default();
        hash_player(&players[0], &mut h);
        hash_player(&players[1], &mut h);
        h.write_u64(deck(trade_deck));
        h.write_u64(unordered(trade_row));
        h.write_u64(unordered(scrap_heap));
        h.write_u64(unordered(allies_used));
        h.write_u64(unordered(bases_used));
        h.write_u64(ordered(choosing));
        (rng_seed, explorers, current, turn, trade, combat).hash(&mut h);
        (may_scrap, must_scrap, may_destroy, may_scrap_row, recycling).hash(&mut h);
        (
            factions_played,
            may_on_top,
            free_ships,
            scrap_to_draw,
            copying,
        )
            .hash(&mut h);
        h.finish()
    }

    /// The largest pile whose order doesn't matter.
    const LARGEST: usize = capacity::OWNED;
    const _: () = assert!(
        capacity::SCRAP_HEAP <= LARGEST
            && capacity::IN_PLAY <= LARGEST
            && capacity::SHIPS <= LARGEST
            && capacity::BASES <= LARGEST
    );

    fn count(cards: &[Card], card: Card) -> usize {
        cards.iter().filter(|&&c| c == card).count()
    }

    /// Whether `a` and `b` hold the same cards, in any order.
    fn same_cards(a: &[Card], b: &[Card]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        if a == b {
            return true;
        }
        let n = a.len();
        if n <= 16 {
            // Each card as often in both: quicker than sorting copies.
            return a
                .iter()
                .enumerate()
                .all(|(i, c)| a[..i].contains(c) || count(a, *c) == count(b, *c));
        }
        let mut x = [Card::Scout; LARGEST];
        let mut y = [Card::Scout; LARGEST];
        x[..n].copy_from_slice(a);
        y[..n].copy_from_slice(b);
        x[..n].sort_unstable();
        y[..n].sort_unstable();
        x[..n] == y[..n]
    }

    fn players_merged(a: &PlayerState, b: &PlayerState) -> bool {
        let PlayerState {
            authority,
            deck,
            hand,
            in_play,
            discard,
            bases,
            to_discard,
            known_top,
        } = a;
        *authority == b.authority
            && *to_discard == b.to_discard
            && *known_top == b.known_top
            && *deck == b.deck
            && same_cards(hand, &b.hand)
            && same_cards(in_play, &b.in_play)
            && same_cards(discard, &b.discard)
            && same_cards(bases, &b.bases)
    }

    /// Whether `a` and `b` merge: equal up to the order of every pile but
    /// the decks and the pending choices (`planner.merge_key`).
    pub fn merged(a: &GameState, b: &GameState) -> bool {
        let GameState {
            players,
            rng_seed,
            trade_deck,
            trade_row,
            explorers,
            scrap_heap,
            current,
            turn,
            trade,
            combat,
            allies_used,
            bases_used,
            may_scrap,
            must_scrap,
            may_destroy,
            may_scrap_row,
            choosing,
            recycling,
            factions_played,
            may_on_top,
            free_ships,
            scrap_to_draw,
            copying,
        } = a;
        (
            rng_seed,
            explorers,
            current,
            turn,
            trade,
            combat,
            may_scrap,
            must_scrap,
            may_destroy,
            may_scrap_row,
        ) == (
            &b.rng_seed,
            &b.explorers,
            &b.current,
            &b.turn,
            &b.trade,
            &b.combat,
            &b.may_scrap,
            &b.must_scrap,
            &b.may_destroy,
            &b.may_scrap_row,
        ) && (
            recycling,
            factions_played,
            may_on_top,
            free_ships,
            scrap_to_draw,
            copying,
        ) == (
            &b.recycling,
            &b.factions_played,
            &b.may_on_top,
            &b.free_ships,
            &b.scrap_to_draw,
            &b.copying,
        ) && *choosing == b.choosing
            && *trade_deck == b.trade_deck
            && players_merged(&players[0], &b.players[0])
            && players_merged(&players[1], &b.players[1])
            && same_cards(trade_row, &b.trade_row)
            && same_cards(scrap_heap, &b.scrap_heap)
            && same_cards(allies_used, &b.allies_used)
            && same_cards(bases_used, &b.bases_used)
    }

    /// `state` with every pile but the decks and the pending choices sorted:
    /// equal for two states exactly when `merged` says they merge. The
    /// reference `merged` and `merge_hash` are tested against.
    #[cfg(test)]
    pub fn merge_key(state: &GameState) -> GameState {
        let mut key = state.clone();
        for player in &mut key.players {
            player.hand.sort_unstable();
            player.in_play.sort_unstable();
            player.discard.sort_unstable();
            player.bases.sort_unstable();
        }
        key.trade_row.sort_unstable();
        key.scrap_heap.sort_unstable();
        key.allies_used.sort_unstable();
        key.bases_used.sort_unstable();
        key
    }
}

pub use merge::{merge_hash, merged};

const NONE: u32 = u32::MAX;

/// The states listed so far, merged, as entries `0..len()` whose states the
/// caller keeps (`at`): the table holds only indices, chained by hash.
#[derive(Default)]
struct Table {
    /// The latest entry of each hash.
    heads: FxHashMap<u64, u32>,
    /// Each entry's predecessor of the same hash, or `NONE`.
    next: Vec<u32>,
}

/// What `Table::claim` found.
enum Claim {
    Seen(usize),
    New(usize),
}

/// A world's share of the budget ran out.
struct OverBudget;

impl Table {
    fn len(&self) -> usize {
        self.next.len()
    }

    /// The entry `state` merges with: `Seen` if listed, else `New`, a new
    /// entry the caller keeps `state` as, or `OverBudget` if `limit` entries
    /// are listed already. One lookup of the hash either way.
    fn claim<'s>(
        &mut self,
        state: &GameState,
        limit: usize,
        at: impl Fn(usize) -> &'s GameState,
    ) -> Result<Claim, OverBudget> {
        let id = self.next.len();
        let spent = id >= limit;
        match self.heads.entry(merge_hash(state)) {
            Entry::Occupied(mut head) => {
                let mut entry = *head.get();
                while entry != NONE {
                    if merged(at(entry as usize), state) {
                        return Ok(Claim::Seen(entry as usize));
                    }
                    entry = self.next[entry as usize];
                }
                if spent {
                    return Err(OverBudget);
                }
                self.next.push(head.insert(id as u32));
            }
            Entry::Vacant(_) if spent => return Err(OverBudget),
            Entry::Vacant(head) => {
                head.insert(id as u32);
                self.next.push(NONE);
            }
        }
        Ok(Claim::New(id))
    }
}

/// The legal actions in `state` with their indices in `legal_actions`, the
/// greedy policy's move first (`planner._ordered`).
fn ordered(state: &GameState) -> Vec<(usize, Action)> {
    let actions = legal_actions(state);
    let first = greedy_move(state.must_scrap, &actions);
    let at = actions
        .iter()
        .position(|&a| a == first)
        .expect("greedy picks a legal action");
    let mut found = vec![(at, first)];
    found.extend(actions.into_iter().enumerate().filter(|&(_, a)| a != first));
    found
}

/// Where `action` leads from `state` (`planner._end`), and whether it is an
/// end of turn (`EndTurn` without the next player's automatic play, or a
/// play that wins) rather than a state within the turn.
fn step(state: &GameState, action: Action) -> (GameState, bool) {
    if action == Action::EndTurn {
        let mut end = state.clone();
        apply_legal(&mut end, action);
        return (end, true);
    }
    let child = advance_legal(state, action);
    let end = winner(&child).is_some();
    (child, end)
}

/// The result of an enumeration.
#[derive(Debug, Default)]
pub struct Enumeration {
    /// The distinct ends each world listed (the first found of each merged
    /// set), world after world, in the order found: the rows.
    pub ends: Vec<GameState>,
    /// Each row's world.
    pub world: Vec<usize>,
    /// Each candidate's line from the first world's state, as indices into
    /// `legal_actions` at each step.
    pub lines: Vec<Vec<usize>>,
    /// For each world, for each candidate it reached, in candidate order: the
    /// rows whose best is the candidate's value in that world. A world that
    /// ran out of budget reaches a prefix of the candidates.
    pub reached: Vec<Vec<Vec<usize>>>,
    /// Distinct states visited, the shared ones once (`Plan.states`).
    pub states: usize,
    /// Whether every world listed every end of the turn (`Plan.complete`).
    pub complete: bool,
}

/// A node of a world's graph of plays: an end (its row), or a state within
/// the turn (its index among the world's states within the turn) and its
/// children (`edges[start..start + len]`).
#[derive(Clone, Copy)]
enum Node {
    End(usize),
    Within {
        state: usize,
        start: usize,
        len: usize,
    },
}

/// One world's enumeration after the shared part (`planner._World`): the
/// states it has listed, as nodes of the graph of plays between them, and
/// the rows found through the current candidate.
struct World<'a> {
    table: Table,
    nodes: Vec<Node>,
    /// The states within the turn, the nodes' that aren't ends.
    within: Vec<GameState>,
    /// The nodes' children, each node's together.
    edges: Vec<usize>,
    /// The children found so far of the nodes being expanded, innermost last.
    pending: Vec<usize>,
    limit: usize,
    /// The candidate a node was last noted for, to note each once.
    stamp: Vec<usize>,
    candidate: usize,
    noted: Vec<usize>,
    ends: &'a mut Vec<GameState>,
    world_of: &'a mut Vec<usize>,
    index: usize,
}

impl World<'_> {
    /// Note every end reachable from `node`, listed before: they are the
    /// ends `_World._note` takes the best of when it reuses `node`'s value.
    fn note(&mut self, node: usize) {
        if self.stamp[node] == self.candidate {
            return;
        }
        self.stamp[node] = self.candidate;
        match self.nodes[node] {
            Node::End(row) => self.noted.push(row),
            Node::Within { start, len, .. } => {
                for i in start..start + len {
                    self.note(self.edges[i]);
                }
            }
        }
    }

    /// The state a play just reached, the last of `ends` if it is an end,
    /// else the last of `within`, and every end reachable from it, depth
    /// first, the greedy policy's moves first (`_World.end`, `_World.best`):
    /// its node, if listed, the ends it reaches noted and the state dropped
    /// (`_World._seen`), else a new node. Claimed before its children, so
    /// the budget counts it.
    fn settle(&mut self, end: bool) -> Result<usize, OverBudget> {
        let (nodes, within, ends) = (&self.nodes, &self.within, &*self.ends);
        let state = if end { ends.last() } else { within.last() };
        let at = |node: usize| match nodes[node] {
            Node::End(row) => &ends[row],
            Node::Within { state, .. } => &within[state],
        };
        let claim = self
            .table
            .claim(state.expect("a state just reached"), self.limit, at);
        let id = match claim {
            Ok(Claim::New(id)) => id,
            Ok(Claim::Seen(_)) | Err(OverBudget) => {
                if end {
                    self.ends.pop();
                } else {
                    self.within.pop();
                }
                let Ok(Claim::Seen(node)) = claim else {
                    return Err(OverBudget);
                };
                self.note(node);
                return Ok(node);
            }
        };
        self.stamp.push(self.candidate);
        if end {
            let row = self.ends.len() - 1;
            self.world_of.push(self.index);
            self.noted.push(row);
            self.nodes.push(Node::End(row));
            return Ok(id);
        }
        let at = self.within.len() - 1;
        self.nodes.push(Node::Within {
            state: at,
            start: 0,
            len: 0,
        });
        let first = self.pending.len();
        for (_, action) in ordered(&self.within[at]) {
            // The play's state is made in place, a copy of its parent's, so
            // a state is copied once, and kept only if new.
            let end = if action == Action::EndTurn {
                self.ends
                    .extend_from_slice(std::slice::from_ref(&self.within[at]));
                let end = self.ends.last_mut().expect("just pushed");
                apply_legal(end, action);
                true
            } else {
                self.within.extend_from_within(at..=at);
                let child = self.within.last_mut().expect("just pushed");
                apply_legal(child, action);
                auto_play(child);
                if winner(child).is_some() {
                    let won = self.within.pop().expect("just pushed");
                    self.ends.push(won);
                    true
                } else {
                    false
                }
            };
            let child = self.settle(end)?;
            self.pending.push(child);
        }
        let start = self.edges.len();
        self.edges.extend(self.pending.drain(first..));
        self.nodes[id] = Node::Within {
            state: at,
            start,
            len: self.edges.len() - start,
        };
        Ok(id)
    }

    /// The ends through `action` from `state`, the current candidate's.
    fn through(&mut self, state: &GameState, action: Action) -> Result<usize, OverBudget> {
        self.pending.clear();
        let (child, end) = step(state, action);
        if end {
            self.ends.push(child);
        } else {
            self.within.push(child);
        }
        self.settle(end)
    }
}

/// The shared part before the first chance event (`walk` in `planner.plan`).
struct Shared {
    /// Each shared state's parent and the play (as its index in
    /// `legal_actions`, and itself) that reaches it; the root's is unused.
    parents: Vec<(usize, usize, Action)>,
    /// The shared states, the table's entries.
    states: Vec<GameState>,
    table: Table,
    candidates: Vec<(usize, usize, Action)>,
    limit: usize,
}

impl Shared {
    fn walk(&mut self, index: usize) -> Result<(), OverBudget> {
        for (at, action) in ordered(&self.states[index]) {
            if action == Action::EndTurn {
                self.candidates.push((index, at, action));
                continue;
            }
            // Made in place, as in `World::settle`.
            self.states.extend_from_within(index..=index);
            let child = self.states.last_mut().expect("just pushed");
            apply_legal(child, action);
            auto_play(child);
            let states = &self.states;
            let child = &states[states.len() - 1];
            if winner(child).is_some() || reveals(&states[index], child) {
                self.states.pop();
                self.candidates.push((index, at, action));
                continue;
            }
            match self.table.claim(child, self.limit, |i| &states[i]) {
                Ok(Claim::New(_)) => {}
                Ok(Claim::Seen(_)) => {
                    self.states.pop();
                    continue;
                }
                Err(OverBudget) => {
                    self.states.pop();
                    return Err(OverBudget);
                }
            }
            self.parents.push((index, at, action));
            self.walk(self.states.len() - 1)?;
        }
        Ok(())
    }

    /// Shared state `index` in another world, replayed along its line: its
    /// index in `replayed`, where `slots` keeps each shared state's (or
    /// `NONE`), the world's own state in slot 0.
    fn replay(&self, slots: &mut [u32], replayed: &mut Vec<GameState>, index: usize) -> usize {
        if slots[index] != NONE {
            return slots[index] as usize;
        }
        let (parent, _, action) = self.parents[index];
        let from = self.replay(slots, replayed, parent);
        let state = advance_legal(&replayed[from], action);
        slots[index] = replayed.len() as u32;
        replayed.push(state);
        replayed.len() - 1
    }

    fn line(&self, candidate: usize) -> Vec<usize> {
        let (mut index, at, _) = self.candidates[candidate];
        let mut line = vec![at];
        while index > 0 {
            let (parent, at, _) = self.parents[index];
            line.push(at);
            index = parent;
        }
        line.reverse();
        line
    }
}

/// Room for `states` states and the one being tried, so the lists of states
/// rarely grow: memory not written to is never touched.
fn reserved(states: usize) -> Vec<GameState> {
    Vec::with_capacity(states.min(RESERVED) + 1)
}

/// The most states `reserved` makes room for, about 60 MB.
const RESERVED: usize = 25_000;

/// The planner's enumeration over `worlds` (sampled worlds of one view, or
/// one true state), at most `budget` states visited in all.
pub fn enumerate(worlds: &[GameState], budget: usize) -> Enumeration {
    assert!(!worlds.is_empty(), "a plan needs at least 1 world");
    let first = &worlds[0];
    let mut shared = Shared {
        parents: vec![(0, 0, Action::EndTurn)],
        states: reserved(budget / 2),
        table: Table::default(),
        candidates: Vec::new(),
        limit: budget / 2,
    };
    shared.states.push(first.clone());
    let states = &shared.states;
    let root = shared.table.claim(first, usize::MAX, |i| &states[i]);
    assert!(
        matches!(root, Ok(Claim::New(0))),
        "the root is the first entry"
    );
    let mut found = Enumeration {
        ends: reserved(budget),
        complete: shared.limit >= 1,
        ..Enumeration::default()
    };
    if found.complete && shared.walk(0).is_err() {
        found.complete = false;
    }
    let share = (budget - shared.table.len()) / worlds.len();
    found.states = shared.table.len();
    for (index, world) in worlds.iter().enumerate() {
        // The first world's shared states are `shared.states`; the others'
        // are replayed as their candidates need them.
        let mut slots = Vec::new();
        let mut replayed = Vec::new();
        if index > 0 {
            slots = vec![NONE; shared.states.len()];
            slots[0] = 0;
            replayed.push(world.clone());
        }
        let mut search = World {
            table: Table::default(),
            nodes: Vec::new(),
            within: reserved(share),
            edges: Vec::new(),
            pending: Vec::new(),
            limit: share,
            stamp: Vec::new(),
            candidate: 0,
            noted: Vec::new(),
            ends: &mut found.ends,
            world_of: &mut found.world,
            index,
        };
        let mut reached = Vec::new();
        for (candidate, &(at, _, action)) in shared.candidates.iter().enumerate() {
            search.candidate = candidate + 1;
            search.noted.clear();
            let state = if index == 0 {
                &shared.states[at]
            } else {
                let slot = shared.replay(&mut slots, &mut replayed, at);
                &replayed[slot]
            };
            let result = search.through(state, action);
            if result.is_err() {
                found.complete = false;
                if !search.noted.is_empty() {
                    reached.push(std::mem::take(&mut search.noted));
                }
                break;
            }
            reached.push(std::mem::take(&mut search.noted));
        }
        found.states += search.table.len();
        found.reached.push(reached);
    }
    found.lines = (0..shared.candidates.len())
        .map(|c| shared.line(c))
        .collect();
    found
}

/// A plan (`planner.Plan`).
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    /// The decisions to play from the first world's state, up to and
    /// including the next chance event or the end of the turn.
    pub line: Vec<Action>,
    /// The mean over worlds of the best end found through the line, from
    /// the mover's side (`None` if no end was found).
    pub value: Option<f64>,
    pub states: usize,
    pub complete: bool,
}

/// The best line from `worlds`, at most `budget` states visited, the ends
/// scored by `value` from the mover's side (`planner.plan` with a
/// `ValueEvaluator`). A candidate is worth, in each world that reached it,
/// the best score of the ends `enumerate` gives it there; the candidates
/// reached in the most worlds compete by their mean, ties to the first
/// found (`planner._choose`); if none is reached, the line is the greedy
/// move. Scores are whole numbers, so their sums are exact, and the mean is
/// the same correctly rounded division as Python's `int / int`.
pub fn plan(worlds: &[GameState], budget: usize, value: &Value) -> Plan {
    let found = enumerate(worlds, budget);
    let first = &worlds[0];
    let mut row = vec![0.0; features::columns()];
    let mut scratch = Vec::new();
    let scores: Vec<i64> = found
        .ends
        .iter()
        .map(|end| {
            row.fill(0.0);
            features::encode_into(&mut row, end);
            value.score(end, &row, first.current, &mut scratch)
        })
        .collect();
    let mut values: Vec<Vec<i64>> = vec![Vec::new(); found.lines.len()];
    for reached in &found.reached {
        for (candidate, rows) in reached.iter().enumerate() {
            let best = rows.iter().map(|&r| scores[r]).max();
            values[candidate].push(best.expect("a candidate reaches an end"));
        }
    }
    let reached = values.iter().map(Vec::len).max().unwrap_or(0);
    if reached == 0 {
        return Plan {
            line: vec![ordered(first)[0].1],
            value: None,
            states: found.states,
            complete: false,
        };
    }
    let mut best: Option<(usize, f64)> = None;
    for (candidate, scores) in values.iter().enumerate() {
        if scores.len() != reached {
            continue;
        }
        let mean = scores.iter().sum::<i64>() as f64 / reached as f64;
        if best.is_none_or(|(_, top)| mean > top) {
            best = Some((candidate, mean));
        }
    }
    let (candidate, mean) = best.expect("a candidate reached in the most worlds");
    let mut state = first.clone();
    let mut line = Vec::new();
    for &i in &found.lines[candidate] {
        let action = legal_actions(&state)[i];
        state = advance_legal(&state, action);
        line.push(action);
    }
    Plan {
        line,
        value: Some(mean),
        states: found.states,
        complete: found.complete,
    }
}

fn sorted(cards: &[Card]) -> Vec<Card> {
    let mut cards = cards.to_vec();
    cards.sort_unstable();
    cards
}

fn player_seen(player: &PlayerView) -> PlayerView {
    PlayerView {
        in_play: sorted(&player.in_play),
        discard: sorted(&player.discard),
        bases: sorted(&player.bases),
        ..player.clone()
    }
}

/// `view` up to the order of its piles (`planner.seen`): sorting by the
/// card enum's order gives the same equality as Python's sort by name.
pub fn seen(view: &View) -> View {
    View {
        players: [player_seen(&view.players[0]), player_seen(&view.players[1])],
        trade_row: sorted(&view.trade_row),
        scrap_heap: sorted(&view.scrap_heap),
        allies_used: sorted(&view.allies_used),
        bases_used: sorted(&view.bases_used),
        ..view.clone()
    }
}

/// `planner.PlannerBot` with a learned value: plans whole turns over
/// `worlds` worlds sampled from its view with an rng seeded as Python's
/// `random.Random(seed)`, and plays each plan's line while its view is, up
/// to pile order, the one the line reached in the first world, planning
/// again if not (`diverged`) or once the line is played. `examined` notes
/// each plan's turn and whether it examined every end of the turn in every
/// world (`Plan::complete`).
pub struct PlannerBot<'a> {
    value: &'a Value,
    worlds: usize,
    budget: usize,
    rng: Random,
    turn: Option<u32>,
    line: VecDeque<(View, Action)>,
    pub plans: u64,
    pub turns: u64,
    pub diverged: u64,
    pub examined: Vec<(u32, bool)>,
}

impl<'a> PlannerBot<'a> {
    pub fn new(value: &'a Value, worlds: usize, budget: usize, seed: u64) -> Self {
        assert!(worlds >= 1, "the planner needs at least 1 world");
        assert!(budget >= 2 * worlds, "at least 2 states per world");
        PlannerBot {
            value,
            worlds,
            budget,
            rng: Random::new(seed),
            turn: None,
            line: VecDeque::new(),
            plans: 0,
            turns: 0,
            diverged: 0,
            examined: Vec::new(),
        }
    }

    /// The bot's move from `view`, among the legal `actions`
    /// (`PlannerBot.choose`).
    pub fn choose(&mut self, view: &View, actions: &[Action]) -> Action {
        if self.turn != Some(view.turn) {
            self.turn = Some(view.turn);
            self.turns += 1;
            self.line.clear();
        }
        if let Some((expected, _)) = self.line.front()
            && *expected != seen(view)
        {
            self.diverged += 1;
            self.line.clear();
        }
        if self.line.is_empty() {
            let worlds: Vec<GameState> = (0..self.worlds)
                .map(|_| sample_world(view, &mut self.rng))
                .collect();
            let made = plan(&worlds, self.budget, self.value);
            self.plans += 1;
            self.examined.push((view.turn, made.complete));
            let mut state = worlds[0].clone();
            for action in made.line {
                self.line
                    .push_back((seen(&observe(&state, view.observer)), action));
                state = advance_legal(&state, action);
            }
        }
        let (_, action) = self.line.pop_front().expect("a plan has a line");
        debug_assert!(actions.contains(&action), "{action:?} not in {actions:?}");
        action
    }
}

#[cfg(test)]
mod tests {
    use super::super::rules::{auto_play, new_game};
    use super::super::value::Form;
    use super::*;

    #[test]
    fn a_plan_s_line_is_legal_and_ends_at_a_chance_event_or_the_turn_s_end() {
        let columns = features::columns();
        let value = Value::new(
            vec![0.0; columns],
            vec![1.0; columns],
            Form::Logistic {
                w: vec![0.01; columns],
                b: 0.0,
            },
        )
        .unwrap();
        let mut state = new_game(4);
        auto_play(&mut state);
        let made = plan(std::slice::from_ref(&state), 2_000, &value);
        assert!(made.value.is_some());
        let mut at = state;
        for (i, &action) in made.line.iter().enumerate() {
            assert!(legal_actions(&at).contains(&action));
            let next = advance_legal(&at, action);
            let stops = action == Action::EndTurn || reveals(&at, &next) || winner(&next).is_some();
            assert_eq!(stops, i + 1 == made.line.len());
            at = next;
        }
    }

    #[test]
    fn the_first_turn_is_listed_within_a_small_budget() {
        let mut state = new_game(0);
        auto_play(&mut state);
        let found = enumerate(std::slice::from_ref(&state), 10_000);
        assert!(found.complete);
        assert!(!found.ends.is_empty());
        assert!(found.states <= 10_000);
        assert_eq!(found.reached.len(), 1);
        assert_eq!(found.reached[0].len(), found.lines.len());
        // the greedy line comes first and reaches an end
        assert!(!found.reached[0][0].is_empty());
    }

    #[test]
    fn a_tiny_budget_is_never_exceeded() {
        let mut state = new_game(3);
        auto_play(&mut state);
        for budget in [1, 2, 3, 5, 8, 50] {
            let found = enumerate(std::slice::from_ref(&state), budget);
            assert!(found.states <= budget, "budget {budget}");
        }
    }

    /// States along greedy games, with every play from each.
    fn positions() -> Vec<GameState> {
        let mut states = Vec::new();
        for seed in 0..6 {
            let mut state = new_game(seed);
            auto_play(&mut state);
            while winner(&state).is_none() && state.turn <= 40 {
                let actions = legal_actions(&state);
                states.extend(actions.iter().map(|&a| step(&state, a).0));
                states.push(state.clone());
                state = advance_legal(&state, greedy_move(state.must_scrap, &actions));
            }
        }
        states
    }

    #[test]
    fn states_merge_as_their_sorted_keys_are_equal() {
        let states = positions();
        let keys: Vec<GameState> = states.iter().map(merge::merge_key).collect();
        let mut merges = 0;
        for (i, a) in states.iter().enumerate() {
            // Neighbours, which often merge, and a stride across the games.
            for j in (i.saturating_sub(12)..i).chain((0..i).step_by(97)) {
                let b = &states[j];
                let same = keys[i] == keys[j];
                assert_eq!(merged(a, b), same, "{i} {j}");
                assert_eq!(merged(b, a), same, "{j} {i}");
                if same {
                    merges += 1;
                    assert_eq!(merge_hash(a), merge_hash(b), "{i} {j}");
                }
            }
        }
        assert!(merges > 100, "only {merges} merges tested");
    }

    #[test]
    fn a_pile_s_order_matters_only_in_the_decks_and_the_choices() {
        for state in positions().into_iter().step_by(7) {
            let mut other = state.clone();
            for player in &mut other.players {
                player.hand.reverse();
                player.in_play.reverse();
                let half = player.discard.len() / 2;
                player.discard.rotate_left(half);
                player.bases.reverse();
            }
            other.trade_row.reverse();
            other.scrap_heap.reverse();
            other.allies_used.reverse();
            other.bases_used.reverse();
            assert!(merged(&state, &other));
            assert_eq!(merge_hash(&state), merge_hash(&other));
            let deck = &mut other.players[state.current].deck;
            if deck.iter().any(|&c| c != deck[0]) {
                deck.reverse();
                if deck[..] != state.players[state.current].deck[..] {
                    assert!(!merged(&state, &other));
                }
            }
            other = state.clone();
            other.trade_deck.swap(0, 1);
            assert_eq!(merged(&state, &other), other.trade_deck == state.trade_deck);
        }
    }
}
