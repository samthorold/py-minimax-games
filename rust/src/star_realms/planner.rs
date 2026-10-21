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
//! (`merge_key`); and a world stopping at its share, its last candidate worth
//! the best end found through it so far. The enumeration never reads a
//! value, so it can run before the scoring, and the ends come out in the
//! order the Python planner scores them.
//!
//! Then the choice (`plan`, `planner._choose`), scoring the ends with a
//! learned value's forward pass (`value.rs`), and the bot that plays the
//! plans (`PlannerBot`, `planner.PlannerBot`), for self-play in Rust.

use std::collections::VecDeque;
use std::collections::hash_map::Entry;

use rustc_hash::{FxHashMap, FxHashSet};

use super::bots::greedy_move;
use super::card::Card;
use super::features;
use super::rng::Random;
use super::rules::{Action, advance_legal, apply_legal, legal_actions, winner};
use super::state::{GameState, PlayerState};
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

fn sort_player(player: &mut PlayerState) {
    player.hand.sort_unstable();
    player.in_play.sort_unstable();
    player.discard.sort_unstable();
    player.bases.sort_unstable();
}

/// `state` with every pile but the decks (and the pending choices, which are
/// ordered) sorted: equal for two states exactly when `planner.merge_key` is.
pub fn merge_key(state: &GameState) -> GameState {
    let mut key = state.clone();
    for player in &mut key.players {
        sort_player(player);
    }
    key.trade_row.sort_unstable();
    key.scrap_heap.sort_unstable();
    key.allies_used.sort_unstable();
    key.bases_used.sort_unstable();
    key
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

/// Where `action` leads from `state` (`planner._end`): an end of turn
/// (`EndTurn` without the next player's automatic play, or a play that
/// wins) or a state within the turn.
enum Step {
    End(GameState),
    Within(GameState),
}

fn step(state: &GameState, action: Action) -> Step {
    if action == Action::EndTurn {
        let mut end = state.clone();
        apply_legal(&mut end, action);
        return Step::End(end);
    }
    let child = advance_legal(state, action);
    if winner(&child).is_some() {
        Step::End(child)
    } else {
        Step::Within(child)
    }
}

/// A world's share of the budget ran out.
struct OverBudget;

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

enum Claim {
    Seen(usize),
    New(usize),
}

enum Node {
    End(usize),
    Within(Vec<usize>),
}

/// One world's enumeration after the shared part (`planner._World`): the
/// states it has listed, as nodes of the graph of plays between them, and
/// the rows found through the current candidate.
struct World<'a> {
    keys: FxHashMap<GameState, usize>,
    nodes: Vec<Node>,
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
    /// The node of `key`: `Seen` if listed, else `New` and claimed (the
    /// caller pushes it), or `OverBudget` if the share is spent
    /// (`_World._seen`). One hash of the key either way.
    fn claim(&mut self, key: GameState) -> Result<Claim, OverBudget> {
        let next = self.nodes.len();
        let spent = self.keys.len() >= self.limit;
        match self.keys.entry(key) {
            Entry::Occupied(entry) => Ok(Claim::Seen(*entry.get())),
            Entry::Vacant(_) if spent => Err(OverBudget),
            Entry::Vacant(entry) => {
                entry.insert(next);
                self.stamp.push(self.candidate);
                Ok(Claim::New(next))
            }
        }
    }

    /// Note every end reachable from `node`, listed before: they are the
    /// ends `_World._note` takes the best of when it reuses `node`'s value.
    fn note(&mut self, node: usize) {
        if self.stamp[node] == self.candidate {
            return;
        }
        self.stamp[node] = self.candidate;
        let children = match &self.nodes[node] {
            Node::End(row) => {
                self.noted.push(*row);
                return;
            }
            Node::Within(children) => children.len(),
        };
        for i in 0..children {
            let Node::Within(children) = &self.nodes[node] else {
                unreachable!("a node within the turn stays one");
            };
            self.note(children[i]);
        }
    }

    /// `state`, an end of turn (`_World.end`).
    fn end(&mut self, state: GameState) -> Result<usize, OverBudget> {
        let id = match self.claim(merge_key(&state))? {
            Claim::Seen(node) => {
                self.note(node);
                return Ok(node);
            }
            Claim::New(id) => id,
        };
        let row = self.ends.len();
        self.ends.push(state);
        self.world_of.push(self.index);
        self.noted.push(row);
        self.nodes.push(Node::End(row));
        Ok(id)
    }

    /// Every end reachable from `state`, depth first, the greedy policy's
    /// moves first (`_World.best`).
    fn best(&mut self, state: &GameState) -> Result<usize, OverBudget> {
        // Claimed before its children, so the budget counts it.
        let id = match self.claim(merge_key(state))? {
            Claim::Seen(node) => {
                self.note(node);
                return Ok(node);
            }
            Claim::New(id) => id,
        };
        self.nodes.push(Node::Within(Vec::new()));
        let mut children = Vec::new();
        for (_, action) in ordered(state) {
            children.push(self.through(state, action)?);
        }
        self.nodes[id] = Node::Within(children);
        Ok(id)
    }

    fn through(&mut self, state: &GameState, action: Action) -> Result<usize, OverBudget> {
        match step(state, action) {
            Step::End(end) => self.end(end),
            Step::Within(child) => self.best(&child),
        }
    }
}

/// The shared part before the first chance event (`walk` in `planner.plan`).
struct Shared {
    /// Each shared state's parent and the play (as its index in
    /// `legal_actions`, and itself) that reaches it; the root's is unused.
    parents: Vec<(usize, usize, Action)>,
    states: Vec<GameState>,
    seen: FxHashSet<GameState>,
    candidates: Vec<(usize, usize, Action)>,
    limit: usize,
}

impl Shared {
    fn walk(&mut self, index: usize) -> Result<(), OverBudget> {
        let state = self.states[index].clone();
        for (at, action) in ordered(&state) {
            if action == Action::EndTurn {
                self.candidates.push((index, at, action));
                continue;
            }
            let child = advance_legal(&state, action);
            if winner(&child).is_some() || reveals(&state, &child) {
                self.candidates.push((index, at, action));
                continue;
            }
            let key = merge_key(&child);
            if self.seen.contains(&key) {
                continue;
            }
            if self.seen.len() >= self.limit {
                return Err(OverBudget);
            }
            self.seen.insert(key);
            self.parents.push((index, at, action));
            self.states.push(child);
            self.walk(self.states.len() - 1)?;
        }
        Ok(())
    }

    /// Shared state `index` in a world, replayed along its line and cached.
    fn replay(&self, cache: &mut Vec<Option<GameState>>, index: usize) -> GameState {
        if let Some(state) = &cache[index] {
            return state.clone();
        }
        let (parent, _, action) = self.parents[index];
        let state = advance_legal(&self.replay(cache, parent), action);
        cache[index] = Some(state.clone());
        state
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

/// The planner's enumeration over `worlds` (sampled worlds of one view, or
/// one true state), at most `budget` states visited in all.
pub fn enumerate(worlds: &[GameState], budget: usize) -> Enumeration {
    assert!(!worlds.is_empty(), "a plan needs at least 1 world");
    let first = &worlds[0];
    let mut shared = Shared {
        parents: vec![(0, 0, Action::EndTurn)],
        states: vec![first.clone()],
        seen: FxHashSet::from_iter([merge_key(first)]),
        candidates: Vec::new(),
        limit: budget / 2,
    };
    let mut found = Enumeration {
        complete: shared.limit >= 1,
        ..Enumeration::default()
    };
    if found.complete && shared.walk(0).is_err() {
        found.complete = false;
    }
    let share = (budget - shared.seen.len()) / worlds.len();
    found.states = shared.seen.len();
    for (index, world) in worlds.iter().enumerate() {
        let mut cache: Vec<Option<GameState>> = if index == 0 {
            shared.states.iter().cloned().map(Some).collect()
        } else {
            let mut cache = vec![None; shared.states.len()];
            cache[0] = Some(world.clone());
            cache
        };
        let mut search = World {
            keys: FxHashMap::default(),
            nodes: Vec::new(),
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
            let state = shared.replay(&mut cache, at);
            let result = search.through(&state, action);
            if result.is_err() {
                found.complete = false;
                if !search.noted.is_empty() {
                    reached.push(std::mem::take(&mut search.noted));
                }
                break;
            }
            reached.push(std::mem::take(&mut search.noted));
        }
        found.states += search.keys.len();
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
/// again if not (`diverged`) or once the line is played.
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
}
