//! The Python boundary: the extension module `minimax_games._native`, whose
//! submodule `star_realms` exposes the Rust engine. Actions cross as
//! `(kind, card name or None)` pairs, named as Python's `Action` variants, and
//! states as dicts of card names, so the differential tests compare them with
//! the Python engine's directly. Its stub is `minimax_games/_native/`.

use std::time::Instant;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::star_realms::bots::{greedy_move, play_greedy};
use crate::star_realms::card::Card;
use crate::star_realms::rules::{
    Action, advance, advance_legal, auto_play, legal_actions, new_game, winner,
};
use crate::star_realms::state::{GameState, PlayerState};

type PyAction = (&'static str, Option<&'static str>);

fn to_py(action: Action) -> PyAction {
    (action.kind(), action.card().map(Card::name))
}

fn from_py(kind: &str, card: Option<&str>) -> PyResult<Action> {
    let card = match card {
        None => None,
        Some(name) => Some(
            Card::from_name(name)
                .ok_or_else(|| PyValueError::new_err(format!("unknown card {name:?}")))?,
        ),
    };
    Action::from_parts(kind, card)
        .ok_or_else(|| PyValueError::new_err(format!("unknown action ({kind:?}, {card:?})")))
}

fn names(cards: &[Card]) -> Vec<&'static str> {
    cards.iter().map(|card| card.name()).collect()
}

fn player_dict<'py>(py: Python<'py>, player: &PlayerState) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("authority", player.authority)?;
    d.set_item("deck", names(&player.deck))?;
    d.set_item("hand", names(&player.hand))?;
    d.set_item("in_play", names(&player.in_play))?;
    d.set_item("discard", names(&player.discard))?;
    d.set_item("bases", names(&player.bases))?;
    Ok(d)
}

fn state_dict<'py>(py: Python<'py>, state: &GameState) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let players = vec![
        player_dict(py, &state.players[0])?,
        player_dict(py, &state.players[1])?,
    ];
    d.set_item("players", players)?;
    d.set_item("rng_seed", state.rng_seed)?;
    d.set_item("trade_deck", names(&state.trade_deck))?;
    d.set_item("trade_row", names(&state.trade_row))?;
    d.set_item("explorers", state.explorers)?;
    d.set_item("scrap_heap", names(&state.scrap_heap))?;
    d.set_item("current", state.current)?;
    d.set_item("turn", state.turn)?;
    d.set_item("trade", state.trade)?;
    d.set_item("combat", state.combat)?;
    d.set_item("allies_used", names(&state.allies_used))?;
    d.set_item("bases_used", names(&state.bases_used))?;
    d.set_item("factions_played", state.factions_played.to_vec())?;
    Ok(d)
}

/// A first-slice game, dealt from a seed and auto-played to its first
/// decision, as `rules.auto_play(rules.new_game(seed, deck))` deals it.
#[pyclass(module = "minimax_games._native.star_realms")]
struct Game {
    state: GameState,
}

#[pymethods]
impl Game {
    #[new]
    fn new(seed: u64) -> Self {
        let mut state = new_game(seed);
        auto_play(&mut state);
        Game { state }
    }

    /// The legal actions, in `rules.legal_actions` order.
    fn legal_actions(&self) -> Vec<PyAction> {
        legal_actions(&self.state).into_iter().map(to_py).collect()
    }

    /// Take the decision `action`, then auto-play (`rules.advance`).
    #[pyo3(signature = (kind, card=None))]
    fn advance(&mut self, kind: &str, card: Option<&str>) -> PyResult<()> {
        let action = from_py(kind, card)?;
        self.state =
            advance(&self.state, action).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(())
    }

    /// The state, as a dict of the `GameState` fields the first slice uses.
    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        state_dict(py, &self.state)
    }

    fn winner(&self) -> Option<usize> {
        winner(&self.state)
    }

    /// The greedy bot's choice now (`bots.greedy_move`).
    fn greedy_move(&self) -> PyAction {
        to_py(greedy_move(&legal_actions(&self.state)))
    }
}

/// Each card's data, in catalogue order, as dicts of `card.Card`'s fields
/// (effects as `(kind, amount)` pairs), for checking against the catalogue.
#[pyfunction]
fn cards(py: Python<'_>) -> PyResult<Vec<Bound<'_, PyDict>>> {
    use crate::star_realms::card::Effect;
    let effects = |effects: &[Effect]| -> Vec<(&'static str, i64)> {
        effects
            .iter()
            .map(|&e| match e {
                Effect::Trade(n) => ("Trade", i64::from(n)),
                Effect::Combat(n) => ("Combat", i64::from(n)),
                Effect::Authority(n) => ("Authority", i64::from(n)),
                Effect::Draw(n) => ("Draw", n as i64),
            })
            .collect()
    };
    Card::ALL
        .into_iter()
        .map(|card| {
            let data = card.data();
            let d = PyDict::new(py);
            d.set_item("name", data.name)?;
            d.set_item("cost", data.cost)?;
            d.set_item("faction", data.faction.map(|f| f.name()))?;
            d.set_item("type", data.card_type.name())?;
            d.set_item("defense", (!card.is_ship()).then_some(data.defense))?;
            d.set_item("primary", effects(data.primary))?;
            d.set_item("ally", effects(data.ally))?;
            d.set_item("scrap", effects(data.scrap))?;
            d.set_item("copies", data.copies)?;
            Ok(d)
        })
        .collect()
}

/// Play a greedy-vs-greedy first-slice game from `seed`: `(winner, turns,
/// decisions)`, the winner `None` if it passed `max_turns`.
#[pyfunction]
#[pyo3(name = "play_greedy", signature = (seed, max_turns=1000))]
fn py_play_greedy(seed: u64, max_turns: u32) -> (Option<usize>, u32, u64) {
    let result = play_greedy(seed, max_turns);
    (result.winner, result.turns, result.decisions)
}

/// Time `games` greedy-vs-greedy games from `base_seed` on: `(games,
/// decisions, seconds)`.
#[pyfunction]
fn bench_games(py: Python<'_>, games: u64, base_seed: u64) -> (u64, u64, f64) {
    py.detach(|| {
        let start = Instant::now();
        let decisions: u64 = (base_seed..base_seed + games)
            .map(|seed| play_greedy(seed, 1000).decisions)
            .sum();
        (games, decisions, start.elapsed().as_secs_f64())
    })
}

/// Time `advance` alone (`advance_legal`, with automatic play) over every
/// decision of `games` greedy games from `base_seed` on, recorded first:
/// `(calls, seconds)`.
#[pyfunction]
fn bench_advance(py: Python<'_>, games: u64, base_seed: u64) -> (u64, f64) {
    py.detach(|| {
        let mut moves = Vec::new();
        for seed in base_seed..base_seed + games {
            let mut state = new_game(seed);
            auto_play(&mut state);
            while winner(&state).is_none() && state.turn <= 1000 {
                let action = greedy_move(&legal_actions(&state));
                let next = advance_legal(&state, action);
                moves.push((state, action));
                state = next;
            }
        }
        let start = Instant::now();
        let mut turns = 0u64;
        for (state, action) in &moves {
            turns += u64::from(std::hint::black_box(advance_legal(state, *action)).turn);
        }
        let seconds = start.elapsed().as_secs_f64();
        std::hint::black_box(turns);
        (moves.len() as u64, seconds)
    })
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let star_realms = PyModule::new(py, "star_realms")?;
    star_realms.add_class::<Game>()?;
    star_realms.add_function(wrap_pyfunction!(cards, &star_realms)?)?;
    star_realms.add_function(wrap_pyfunction!(py_play_greedy, &star_realms)?)?;
    star_realms.add_function(wrap_pyfunction!(bench_games, &star_realms)?)?;
    star_realms.add_function(wrap_pyfunction!(bench_advance, &star_realms)?)?;
    m.add_submodule(&star_realms)?;
    // So `import minimax_games._native.star_realms` finds it too.
    py.import("sys")?
        .getattr("modules")?
        .set_item("minimax_games._native.star_realms", &star_realms)?;
    Ok(())
}
