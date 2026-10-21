//! The Python boundary: the extension module `minimax_games._native`, whose
//! submodule `star_realms` exposes the Rust engine. Actions cross as tuples
//! `(kind, *fields)` named as Python's `Action` variants and their fields
//! (a card as its name, a tuple of cards as a tuple of names, and a
//! `Choose`'s option as its index among the card's options), and states and
//! views as dicts of card names, so the differential tests compare them with
//! the Python engine's directly. Its stub is `minimax_games/_native/`.

use std::time::Instant;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::star_realms::bots::{greedy_move, play_greedy};
use crate::star_realms::card::{Card, Effect};
use crate::star_realms::features;
use crate::star_realms::planner::enumerate;
use crate::star_realms::rng::Random;
use crate::star_realms::rules::{
    Action, Few, advance, advance_legal, auto_play, legal_actions, new_game, winner,
};
use crate::star_realms::state::{GameState, PlayerState};
use crate::star_realms::view::{PlayerView, View, observe, sample_world};

fn names(cards: &[Card]) -> Vec<&'static str> {
    cards.iter().map(|card| card.name()).collect()
}

fn to_py<'py>(py: Python<'py>, action: Action) -> PyResult<Bound<'py, PyTuple>> {
    let card = |card: Card| card.name().into_bound_py_any(py);
    let few = |few: Few| -> PyResult<Bound<'py, PyAny>> {
        Ok(PyTuple::new(py, names(few.as_slice()))?.into_any())
    };
    let fields = match action {
        Action::PlayCard(c)
        | Action::UseAlly(c)
        | Action::UseBase(c)
        | Action::Scrap(c)
        | Action::Acquire(c)
        | Action::AcquireOnTop(c)
        | Action::AcquireFreeShip(c)
        | Action::AttackBase(c)
        | Action::DestroyTarget(c)
        | Action::Discard(c)
        | Action::ScrapHandCard(c)
        | Action::ScrapDiscardCard(c)
        | Action::ScrapTradeRowCard(c)
        | Action::Copy(c) => vec![card(c)?],
        Action::Choose(c, option) => vec![card(c)?, option.into_bound_py_any(py)?],
        Action::Recycle(cards) => vec![few(cards)?],
        Action::ScrapAndDraw(hand, discard) => vec![few(hand)?, few(discard)?],
        Action::AttackPlayer | Action::EndTurn => vec![],
    };
    let mut items = vec![action.kind().into_bound_py_any(py)?];
    items.extend(fields);
    PyTuple::new(py, items)
}

fn card_named(name: &str) -> PyResult<Card> {
    Card::from_name(name).ok_or_else(|| PyValueError::new_err(format!("unknown card {name:?}")))
}

fn from_py(action: &Bound<'_, PyTuple>) -> PyResult<Action> {
    let unknown = || PyValueError::new_err(format!("unknown action {action}"));
    let kind: String = action.get_item(0)?.extract()?;
    let fields: Vec<Bound<'_, PyAny>> = action.iter().skip(1).collect();
    let card = |i: usize| card_named(&fields[i].extract::<String>()?);
    let few = |i: usize| -> PyResult<Few> {
        let cards = fields[i]
            .extract::<Vec<String>>()?
            .iter()
            .map(|name| card_named(name))
            .collect::<PyResult<Vec<Card>>>()?;
        Few::new(&cards).ok_or_else(unknown)
    };
    Ok(match (kind.as_str(), fields.len()) {
        ("PlayCard", 1) => Action::PlayCard(card(0)?),
        ("UseAlly", 1) => Action::UseAlly(card(0)?),
        ("UseBase", 1) => Action::UseBase(card(0)?),
        ("Scrap", 1) => Action::Scrap(card(0)?),
        ("Acquire", 1) => Action::Acquire(card(0)?),
        ("AcquireOnTop", 1) => Action::AcquireOnTop(card(0)?),
        ("AcquireFreeShip", 1) => Action::AcquireFreeShip(card(0)?),
        ("AttackPlayer", 0) => Action::AttackPlayer,
        ("AttackBase", 1) => Action::AttackBase(card(0)?),
        ("DestroyTarget", 1) => Action::DestroyTarget(card(0)?),
        ("Discard", 1) => Action::Discard(card(0)?),
        ("ScrapHandCard", 1) => Action::ScrapHandCard(card(0)?),
        ("ScrapDiscardCard", 1) => Action::ScrapDiscardCard(card(0)?),
        ("ScrapTradeRowCard", 1) => Action::ScrapTradeRowCard(card(0)?),
        ("Choose", 2) => Action::Choose(card(0)?, fields[1].extract()?),
        ("Copy", 1) => Action::Copy(card(0)?),
        ("Recycle", 1) => Action::Recycle(few(0)?),
        ("ScrapAndDraw", 2) => Action::ScrapAndDraw(few(0)?, few(1)?),
        ("EndTurn", 0) => Action::EndTurn,
        _ => return Err(unknown()),
    })
}

fn player_dict<'py>(py: Python<'py>, player: &PlayerState) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("authority", player.authority)?;
    d.set_item("deck", names(&player.deck))?;
    d.set_item("hand", names(&player.hand))?;
    d.set_item("in_play", names(&player.in_play))?;
    d.set_item("discard", names(&player.discard))?;
    d.set_item("bases", names(&player.bases))?;
    d.set_item("to_discard", player.to_discard)?;
    d.set_item("known_top", player.known_top)?;
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
    d.set_item("may_scrap", state.may_scrap)?;
    d.set_item("must_scrap", state.must_scrap)?;
    d.set_item("may_destroy", state.may_destroy)?;
    d.set_item("may_scrap_row", state.may_scrap_row)?;
    d.set_item("choosing", names(&state.choosing))?;
    d.set_item("recycling", state.recycling)?;
    d.set_item("factions_played", state.factions_played.to_vec())?;
    d.set_item("may_on_top", state.may_on_top)?;
    d.set_item("free_ships", state.free_ships)?;
    d.set_item("scrap_to_draw", state.scrap_to_draw)?;
    d.set_item("copying", state.copying)?;
    Ok(d)
}

fn item<'py, T: FromPyObjectOwned<'py>>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<T> {
    d.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("no {key:?} in the state")))?
        .extract()
        .map_err(Into::into)
}

fn pile(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<Card>> {
    item::<Vec<String>>(d, key)?
        .iter()
        .map(|name| card_named(name))
        .collect()
}

fn player_from_dict(d: &Bound<'_, PyDict>) -> PyResult<PlayerState> {
    Ok(PlayerState {
        authority: item(d, "authority")?,
        deck: pile(d, "deck")?,
        hand: pile(d, "hand")?,
        in_play: pile(d, "in_play")?,
        discard: pile(d, "discard")?,
        bases: pile(d, "bases")?,
        to_discard: item(d, "to_discard")?,
        known_top: item(d, "known_top")?,
    })
}

/// A state from a dict of the `GameState` fields, as `state_dict` makes it.
fn state_from_dict(d: &Bound<'_, PyDict>) -> PyResult<GameState> {
    let players: Vec<Bound<'_, PyDict>> = item(d, "players")?;
    let [first, second] = players.as_slice() else {
        return Err(PyValueError::new_err("a state has 2 players"));
    };
    let factions: Vec<u32> = item(d, "factions_played")?;
    Ok(GameState {
        players: [player_from_dict(first)?, player_from_dict(second)?],
        rng_seed: item(d, "rng_seed")?,
        trade_deck: pile(d, "trade_deck")?,
        trade_row: pile(d, "trade_row")?,
        explorers: item(d, "explorers")?,
        scrap_heap: pile(d, "scrap_heap")?,
        current: item(d, "current")?,
        turn: item(d, "turn")?,
        trade: item(d, "trade")?,
        combat: item(d, "combat")?,
        allies_used: pile(d, "allies_used")?,
        bases_used: pile(d, "bases_used")?,
        may_scrap: item(d, "may_scrap")?,
        must_scrap: item(d, "must_scrap")?,
        may_destroy: item(d, "may_destroy")?,
        may_scrap_row: item(d, "may_scrap_row")?,
        choosing: pile(d, "choosing")?,
        recycling: item(d, "recycling")?,
        factions_played: factions
            .try_into()
            .map_err(|_| PyValueError::new_err("factions_played has 4 counts"))?,
        may_on_top: item(d, "may_on_top")?,
        free_ships: item(d, "free_ships")?,
        scrap_to_draw: item(d, "scrap_to_draw")?,
        copying: item(d, "copying")?,
    })
}

fn player_view_dict<'py>(py: Python<'py>, player: &PlayerView) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("authority", player.authority)?;
    d.set_item("hand_size", player.hand_size)?;
    d.set_item("deck_size", player.deck_size)?;
    d.set_item("hand", names(&player.hand))?;
    d.set_item("unseen", names(&player.unseen))?;
    d.set_item("in_play", names(&player.in_play))?;
    d.set_item("discard", names(&player.discard))?;
    d.set_item("bases", names(&player.bases))?;
    d.set_item("to_discard", player.to_discard)?;
    d.set_item("deck_top", names(&player.deck_top))?;
    Ok(d)
}

fn view_dict<'py>(py: Python<'py>, view: &View) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("observer", view.observer)?;
    let players = vec![
        player_view_dict(py, &view.players[0])?,
        player_view_dict(py, &view.players[1])?,
    ];
    d.set_item("players", players)?;
    d.set_item("trade_deck", names(&view.trade_deck))?;
    d.set_item("trade_row", names(&view.trade_row))?;
    d.set_item("explorers", view.explorers)?;
    d.set_item("scrap_heap", names(&view.scrap_heap))?;
    d.set_item("current", view.current)?;
    d.set_item("turn", view.turn)?;
    d.set_item("trade", view.trade)?;
    d.set_item("combat", view.combat)?;
    d.set_item("allies_used", names(&view.allies_used))?;
    d.set_item("bases_used", names(&view.bases_used))?;
    d.set_item("may_scrap", view.may_scrap)?;
    d.set_item("must_scrap", view.must_scrap)?;
    d.set_item("may_destroy", view.may_destroy)?;
    d.set_item("may_scrap_row", view.may_scrap_row)?;
    d.set_item("choosing", names(&view.choosing))?;
    d.set_item("recycling", view.recycling)?;
    d.set_item("factions_played", view.factions_played.to_vec())?;
    d.set_item("may_on_top", view.may_on_top)?;
    d.set_item("free_ships", view.free_ships)?;
    d.set_item("scrap_to_draw", view.scrap_to_draw)?;
    d.set_item("copying", view.copying)?;
    Ok(d)
}

fn observer_index(observer: usize) -> PyResult<usize> {
    if observer < 2 {
        Ok(observer)
    } else {
        Err(PyValueError::new_err(format!(
            "observer must be 0 or 1, not {observer}"
        )))
    }
}

/// A game, dealt from a seed and auto-played to its first decision, as
/// `rules.auto_play(rules.new_game(seed))` deals it.
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
    fn legal_actions<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyTuple>>> {
        legal_actions(&self.state)
            .into_iter()
            .map(|a| to_py(py, a))
            .collect()
    }

    /// Take the decision `action`, then auto-play (`rules.advance`).
    fn advance(&mut self, action: &Bound<'_, PyTuple>) -> PyResult<()> {
        let action = from_py(action)?;
        self.state =
            advance(&self.state, action).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(())
    }

    /// The state, as a dict of the `GameState` fields.
    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        state_dict(py, &self.state)
    }

    fn winner(&self) -> Option<usize> {
        winner(&self.state)
    }

    /// The greedy bot's choice now (`bots.greedy_move`).
    fn greedy_move<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let action = greedy_move(self.state.must_scrap, &legal_actions(&self.state));
        to_py(py, action)
    }

    /// What `observer` knows (`view.observe`), as a dict of the `View`
    /// fields.
    fn observe<'py>(&self, py: Python<'py>, observer: usize) -> PyResult<Bound<'py, PyDict>> {
        view_dict(py, &observe(&self.state, observer_index(observer)?))
    }

    /// A world consistent with `observer`'s view, sampled as
    /// `view.sample_world(observe(state, observer), random.Random(seed))`
    /// samples it, draw for draw.
    fn sample_world(&self, observer: usize, seed: u64) -> PyResult<Game> {
        let view = observe(&self.state, observer_index(observer)?);
        let state = sample_world(&view, &mut Random::new(seed));
        Ok(Game { state })
    }
}

/// An effect as `(kind, fields)`, an "or" choice's options as lists of
/// effects.
fn effect_py(py: Python<'_>, effect: Effect) -> PyResult<Bound<'_, PyAny>> {
    let int = |n: i64| n.into_bound_py_any(py);
    let fields: Vec<Bound<'_, PyAny>> = match effect {
        Effect::Trade(n)
        | Effect::Combat(n)
        | Effect::Authority(n)
        | Effect::ShipsGainCombat(n) => vec![int(i64::from(n))?],
        Effect::Draw(n) => vec![int(n as i64)?],
        Effect::OpponentDiscards(n)
        | Effect::ScrapHandOrDiscard(n)
        | Effect::ScrapTradeRow(n)
        | Effect::DiscardThenDraw(n)
        | Effect::ScrapThenDraw(n) => vec![int(i64::from(n))?],
        Effect::DrawIfBases { bases, cards } => vec![int(bases as i64)?, int(cards as i64)?],
        Effect::Choice(options) => {
            let options = options
                .iter()
                .map(|option| effects_py(py, option))
                .collect::<PyResult<Vec<_>>>()?;
            vec![PyList::new(py, options)?.into_any()]
        }
        Effect::DestroyBase
        | Effect::NextShipOnTop
        | Effect::AcquireShipFreeOnTop
        | Effect::DrawPerFactionCard
        | Effect::ScrapFromHand
        | Effect::CopyShip
        | Effect::AllyForAllFactions => vec![],
    };
    (effect.kind(), fields).into_bound_py_any(py)
}

fn effects_py<'py>(py: Python<'py>, effects: &[Effect]) -> PyResult<Bound<'py, PyList>> {
    let effects = effects
        .iter()
        .map(|&e| effect_py(py, e))
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, effects)
}

/// Each card's data, in catalogue order, as dicts of `card.Card`'s fields
/// (effects as `(kind, fields)` pairs), for checking against the catalogue.
#[pyfunction]
fn cards(py: Python<'_>) -> PyResult<Vec<Bound<'_, PyDict>>> {
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
            d.set_item("primary", effects_py(py, data.primary)?)?;
            d.set_item("ally", effects_py(py, data.ally)?)?;
            d.set_item("scrap", effects_py(py, data.scrap)?)?;
            d.set_item("copies", data.copies)?;
            Ok(d)
        })
        .collect()
}

/// Play a greedy-vs-greedy game from `seed`: `(winner, turns, decisions)`,
/// the winner `None` if it passed `max_turns`.
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
                let action = greedy_move(state.must_scrap, &legal_actions(&state));
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

fn int_array(py: Python<'_>, values: Vec<i64>) -> Bound<'_, PyArray1<i64>> {
    values.into_pyarray(py)
}

fn int(n: usize) -> i64 {
    i64::try_from(n).expect("an index fits in an i64")
}

/// The whole-turn planner's enumeration (`planner.rs`) over `worlds`, dicts
/// of the `GameState` fields, at most `budget` states visited, as a dict:
/// `features`, one row per distinct end (`features.encode`, a float64
/// matrix handed over without a copy), and per row its `world`, `winner`
/// (-1 for none) and `current` player; `lines`, each candidate's line as
/// indices into `legal_actions` from the first world's state; `reached`,
/// how many candidates each world reached; `offsets` and `rows`: for each
/// world's reached candidates in turn, `rows[offsets[k]:offsets[k + 1]]`
/// are the rows whose best score is the candidate's value there; `states`
/// and `complete`, as `Plan`'s. With `ends`, also the ends as state dicts.
#[pyfunction]
#[pyo3(signature = (worlds, budget, ends=false))]
fn plan_ends<'py>(
    py: Python<'py>,
    worlds: Vec<Bound<'py, PyDict>>,
    budget: usize,
    ends: bool,
) -> PyResult<Bound<'py, PyDict>> {
    if worlds.is_empty() {
        return Err(PyValueError::new_err("a plan needs at least 1 world"));
    }
    let worlds = worlds
        .iter()
        .map(state_from_dict)
        .collect::<PyResult<Vec<_>>>()?;
    let (found, matrix) = py.detach(|| {
        let found = enumerate(&worlds, budget);
        let columns = features::columns();
        let mut data = vec![0.0; found.ends.len() * columns];
        for (row, end) in data.chunks_exact_mut(columns).zip(&found.ends) {
            features::encode_into(row, end);
        }
        let matrix = Array2::from_shape_vec((found.ends.len(), columns), data)
            .expect("rows of the encoder's width");
        (found, matrix)
    });
    let d = PyDict::new(py);
    let matrix: Bound<'py, PyArray2<f64>> = matrix.into_pyarray(py);
    d.set_item("features", matrix)?;
    let world = found.world.iter().map(|&w| int(w)).collect();
    d.set_item("world", int_array(py, world))?;
    let winners = found
        .ends
        .iter()
        .map(|end| winner(end).map_or(-1, int))
        .collect();
    d.set_item("winner", int_array(py, winners))?;
    let current = found.ends.iter().map(|end| int(end.current)).collect();
    d.set_item("current", int_array(py, current))?;
    d.set_item("lines", &found.lines)?;
    let reached: Vec<usize> = found.reached.iter().map(Vec::len).collect();
    d.set_item("reached", reached)?;
    let mut offsets = vec![0];
    let mut rows = Vec::new();
    for candidate in found.reached.iter().flatten() {
        rows.extend(candidate.iter().map(|&r| int(r)));
        offsets.push(int(rows.len()));
    }
    d.set_item("offsets", int_array(py, offsets))?;
    d.set_item("rows", int_array(py, rows))?;
    d.set_item("states", found.states)?;
    d.set_item("complete", found.complete)?;
    if ends {
        let ends = found
            .ends
            .iter()
            .map(|end| state_dict(py, end))
            .collect::<PyResult<Vec<_>>>()?;
        d.set_item("ends", ends)?;
    }
    Ok(d)
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
    star_realms.add_function(wrap_pyfunction!(plan_ends, &star_realms)?)?;
    m.add_submodule(&star_realms)?;
    // So `import minimax_games._native.star_realms` finds it too.
    py.import("sys")?
        .getattr("modules")?
        .set_item("minimax_games._native.star_realms", &star_realms)?;
    Ok(())
}
