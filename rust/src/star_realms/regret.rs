//! Rollouts for the regret instrument (docs/star-realms.md, Learned values:
//! regret by kind of decision; `regret.py` is the Python side): each option
//! at a decision, a line of decisions from the mover's turn, is played to
//! the game's end from worlds sampled from the mover's view, the planner
//! with one value playing both sides.
//!
//! A rollout is fixed by its seed: the world (`sample_world` from the
//! mover's view, with an rng seeded from it), each seat's planner's seed,
//! and, if asked for, the base of keyed shuffles (`state::Keyed`) and the
//! key of each planner's sampled worlds (`Keying`). Every
//! option rolled out with the same seed gets the same world and the same
//! future randomness, as far as the options leave it the same: common random
//! numbers.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use super::planner::{PlannerBot, Ranked, ranked};
use super::rng::{Random, mix};
use super::rules::{Action, IllegalAction, advance, apply_legal, auto_play, legal_actions, winner};
use super::state::{GameState, Keyed};
use super::value::Value;
use super::view::{observe, sample_world};

/// The planner both sides play: its value, worlds and states per plan.
#[derive(Clone, Copy, Debug)]
pub struct Policy<'a> {
    pub value: &'a Value,
    pub worlds: usize,
    pub budget: usize,
}

/// How a rollout ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rollout {
    /// The winner, `None` if the game passed `max_turns`.
    pub winner: Option<usize>,
    pub turns: u32,
}

/// The candidates the planner (`policy`, its rng seeded with `seed`, as
/// `PlannerBot` seeds it) compares at `start`, the mover to move, best
/// first by its judgement (`planner::ranked`): the first is the line the
/// planner plays.
pub fn shortlist(policy: Policy<'_>, start: &GameState, seed: u64) -> Vec<Ranked> {
    let view = observe(start, start.current);
    let mut rng = Random::new(seed);
    let worlds: Vec<GameState> = (0..policy.worlds)
        .map(|_| sample_world(&view, &mut rng))
        .collect();
    ranked(&worlds, policy.budget, policy.value)
}

/// The world a rollout with `seed` starts from: sampled from the view of
/// the player to move at `start`, with keyed shuffles from the seed if
/// `keyed`.
pub fn world(start: &GameState, seed: u64, keyed: bool) -> GameState {
    let view = observe(start, start.current);
    let mut state = sample_world(&view, &mut Random::new(mix(seed)));
    if keyed {
        state.keyed = Some(Keyed::new(mix(seed ^ 3)));
    }
    state
}

/// Which of a rollout's randomness is keyed, beyond its seed: the
/// reshuffles (`state::Keyed`) and the planners' sampled worlds
/// (`PlannerBot::with_keyed_worlds`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Keying {
    pub shuffles: bool,
    pub worlds: bool,
}

/// `line` played from `start` in the world of `seed`, then the game played
/// out by `policy` on both sides, each seat's planner seeded from `seed`
/// (its worlds keyed by it if `keying.worlds`), stopping unfinished past
/// `max_turns`. An error if the line isn't legal in that world (a line
/// that goes past a chance event).
pub fn rollout(
    policy: Policy<'_>,
    start: &GameState,
    line: &[Action],
    seed: u64,
    keying: Keying,
    max_turns: u32,
) -> Result<Rollout, IllegalAction> {
    let mut state = world(start, seed, keying.shuffles);
    for &action in line {
        if winner(&state).is_some() {
            break;
        }
        state = advance(&state, action)?;
    }
    let mut bots = [1, 2].map(|seat: u64| {
        let key = mix(seed ^ seat);
        PlannerBot::new(policy.value, policy.worlds, policy.budget, key)
            .with_keyed_worlds(keying.worlds.then_some(key))
    });
    while winner(&state).is_none() && state.turn <= max_turns {
        let actions = legal_actions(&state);
        let action = bots[state.current].choose(&observe(&state, state.current), &actions);
        apply_legal(&mut state, action);
        auto_play(&mut state);
    }
    Ok(Rollout {
        winner: winner(&state),
        turns: state.turn,
    })
}

/// Every line of `lines` rolled out with every seed of `seeds`, on
/// `threads` threads: the results line by line, each line's in `seeds`'
/// order, the same with any number of threads.
pub fn rollouts(
    policy: Policy<'_>,
    start: &GameState,
    lines: &[Vec<Action>],
    seeds: &[u64],
    keying: Keying,
    threads: usize,
    max_turns: u32,
) -> Result<Vec<Rollout>, IllegalAction> {
    let jobs: Vec<(usize, u64)> = (0..lines.len())
        .flat_map(|l| seeds.iter().map(move |&s| (l, s)))
        .collect();
    let next = AtomicUsize::new(0);
    let done: Vec<Vec<(usize, Result<Rollout, IllegalAction>)>> = thread::scope(|scope| {
        let workers: Vec<_> = (0..threads.clamp(1, jobs.len().max(1)))
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&(line, seed)) = jobs.get(i) else {
                            return done;
                        };
                        let found = rollout(policy, start, &lines[line], seed, keying, max_turns);
                        done.push((i, found));
                    }
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|w| w.join().expect("a rollout thread panicked"))
            .collect()
    });
    let mut results: Vec<Option<Rollout>> = vec![None; jobs.len()];
    for (i, found) in done.into_iter().flatten() {
        results[i] = Some(found?);
    }
    Ok(results
        .into_iter()
        .map(|r| r.expect("every rollout played"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::super::bots::greedy_move;
    use super::super::features;
    use super::super::planner::merged;
    use super::super::rules::new_game;
    use super::super::value::Form;
    use super::*;

    fn small_value() -> Value {
        let columns = features::columns();
        let w = (0..columns)
            .map(|i| 0.02 * ((i * 7 % 11) as f64 - 5.0))
            .collect();
        Value::new(
            vec![0.0; columns],
            vec![1.0; columns],
            Form::Logistic { w, b: 0.0 },
        )
        .unwrap()
    }

    /// A greedy game's state at the start of turn `turn`.
    fn midgame(seed: u64, turn: u32) -> GameState {
        let mut state = new_game(seed);
        auto_play(&mut state);
        while state.turn < turn {
            let action = greedy_move(state.must_scrap, &legal_actions(&state));
            apply_legal(&mut state, action);
            auto_play(&mut state);
        }
        state
    }

    #[test]
    fn the_shortlist_s_options_end_the_turn_differently() {
        let value = small_value();
        let policy = Policy {
            value: &value,
            worlds: 1,
            budget: 4_000,
        };
        let mut checked = 0;
        for seed in 0..6 {
            let start = midgame(seed, 11);
            let options = shortlist(policy, &start, 9);
            // the planner's one world, sampled as `shortlist` samples it
            let world = sample_world(&observe(&start, start.current), &mut Random::new(9));
            let mut ends: Vec<GameState> = Vec::new();
            for Ranked { line, .. } in &options {
                let mut state = world.clone();
                for &action in line {
                    if action == Action::EndTurn {
                        apply_legal(&mut state, action);
                    } else {
                        state = advance(&state, action).expect("a legal line");
                    }
                }
                if line.last() == Some(&Action::EndTurn) {
                    assert!(ends.iter().all(|e| !merged(e, &state)), "{line:?}");
                    ends.push(state);
                }
            }
            checked += ends.len();
        }
        assert!(checked > 12, "only {checked} ends checked");
    }
}
