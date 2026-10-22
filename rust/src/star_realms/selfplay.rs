//! Self-play: whole seeded games between the planner with a learned value,
//! the heuristic and greedy, played across threads, each game giving the
//! feature rows of its turn-boundary states and its result (`selfplay.py`
//! is the Python side).
//!
//! A game is played as `fit.boundaries` plays it (and `match.play_match`,
//! but stopping unfinished past `max_turns`): deal and auto-play, then
//! until someone wins ask the current player's bot for a decision from its
//! view, apply it, keep the state if it ended the turn (before the next
//! player's automatic play), and auto-play. Seat 0's bot is seeded with the
//! game's seed and seat 1's with the next, as the tournament seeds them.
//!
//! Each game depends only on its seed and players, and the games come back
//! in the order asked whatever thread played them, so a batch's results are
//! the same with any number of threads.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use super::bots::greedy_move;
use super::features;
use super::heuristic::heuristic_move;
use super::planner::{Note, PlannerBot, Replies};
use super::rules::{Action, apply_legal, auto_play, legal_actions, new_game, winner};
use super::value::Value;
use super::view::observe;

/// Who plays a seat.
#[derive(Clone, Debug)]
pub enum Player {
    Greedy,
    Heuristic,
    /// The whole-turn planner (`PlannerBot`) scoring with `value`, looking
    /// through the opponent's replies if `replies` is given.
    Planner {
        value: Value,
        worlds: usize,
        budget: usize,
        replies: Option<Replies>,
    },
}

enum Bot<'a> {
    Greedy,
    Heuristic,
    Planner(Box<PlannerBot<'a>>),
}

impl<'a> Bot<'a> {
    fn new(player: &'a Player, seed: u64) -> Self {
        match player {
            Player::Greedy => Bot::Greedy,
            Player::Heuristic => Bot::Heuristic,
            Player::Planner {
                value,
                worlds,
                budget,
                replies,
            } => Bot::Planner(Box::new(
                PlannerBot::new(value, *worlds, *budget, seed).with_replies(*replies),
            )),
        }
    }

    fn examined(&self) -> &[(u32, bool)] {
        match self {
            Bot::Planner(bot) => &bot.examined,
            Bot::Greedy | Bot::Heuristic => &[],
        }
    }

    fn noted(&self) -> &[(u32, usize, Note)] {
        match self {
            Bot::Planner(bot) => &bot.noted,
            Bot::Greedy | Bot::Heuristic => &[],
        }
    }

    fn seconds(&self) -> f64 {
        match self {
            Bot::Planner(bot) => bot.seconds,
            Bot::Greedy | Bot::Heuristic => 0.0,
        }
    }

    fn plans(&self) -> (u64, u64) {
        match self {
            Bot::Planner(bot) => (bot.plans, bot.diverged),
            Bot::Greedy | Bot::Heuristic => (0, 0),
        }
    }
}

/// What a game records beyond its rows and result: each is off by default,
/// and none changes a game.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Record {
    /// Every decision, in order.
    pub actions: bool,
    /// The judgement of each turn-boundary state by the player who ended
    /// the turn: its value's win probability for the state's row, NaN if
    /// that player doesn't judge (greedy and the heuristic).
    pub judgements: bool,
    /// What the diagnostics need: at each turn-boundary state, the cards
    /// each player has left to draw before their discard pile is
    /// reshuffled; and each plan's turn and whether it examined every end
    /// of the turn.
    pub diagnostics: bool,
    /// Each plan's turn, seat, states and what looking through the replies
    /// came to (`Note`, the default for a plan that didn't), and each seat's
    /// time spent planning.
    pub replies: bool,
}

/// One plan's record (`Record::replies`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanNote {
    pub turn: u32,
    pub seat: usize,
    /// States the plan visited, its replies' included.
    pub states: usize,
    pub note: Note,
}

/// A game to play: its seed and the players of seats 0 and 1, as indices
/// into the batch's players.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Game {
    pub seed: u64,
    pub seats: [usize; 2],
}

/// How a game went.
#[derive(Clone, Debug, PartialEq)]
pub struct Played {
    /// The feature rows of its turn-boundary states, in order, one after
    /// another (`features::columns()` each).
    pub rows: Vec<f64>,
    /// Who had just ended their turn at each of those states.
    pub ended: Vec<usize>,
    /// The judgement of each of those states by whoever ended the turn, if
    /// asked for (`Record::judgements`).
    pub judgements: Vec<f64>,
    /// The cards left in the deck of the player who ended and of the next
    /// player at each of those states, if asked for (`Record::diagnostics`).
    pub left: Vec<[u32; 2]>,
    /// Each plan's turn, seat and whether it examined every end of the
    /// turn in every world, in the order made, if asked for
    /// (`Record::diagnostics`).
    pub examined: Vec<(u32, usize, bool)>,
    /// The winner, `None` if the game passed `max_turns`.
    pub winner: Option<usize>,
    pub turns: u32,
    /// Each seat's plans, and plans made again because the game went
    /// otherwise than planned (0 for a bot that doesn't plan).
    pub plans: [u64; 2],
    pub diverged: [u64; 2],
    /// Every decision, in order, if asked for (`Record::actions`).
    pub actions: Vec<Action>,
    /// Each plan's record, in the order made, if asked for
    /// (`Record::replies`).
    pub replies: Vec<PlanNote>,
    /// Each seat's seconds spent planning, if asked for (`Record::replies`):
    /// measured, so it varies by run.
    pub seconds: [f64; 2],
}

/// Play `game` among `players`.
pub fn play_game(players: &[Player], game: Game, max_turns: u32, record: Record) -> Played {
    let mut bots = [
        Bot::new(&players[game.seats[0]], game.seed),
        Bot::new(&players[game.seats[1]], game.seed.wrapping_add(1)),
    ];
    let columns = features::columns();
    let mut played = Played {
        rows: Vec::new(),
        ended: Vec::new(),
        judgements: Vec::new(),
        left: Vec::new(),
        examined: Vec::new(),
        winner: None,
        turns: 0,
        plans: [0; 2],
        diverged: [0; 2],
        actions: Vec::new(),
        replies: Vec::new(),
        seconds: [0.0; 2],
    };
    let mut scratch = Vec::new();
    let mut state = new_game(game.seed);
    auto_play(&mut state);
    while winner(&state).is_none() && state.turn <= max_turns {
        let actions = legal_actions(&state);
        let action = match &mut bots[state.current] {
            Bot::Greedy => greedy_move(state.must_scrap, &actions),
            Bot::Heuristic => heuristic_move(&observe(&state, state.current), &actions),
            Bot::Planner(bot) => bot.choose(&observe(&state, state.current), &actions),
        };
        if record.actions {
            played.actions.push(action);
        }
        apply_legal(&mut state, action);
        if action == Action::EndTurn {
            let start = played.rows.len();
            played.rows.resize(start + columns, 0.0);
            features::encode_into(&mut played.rows[start..], &state);
            let ended = 1 - state.current;
            played.ended.push(ended);
            if record.judgements {
                played.judgements.push(match &players[game.seats[ended]] {
                    Player::Planner { value, .. } => {
                        value.predict(&played.rows[start..], &mut scratch)
                    }
                    Player::Greedy | Player::Heuristic => f64::NAN,
                });
            }
            if record.diagnostics {
                let left = |seat: usize| state.players[seat].deck.len() as u32;
                played.left.push([left(ended), left(1 - ended)]);
            }
        }
        auto_play(&mut state);
    }
    played.winner = winner(&state);
    played.turns = state.turn;
    for (seat, bot) in bots.iter().enumerate() {
        (played.plans[seat], played.diverged[seat]) = bot.plans();
    }
    if record.diagnostics {
        // the seats' plans interleave as their turns do
        let mut examined: Vec<(u32, usize, bool)> = bots
            .iter()
            .enumerate()
            .flat_map(|(seat, bot)| {
                bot.examined()
                    .iter()
                    .map(move |&(turn, complete)| (turn, seat, complete))
            })
            .collect();
        examined.sort_by_key(|&(turn, ..)| turn);
        played.examined = examined;
    }
    if record.replies {
        let mut notes: Vec<PlanNote> = bots
            .iter()
            .enumerate()
            .flat_map(|(seat, bot)| {
                bot.noted()
                    .iter()
                    .map(move |&(turn, states, note)| PlanNote {
                        turn,
                        seat,
                        states,
                        note,
                    })
            })
            .collect();
        notes.sort_by_key(|n| n.turn);
        played.replies = notes;
        played.seconds = [bots[0].seconds(), bots[1].seconds()];
    }
    played
}

/// Play every game of `games` among `players` on `threads` threads (at
/// least 1), each taking the next unplayed game: the results in `games`'
/// order, the same with any number of threads.
pub fn play_games(
    players: &[Player],
    games: &[Game],
    threads: usize,
    max_turns: u32,
    record: Record,
) -> Vec<Played> {
    let next = AtomicUsize::new(0);
    let mut results: Vec<Option<Played>> = vec![None; games.len()];
    let done: Vec<Vec<(usize, Played)>> = thread::scope(|scope| {
        let workers: Vec<_> = (0..threads.clamp(1, games.len().max(1)))
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&game) = games.get(i) else {
                            return done;
                        };
                        done.push((i, play_game(players, game, max_turns, record)));
                    }
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|w| w.join().expect("a self-play thread panicked"))
            .collect()
    });
    for (i, played) in done.into_iter().flatten() {
        results[i] = Some(played);
    }
    results
        .into_iter()
        .map(|r| r.expect("every game played"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::heuristic::play_heuristic;
    use super::super::value::Form;
    use super::*;

    fn planner() -> Player {
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
        Player::Planner {
            value,
            worlds: 2,
            budget: 200,
            replies: None,
        }
    }

    #[test]
    fn a_heuristic_game_is_play_heuristic_s() {
        let players = [Player::Heuristic];
        for seed in 0..5 {
            let game = Game {
                seed,
                seats: [0, 0],
            };
            let played = play_game(&players, game, 1000, Record::default());
            let expected = play_heuristic(seed, 1000);
            assert_eq!(
                (played.winner, played.turns),
                (expected.winner, expected.turns)
            );
            assert_eq!(played.ended.len() * features::columns(), played.rows.len());
        }
    }

    #[test]
    fn self_play_notes_each_plan_s_replies() {
        let Player::Planner { value, .. } = planner() else {
            unreachable!()
        };
        let replying = Player::Planner {
            value: value.clone(),
            worlds: 2,
            budget: 200,
            replies: Some(Replies {
                candidates: 3,
                worlds: 2,
                budget: 60,
            }),
        };
        let players = [replying, planner()];
        let games: Vec<Game> = (0..4)
            .map(|seed| Game {
                seed,
                seats: [seed as usize % 2, 1 - seed as usize % 2],
            })
            .collect();
        let record = Record {
            replies: true,
            ..Record::default()
        };
        let played = play_games(&players, &games, 2, 300, record);
        let mut looked = 0;
        for (game, p) in games.iter().zip(&played) {
            let replier = game.seats.iter().position(|&s| s == 0).unwrap();
            for seat in 0..2 {
                let notes: Vec<_> = p.replies.iter().filter(|n| n.seat == seat).collect();
                assert_eq!(notes.len() as u64, p.plans[seat]);
                for n in notes {
                    assert!(n.states > 0);
                    if seat == replier {
                        assert!(n.note.looked <= 3);
                        looked += usize::from(n.note.looked >= 2);
                        assert!(n.states <= 200 + 3 * 2 * 60);
                    } else {
                        assert_eq!(n.note, Note::default());
                        assert!(n.states <= 200);
                    }
                }
            }
            assert!(p.seconds.iter().all(|&s| s > 0.0));
        }
        assert!(looked > 10, "only {looked} plans looked through replies");
    }

    #[test]
    fn threads_don_t_change_the_results() {
        let Player::Planner { value, .. } = planner() else {
            unreachable!()
        };
        let replying = Player::Planner {
            value,
            worlds: 2,
            budget: 200,
            replies: Some(Replies {
                candidates: 3,
                worlds: 2,
                budget: 40,
            }),
        };
        let players = [planner(), Player::Heuristic, Player::Greedy, replying];
        let games: Vec<Game> = (0..6)
            .map(|seed| Game {
                seed,
                seats: [seed as usize % 4, (seed as usize + 1) % 4],
            })
            .collect();
        let record = Record {
            actions: true,
            judgements: false,
            diagnostics: true,
            replies: true,
        };
        // the time spent planning is measured, so it is left out
        let untimed = |played: Vec<Played>| -> Vec<Played> {
            played
                .into_iter()
                .map(|p| Played {
                    seconds: [0.0; 2],
                    ..p
                })
                .collect()
        };
        let one = untimed(play_games(&players, &games, 1, 300, record));
        assert_eq!(untimed(play_games(&players, &games, 3, 300, record)), one);
        assert!(one.iter().any(|p| p.plans[0] > 0));
    }
}
