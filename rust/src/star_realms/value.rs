//! The learned value's forward pass, `value.Model.predict` and
//! `value.ValueEvaluator` hand-written: the probability that the player who
//! has just ended their turn goes on to win, from a state's feature row
//! (`features.rs`), and the planner's score of an end of turn from it.
//!
//! The arithmetic is numpy's, operation for operation: standardise each
//! feature (`(x - mean) / std`), then a logistic regression (`z = x . w + b`)
//! or one hidden layer of tanh units (`z = tanh(x W1 + b1) . w2 + b2`), and
//! `sigmoid(z) = 0.5 * (1 + tanh(0.5 * z))`. A model of several members (an
//! ensemble, `value.fit_ensemble`) shares one standardisation, and its
//! probability is the mean of its members', summed in member order and
//! divided by their count, as `Model.predict` does. Only the order of the sums in
//! the products differs (numpy's are blocked and vectorised), so the
//! probabilities agree with numpy's to the last bits, and the scores, rounded
//! to whole points out of `SPAN`, agree (a test checks both).
//!
//! A model may read the extra features too (`features.extra`, #173), hand-made
//! from a row's counts and appended to it before standardising, from tables of
//! the cards Python hands over (`Extra`). Their sums are of small integers and
//! each takes one division in numpy's order, so they are bit-identical.

use super::rules::winner;
use super::state::GameState;

/// `rollout.SPAN`: a probability maps to a score in `-SPAN..SPAN`.
pub const SPAN: f64 = 1_000_000.0;
/// `evaluate.WIN`: the score of a won game.
pub const WIN: i64 = 1_000_000_000;

/// A model's form and parameters (`value.Logistic`, `value.Mlp`).
#[derive(Clone, Debug, PartialEq)]
pub enum Form {
    Logistic {
        w: Vec<f64>,
        b: f64,
    },
    Mlp {
        /// `w1`, `columns` rows of `hidden` weights, row after row.
        w1: Vec<f64>,
        b1: Vec<f64>,
        w2: Vec<f64>,
        b2: f64,
    },
}

/// `features.FACTIONS`' count.
const FACTIONS: usize = 4;
/// `features.HAND`: the cards a hand draws.
const HAND: f64 = 5.0;
/// `len(features.EXTRA)`: per player, combat per hand and kill threat, then
/// per faction its density, pairs and bases.
pub const EXTRA_COLUMNS: usize = 2 * (2 + 3 * FACTIONS);
/// `len(features.MORE)`: per player trade and Authority per hand, the
/// draw, scrap, discard and base densities and the Outposts' defense, then
/// the Authority, combat and trade leads.
pub const MORE_COLUMNS: usize = 2 * 7 + 3;

/// What the more features read of the cards (`features.extra_tables(more=
/// True)["more"]`, #178), by catalogue index: each card's primary trade and
/// Authority, whether it draws, scraps, makes the opponent discard and is a
/// base (1 or 0), and by base index each Outpost's defense (0 for a base).
#[derive(Clone, Debug, PartialEq)]
pub struct More {
    pub trade: Vec<f64>,
    pub authority: Vec<f64>,
    pub draw: Vec<f64>,
    pub scrap: Vec<f64>,
    pub discard: Vec<f64>,
    pub base: Vec<f64>,
    pub outpost_defense: Vec<f64>,
}

/// What the extra features read of the cards (`features.extra_tables`), by
/// catalogue index: each card's primary combat, its faction (`None` for
/// none), and each base's card, and the more features' tables if the model
/// reads them. A row's layout follows from their lengths, as in
/// `features.rs`: per player a pool count per card, a count per base,
/// Authority, base defense and discards owed; then the turn and the Trade
/// Row.
#[derive(Clone, Debug, PartialEq)]
pub struct Extra {
    pub combat: Vec<f64>,
    pub faction: Vec<Option<usize>>,
    pub base_card: Vec<usize>,
    pub more: Option<Box<More>>,
}

/// `held` per hand of a pool of `n` cards, as numpy's `HAND * held / n`
/// where `n > 0`, else 0.
fn per_hand(held: f64, n: f64) -> f64 {
    if n > 0.0 { HAND * held / n } else { 0.0 }
}

fn density(count: f64, n: f64) -> f64 {
    if n > 0.0 { count / n } else { 0.0 }
}

fn dot(counts: &[f64], weights: &[f64]) -> f64 {
    counts.iter().zip(weights).map(|(c, w)| c * w).sum()
}

impl Extra {
    fn cards(&self) -> usize {
        self.combat.len()
    }

    /// How many extra features it appends to a row.
    pub fn columns(&self) -> usize {
        EXTRA_COLUMNS + self.more.as_ref().map_or(0, |_| MORE_COLUMNS)
    }

    /// Whether its tables fit together.
    fn fits(&self) -> bool {
        let cards = self.cards();
        let tables = self.faction.len() == cards
            && self.base_card.iter().all(|&c| c < cards)
            && self.faction.iter().flatten().all(|&f| f < FACTIONS);
        tables
            && self.more.as_ref().is_none_or(|m| {
                [
                    &m.trade,
                    &m.authority,
                    &m.draw,
                    &m.scrap,
                    &m.discard,
                    &m.base,
                ]
                .iter()
                .all(|t| t.len() == cards)
                    && m.outpost_defense.len() == self.base_card.len()
            })
    }

    /// `features._more`: the more features of `row`, given each side's
    /// combat per hand.
    fn push_more(&self, more: &More, row: &[f64], combat: [f64; 2], out: &mut Vec<f64>) {
        let cards = self.cards();
        let bases = self.base_card.len();
        let width = self.player_columns();
        let mut trade = [0.0; 2];
        for (side, offset) in [0, width].into_iter().enumerate() {
            let pool = &row[offset..offset + cards];
            let in_play = &row[offset + cards..offset + cards + bases];
            let n: f64 = pool.iter().sum();
            for (k, w) in [&more.trade, &more.authority].into_iter().enumerate() {
                let on_bases: f64 = in_play
                    .iter()
                    .zip(&self.base_card)
                    .map(|(k, &c)| k * w[c])
                    .sum();
                let value = per_hand(dot(pool, w), n) + on_bases;
                if k == 0 {
                    trade[side] = value;
                }
                out.push(value);
            }
            for w in [&more.draw, &more.scrap, &more.discard, &more.base] {
                out.push(density(dot(pool, w), n));
            }
            out.push(dot(in_play, &more.outpost_defense));
        }
        let authority = cards + bases;
        out.push(row[authority] - row[width + authority]);
        out.push(combat[0] - combat[1]);
        out.push(trade[0] - trade[1]);
    }

    fn player_columns(&self) -> usize {
        self.cards() + self.base_card.len() + 3
    }

    /// The length of the rows it reads.
    pub fn row_columns(&self) -> usize {
        2 * self.player_columns() + 1 + self.cards()
    }

    /// Per player (from `offset`): combat per hand, then per faction its
    /// density, pairs and bases in play.
    fn side(&self, row: &[f64], offset: usize) -> (f64, [[f64; 3]; FACTIONS]) {
        let cards = self.cards();
        let pool = &row[offset..offset + cards];
        let in_play = &row[offset + cards..offset + cards + self.base_card.len()];
        let n: f64 = pool.iter().sum();
        let held: f64 = pool.iter().zip(&self.combat).map(|(c, w)| c * w).sum();
        let mut per_hand = if n > 0.0 { HAND * held / n } else { 0.0 };
        per_hand += in_play
            .iter()
            .zip(&self.base_card)
            .map(|(k, &c)| k * self.combat[c])
            .sum::<f64>();
        let mut factions = [[0.0; 3]; FACTIONS];
        let pairs = n * (n - 1.0);
        for (f, out) in factions.iter_mut().enumerate() {
            let nf: f64 = pool
                .iter()
                .zip(&self.faction)
                .filter(|(_, faction)| **faction == Some(f))
                .map(|(c, _)| c)
                .sum();
            out[0] = if n > 0.0 { nf / n } else { 0.0 };
            out[1] = if pairs > 0.0 {
                HAND * (HAND - 1.0) / 2.0 * nf * (nf - 1.0) / pairs
            } else {
                0.0
            };
            out[2] = in_play
                .iter()
                .zip(&self.base_card)
                .filter(|(_, c)| self.faction[**c] == Some(f))
                .map(|(k, _)| k)
                .sum();
        }
        (per_hand, factions)
    }

    /// Append `row`'s extra features (`features.extra`) to `out`.
    pub fn push(&self, row: &[f64], out: &mut Vec<f64>) {
        let width = self.player_columns();
        let authority = self.cards() + self.base_card.len();
        let sides = [self.side(row, 0), self.side(row, width)];
        for (side, other) in [(0, width), (1, 0)] {
            let (per_hand, factions) = sides[side];
            let defended = row[other + authority] + row[other + authority + 1];
            out.push(per_hand);
            out.push(per_hand - defended);
            out.extend(factions.iter().flatten());
        }
        if let Some(more) = &self.more {
            self.push_more(more, row, [sides[0].0, sides[1].0], out);
        }
    }
}

/// A fitted value (`value.Model`): its members, each a form and its
/// parameters, over the same standardised features, and the extra features
/// it reads, if any.
#[derive(Clone, Debug, PartialEq)]
pub struct Value {
    mean: Vec<f64>,
    std: Vec<f64>,
    members: Vec<Form>,
    extra: Option<Extra>,
}

fn sigmoid(z: f64) -> f64 {
    0.5 * (1.0 + (0.5 * z).tanh())
}

impl Value {
    /// The value of `form` over features standardised by `mean` and `std`,
    /// or why the shapes don't fit together: a model of one member.
    pub fn new(mean: Vec<f64>, std: Vec<f64>, form: Form) -> Result<Value, String> {
        Value::ensemble(mean, std, vec![form])
    }

    /// The mean of `members`' values over features standardised by `mean`
    /// and `std`, or why the shapes don't fit together.
    pub fn ensemble(mean: Vec<f64>, std: Vec<f64>, members: Vec<Form>) -> Result<Value, String> {
        let columns = mean.len();
        if std.len() != columns {
            return Err(format!(
                "{columns} means but {} standard deviations",
                std.len()
            ));
        }
        if members.is_empty() {
            return Err("a model has at least 1 member".to_string());
        }
        for form in &members {
            match form {
                Form::Logistic { w, .. } if w.len() != columns => {
                    return Err(format!("{columns} features but {} weights", w.len()));
                }
                Form::Mlp { w1, b1, w2, .. } => {
                    let hidden = b1.len();
                    if w2.len() != hidden || w1.len() != columns * hidden {
                        return Err(format!(
                            "an mlp of {columns} features and {hidden} hidden units has a {columns}x{hidden} w1 and {hidden} w2, not {} and {}",
                            w1.len(),
                            w2.len()
                        ));
                    }
                }
                Form::Logistic { .. } => {}
            }
        }
        Ok(Value {
            mean,
            std,
            members,
            extra: None,
        })
    }

    /// The value reading the extra features `extra` (`features.extra`) of
    /// each row too, its standardisation and weights covering them after
    /// the row's own, or why the shapes don't fit together.
    pub fn with_extra(self, extra: Extra) -> Result<Value, String> {
        let inputs = self.mean.len();
        if extra.row_columns() + extra.columns() != inputs {
            return Err(format!(
                "a model of {inputs} inputs can't read rows of {} and {} extra features",
                extra.row_columns(),
                extra.columns()
            ));
        }
        if !extra.fits() {
            return Err("the extra features' card tables don't fit together".to_string());
        }
        Ok(Value {
            extra: Some(extra),
            ..self
        })
    }

    /// How many features a row has (the extra features are the model's own).
    pub fn columns(&self) -> usize {
        self.mean.len() - self.extra.as_ref().map_or(0, Extra::columns)
    }

    /// The win probability of the player who just ended, from `row`, raw
    /// features (`Model.predict` of one row). `scratch` is reused for the
    /// standardised row and the hidden layer.
    pub fn predict(&self, row: &[f64], scratch: &mut Vec<f64>) -> f64 {
        assert_eq!(row.len(), self.columns(), "a row of the model's features");
        scratch.clear();
        scratch.extend(
            row.iter()
                .zip(&self.mean)
                .zip(&self.std)
                .map(|((x, mean), std)| (x - mean) / std),
        );
        if let Some(extra) = &self.extra {
            let start = scratch.len();
            extra.push(row, scratch);
            for ((x, mean), std) in scratch[start..]
                .iter_mut()
                .zip(&self.mean[start..])
                .zip(&self.std[start..])
            {
                *x = (*x - mean) / std;
            }
        }
        let columns = self.mean.len();
        let mut total = 0.0;
        for form in &self.members {
            let z = match form {
                Form::Logistic { w, b } => {
                    scratch[..columns]
                        .iter()
                        .zip(w)
                        .map(|(x, w)| x * w)
                        .sum::<f64>()
                        + b
                }
                Form::Mlp { w1, b1, w2, b2 } => {
                    let hidden = b1.len();
                    scratch.truncate(columns);
                    scratch.resize(columns + hidden, 0.0);
                    let (x, h) = scratch.split_at_mut(columns);
                    for (i, &xi) in x.iter().enumerate() {
                        let weights = &w1[i * hidden..(i + 1) * hidden];
                        for (hj, w) in h.iter_mut().zip(weights) {
                            *hj += xi * w;
                        }
                    }
                    for (hj, b) in h.iter_mut().zip(b1) {
                        *hj = (*hj + b).tanh();
                    }
                    h.iter().zip(w2).map(|(h, w)| h * w).sum::<f64>() + b2
                }
            };
            total += sigmoid(z);
        }
        if self.members.len() == 1 {
            total
        } else {
            total / self.members.len() as f64
        }
    }

    /// `ValueEvaluator(model)(state, perspective)`: `WIN` or `-WIN` for a
    /// finished game, else the probability mapped to `-SPAN..SPAN` and
    /// rounded half to even, as Python's `round` does, from the side of the
    /// player who just ended, negated for the other. `row` is `state`'s
    /// feature row.
    pub fn score(
        &self,
        state: &GameState,
        row: &[f64],
        perspective: usize,
        scratch: &mut Vec<f64>,
    ) -> i64 {
        if let Some(won_by) = winner(state) {
            return if won_by == perspective { WIN } else { -WIN };
        }
        let p = self.predict(row, scratch);
        let score = (SPAN * (2.0 * p - 1.0)).round_ties_even() as i64;
        if perspective == 1 - state.current {
            score
        } else {
            -score
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_logistic_is_the_sigmoid_of_a_dot_product() {
        let value = Value::new(
            vec![1.0, 0.0],
            vec![2.0, 1.0],
            Form::Logistic {
                w: vec![0.5, -1.0],
                b: 0.25,
            },
        )
        .unwrap();
        let mut scratch = Vec::new();
        // z = (3 - 1) / 2 * 0.5 + 2 * -1 + 0.25 = -1.25
        let p = value.predict(&[3.0, 2.0], &mut scratch);
        assert!((p - 1.0 / (1.0 + 1.25f64.exp())).abs() < 1e-15);
    }

    #[test]
    fn an_mlp_sums_its_hidden_units() {
        // two features, two hidden units
        let value = Value::new(
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            Form::Mlp {
                w1: vec![1.0, 0.0, 0.0, 2.0],
                b1: vec![0.0, -1.0],
                w2: vec![1.0, 1.0],
                b2: 0.5,
            },
        )
        .unwrap();
        let mut scratch = Vec::new();
        let z = 0.3f64.tanh() + (2.0 * 0.4 - 1.0f64).tanh() + 0.5;
        for _ in 0..2 {
            let p = value.predict(&[0.3, 0.4], &mut scratch);
            assert!((p - sigmoid(z)).abs() < 1e-15);
        }
    }

    #[test]
    fn an_ensemble_is_the_mean_of_its_members() {
        let logistic = |w: f64| Form::Logistic { w: vec![w], b: 0.0 };
        let mlp = Form::Mlp {
            w1: vec![1.0, -1.0],
            b1: vec![0.0, 0.5],
            w2: vec![1.0, 2.0],
            b2: 0.0,
        };
        let one = |form: Form| Value::new(vec![0.0], vec![1.0], form).unwrap();
        let members = vec![logistic(0.5), mlp.clone(), logistic(-2.0)];
        let all = Value::ensemble(vec![0.0], vec![1.0], members).unwrap();
        let mut scratch = Vec::new();
        for x in [0.3, -1.2] {
            let expected = (one(logistic(0.5)).predict(&[x], &mut scratch)
                + one(mlp.clone()).predict(&[x], &mut scratch)
                + one(logistic(-2.0)).predict(&[x], &mut scratch))
                / 3.0;
            assert_eq!(all.predict(&[x], &mut scratch), expected);
        }
        assert!(Value::ensemble(vec![0.0], vec![1.0], vec![]).is_err());
    }

    /// Two cards (the first of faction 0 with 2 combat, the second a base of
    /// faction 1 with 3 combat), so a player's columns are 2 pool counts, 1
    /// base count, Authority, base defense and discards owed: rows of 15.
    fn tiny_extra() -> Extra {
        Extra {
            combat: vec![2.0, 3.0],
            faction: vec![Some(0), Some(1)],
            base_card: vec![1],
            more: None,
        }
    }

    #[test]
    fn extra_features_are_counted_from_the_row() {
        let extra = tiny_extra();
        assert_eq!(extra.row_columns(), 15);
        // ended: 3 of card 0 and 1 of card 1 in the pool, the base in play,
        // 40 Authority and 5 defense; next: 2 of card 0, 30 Authority
        let mut row = vec![0.0; 15];
        row[..6].copy_from_slice(&[3.0, 1.0, 1.0, 40.0, 5.0, 0.0]);
        row[6..12].copy_from_slice(&[2.0, 0.0, 0.0, 30.0, 0.0, 0.0]);
        let mut out = Vec::new();
        extra.push(&row, &mut out);
        assert_eq!(out.len(), EXTRA_COLUMNS);
        let ended_hand = 5.0 * 9.0 / 4.0 + 3.0;
        assert_eq!(out[0], ended_hand);
        assert_eq!(out[1], ended_hand - 30.0);
        // faction 0: density, pairs, bases; then faction 1
        assert_eq!(
            &out[2..8],
            &[0.75, 10.0 * 3.0 * 2.0 / 12.0, 0.0, 0.25, 0.0, 1.0]
        );
        assert_eq!(out[14], 5.0 * 4.0 / 2.0);
        assert_eq!(out[15], 10.0 - 45.0);
        assert_eq!(&out[16..19], &[1.0, 10.0 * 2.0 / 2.0, 0.0]);
    }

    /// `tiny_extra` with the more features' tables: card 0 gives 1 trade
    /// and draws, card 1 (the base, an Outpost of 4 defense) gives 2
    /// Authority, scraps and makes the opponent discard.
    fn tiny_more() -> Extra {
        Extra {
            more: Some(Box::new(More {
                trade: vec![1.0, 0.0],
                authority: vec![0.0, 2.0],
                draw: vec![1.0, 0.0],
                scrap: vec![0.0, 1.0],
                discard: vec![0.0, 1.0],
                base: vec![0.0, 1.0],
                outpost_defense: vec![4.0],
            })),
            ..tiny_extra()
        }
    }

    #[test]
    fn more_features_follow_the_extra_ones() {
        let extra = tiny_more();
        assert_eq!(extra.columns(), EXTRA_COLUMNS + MORE_COLUMNS);
        let mut row = vec![0.0; 15];
        row[..6].copy_from_slice(&[3.0, 1.0, 1.0, 40.0, 5.0, 0.0]);
        row[6..12].copy_from_slice(&[2.0, 0.0, 0.0, 30.0, 0.0, 0.0]);
        let mut out = Vec::new();
        extra.push(&row, &mut out);
        let mut first = Vec::new();
        tiny_extra().push(&row, &mut first);
        assert_eq!(out.len(), EXTRA_COLUMNS + MORE_COLUMNS);
        assert_eq!(&out[..EXTRA_COLUMNS], &first[..]);
        let more = &out[EXTRA_COLUMNS..];
        // ended: trade, Authority (and the base's 2), draw, scrap, discard,
        // base densities, outpost defense
        let ended = [
            5.0 * 3.0 / 4.0,
            5.0 * 2.0 / 4.0 + 2.0,
            0.75,
            0.25,
            0.25,
            0.25,
            4.0,
        ];
        assert_eq!(&more[..7], &ended);
        assert_eq!(&more[7..14], &[5.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        // the leads: Authority, combat per hand, trade per hand
        assert_eq!(&more[14..], &[10.0, first[0] - first[14], ended[0] - 5.0]);
        // a model reads them only if its inputs cover them
        let value = |inputs: usize| {
            Value::new(
                vec![0.0; inputs],
                vec![1.0; inputs],
                Form::Logistic {
                    w: vec![0.0; inputs],
                    b: 0.0,
                },
            )
            .unwrap()
        };
        assert!(value(15 + EXTRA_COLUMNS).with_extra(tiny_more()).is_err());
        let with = value(15 + EXTRA_COLUMNS + MORE_COLUMNS).with_extra(tiny_more());
        assert_eq!(with.unwrap().columns(), 15);
        let mut short = tiny_more();
        short.more.as_mut().unwrap().outpost_defense.clear();
        let refused = value(15 + EXTRA_COLUMNS + MORE_COLUMNS).with_extra(short);
        assert!(refused.is_err());
    }

    #[test]
    fn a_value_with_extra_features_reads_the_widened_row() {
        let inputs = 15 + EXTRA_COLUMNS;
        let mean: Vec<f64> = (0..inputs).map(|i| 0.1 * i as f64).collect();
        let std: Vec<f64> = (0..inputs).map(|i| 1.0 + 0.05 * i as f64).collect();
        let w: Vec<f64> = (0..inputs).map(|i| ((i % 7) as f64 - 3.0) * 0.1).collect();
        let form = Form::Logistic { w, b: 0.2 };
        let plain = Value::new(mean.clone(), std.clone(), form.clone()).unwrap();
        let with = Value::new(mean, std, form)
            .unwrap()
            .with_extra(tiny_extra())
            .unwrap();
        assert_eq!(with.columns(), 15);
        let row: Vec<f64> = (0..15).map(|i| (i % 4) as f64).collect();
        let mut widened = row.clone();
        tiny_extra().push(&row, &mut widened);
        let mut scratch = Vec::new();
        assert_eq!(
            with.predict(&row, &mut scratch),
            plain.predict(&widened, &mut scratch)
        );
        let short = Value::new(
            vec![0.0; 15],
            vec![1.0; 15],
            Form::Logistic {
                w: vec![0.0; 15],
                b: 0.0,
            },
        );
        assert!(short.unwrap().with_extra(tiny_extra()).is_err());
    }

    #[test]
    fn mismatched_shapes_are_refused() {
        let logistic = Form::Logistic {
            w: vec![1.0],
            b: 0.0,
        };
        assert!(Value::new(vec![0.0; 2], vec![1.0; 2], logistic).is_err());
        let mlp = Form::Mlp {
            w1: vec![0.0; 5],
            b1: vec![0.0; 2],
            w2: vec![0.0; 2],
            b2: 0.0,
        };
        assert!(Value::new(vec![0.0; 3], vec![1.0; 3], mlp).is_err());
    }
}
