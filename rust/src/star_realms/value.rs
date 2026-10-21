//! The learned value's forward pass, `value.Model.predict` and
//! `value.ValueEvaluator` hand-written: the probability that the player who
//! has just ended their turn goes on to win, from a state's feature row
//! (`features.rs`), and the planner's score of an end of turn from it.
//!
//! The arithmetic is numpy's, operation for operation: standardise each
//! feature (`(x - mean) / std`), then a logistic regression (`z = x . w + b`)
//! or one hidden layer of tanh units (`z = tanh(x W1 + b1) . w2 + b2`), and
//! `sigmoid(z) = 0.5 * (1 + tanh(0.5 * z))`. Only the order of the sums in
//! the products differs (numpy's are blocked and vectorised), so the
//! probabilities agree with numpy's to the last bits, and the scores, rounded
//! to whole points out of `SPAN`, agree (a test checks both).

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

/// A fitted value (`value.Model`): `form` over standardised features.
#[derive(Clone, Debug, PartialEq)]
pub struct Value {
    mean: Vec<f64>,
    std: Vec<f64>,
    form: Form,
}

fn sigmoid(z: f64) -> f64 {
    0.5 * (1.0 + (0.5 * z).tanh())
}

impl Value {
    /// The value of `form` over features standardised by `mean` and `std`,
    /// or why the shapes don't fit together.
    pub fn new(mean: Vec<f64>, std: Vec<f64>, form: Form) -> Result<Value, String> {
        let columns = mean.len();
        if std.len() != columns {
            return Err(format!(
                "{columns} means but {} standard deviations",
                std.len()
            ));
        }
        match &form {
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
        Ok(Value { mean, std, form })
    }

    /// How many features a row has.
    pub fn columns(&self) -> usize {
        self.mean.len()
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
        let z = match &self.form {
            Form::Logistic { w, b } => scratch.iter().zip(w).map(|(x, w)| x * w).sum::<f64>() + b,
            Form::Mlp { w1, b1, w2, b2 } => {
                let columns = self.columns();
                let hidden = b1.len();
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
        sigmoid(z)
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
