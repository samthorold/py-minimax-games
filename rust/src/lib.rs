//! Compiled game engines for `minimax_games`, built by maturin as the
//! extension module `minimax_games._native` (docs/star-realms.md, "Compiled
//! engine"). The engines are plain Rust, tested by `cargo test`; the Python
//! boundary is behind the `python` feature, which maturin turns on.

pub mod star_realms;

#[cfg(feature = "python")]
mod python;
