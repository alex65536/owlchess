//! # Base types for owlchess
//!
//! This is an auxiliary crate for `owlchess`, which contains some core stuff. It was split from the main crate,
//! so everything declared here can be used in the build script for `owlchess`.
//!
//! Normally you don't want to use this crate directly. Use [`owlchess`](https://crates.io/crates/owlchess) instead.

pub mod bitboard;
pub mod bitboard_consts;
pub mod geometry;
pub mod types;
