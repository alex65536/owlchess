//! Moves and related stuff

mod base;

pub mod san;
pub mod uci;

pub use base::*;

/// Parsed move in SAN format
///
/// This is a convenience alias for [`san::Move`].
pub type SanMove = san::Move;

/// Parsed move in UCI format
///
/// This is a convenience alias for [`uci::Move`].
pub type UciMove = uci::Move;
