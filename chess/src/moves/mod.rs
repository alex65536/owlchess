mod moves;

pub mod uci;

pub use moves::*;

pub type UciMove = uci::Move;
