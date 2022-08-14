//! # Yet another Rust chess library
//!
//! ## 🦉🦉🦉
//!
//! Owlchess is a chess library written in Rust, with emphasis on both speed and safety.
//! Primarily designed for various chess GUIs and tools, it's also possible to use Owlchess
//! to build a fast chess engine.
//!
//! The code is mostly derived from my chess engine [SoFCheck](https://github.com/alex65536/sofcheck),
//! but rewritten in Rust with regard to safety.
//!
//! This crate supports core chess functionality:
//!
//! - generate moves
//! - make moves
//! - calculate game outcome
//! - parse and format boards in [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation)
//! - parse and format moves in UCI and [SAN](https://en.wikipedia.org/wiki/Algebraic_notation_\(chess\))
//!
//! ## Example
//!
//! ```
//! use owlchess::{Board, movegen::legal, Move};
//! use owlchess::{Coord, File, Rank};
//!
//! // Create a board with initial position
//! let b = Board::initial();
//!
//! // Generate all the legal moves
//! let moves = legal::gen_all(&b);
//! assert_eq!(moves.len(), 20);
//!
//! // Parse move "e2e4" from UCI string
//! let mv = Move::from_uci("e2e4", &b).unwrap();
//! let e2 = Coord::from_parts(File::E, Rank::R2);
//! let e4 = Coord::from_parts(File::E, Rank::R4);
//! assert_eq!(mv.src(), e2);
//! assert_eq!(mv.dst(), e4);
//!
//! // Create a new board with made move `mv`
//! let b = b.make_move(mv).unwrap();
//! assert_eq!(b.as_fen(), "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
//! ```

pub use owlchess_base::bitboard;
pub use owlchess_base::types;

pub mod board;
pub mod chain;
pub mod movegen;
pub mod moves;

#[cfg(feature = "selftest")]
pub mod selftest;

use owlchess_base::bitboard_consts;
use owlchess_base::geometry;

mod attack;
mod between;
mod castling;
mod generic;
mod legal;
mod pawns;
mod zobrist;

pub use bitboard::Bitboard;
pub use board::{Board, RawBoard};
pub use chain::MoveChain;
pub use movegen::{MoveList, MovePush};
pub use moves::{Make, Move, MoveKind};
pub use types::{
    CastlingRights, CastlingSide, Cell, Color, Coord, DrawReason, File, GameStatus, Outcome, Piece,
    Rank, WinReason,
};
