//! # 🦉🦉🦉

pub use owlchess_base::bitboard;
pub use owlchess_base::types;

pub mod board;
pub mod chain;
pub mod movegen;
pub mod moves;

use owlchess_base::bitboard_consts;
use owlchess_base::geometry;

mod attack;
mod castling;
mod generic;
mod pawns;
mod zobrist;

pub use bitboard::Bitboard;
pub use board::{Board, RawBoard};
pub use chain::MoveChain;
pub use movegen::{MoveList, MovePush};
pub use moves::{Move, MoveKind, PromoteKind};
pub use types::{
    CastlingRights, CastlingSide, Cell, Color, Coord, DrawKind, File, Outcome, Piece, Rank, WinKind,
};
