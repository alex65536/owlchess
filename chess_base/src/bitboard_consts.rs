//! Simple and useful bitboard constants

use crate::bitboard::Bitboard;
use crate::types::{File, Rank};

/// Bitboards containing all the squares on a given diagonal
///
/// Note that diagonals go down-left.
///
/// The diagonals are numbered in the same way as in [`Coord::diag()`](crate::types::Coord::diag).
pub const DIAG: [Bitboard; 15] = [
    Bitboard::from_raw(0x0000000000000001),
    Bitboard::from_raw(0x0000000000000102),
    Bitboard::from_raw(0x0000000000010204),
    Bitboard::from_raw(0x0000000001020408),
    Bitboard::from_raw(0x0000000102040810),
    Bitboard::from_raw(0x0000010204081020),
    Bitboard::from_raw(0x0001020408102040),
    Bitboard::from_raw(0x0102040810204080),
    Bitboard::from_raw(0x0204081020408000),
    Bitboard::from_raw(0x0408102040800000),
    Bitboard::from_raw(0x0810204080000000),
    Bitboard::from_raw(0x1020408000000000),
    Bitboard::from_raw(0x2040800000000000),
    Bitboard::from_raw(0x4080000000000000),
    Bitboard::from_raw(0x8000000000000000),
];

/// Bitboards containing all the squares on a given antidiagonal
///
/// Note that antidiagonals go down-right.
///
/// The antidiagonals are numbered in the same way as in [`Coord::antidiag()`](crate::types::Coord::antidiag).
pub const ANTIDIAG: [Bitboard; 15] = [
    Bitboard::from_raw(0x0100000000000000),
    Bitboard::from_raw(0x0201000000000000),
    Bitboard::from_raw(0x0402010000000000),
    Bitboard::from_raw(0x0804020100000000),
    Bitboard::from_raw(0x1008040201000000),
    Bitboard::from_raw(0x2010080402010000),
    Bitboard::from_raw(0x4020100804020100),
    Bitboard::from_raw(0x8040201008040201),
    Bitboard::from_raw(0x0080402010080402),
    Bitboard::from_raw(0x0000804020100804),
    Bitboard::from_raw(0x0000008040201008),
    Bitboard::from_raw(0x0000000080402010),
    Bitboard::from_raw(0x0000000000804020),
    Bitboard::from_raw(0x0000000000008040),
    Bitboard::from_raw(0x0000000000000080),
];

const RANK: [Bitboard; 8] = [
    Bitboard::from_raw(0x00000000000000ff),
    Bitboard::from_raw(0x000000000000ff00),
    Bitboard::from_raw(0x0000000000ff0000),
    Bitboard::from_raw(0x00000000ff000000),
    Bitboard::from_raw(0x000000ff00000000),
    Bitboard::from_raw(0x0000ff0000000000),
    Bitboard::from_raw(0x00ff000000000000),
    Bitboard::from_raw(0xff00000000000000),
];

/// Returns a bitboard containing all the squares on a given rank `r`
#[inline]
pub const fn rank(r: Rank) -> Bitboard {
    RANK[r.index()]
}

const FILE: [Bitboard; 8] = [
    Bitboard::from_raw(0x0101010101010101),
    Bitboard::from_raw(0x0202020202020202),
    Bitboard::from_raw(0x0404040404040404),
    Bitboard::from_raw(0x0808080808080808),
    Bitboard::from_raw(0x1010101010101010),
    Bitboard::from_raw(0x2020202020202020),
    Bitboard::from_raw(0x4040404040404040),
    Bitboard::from_raw(0x8080808080808080),
];

/// Returns a bitboard containing all the squares on a given file `f`
#[inline]
pub const fn file(f: File) -> Bitboard {
    FILE[f.index()]
}

/// Bitboard containing all the light squares on the board
pub const LIGHT_SQUARES: Bitboard = Bitboard::from_raw(0xaa55aa55aa55aa55);

/// Bitboard containing all the dark squares on the board
pub const DARK_SQUARES: Bitboard = Bitboard::from_raw(0x55aa55aa55aa55aa);
