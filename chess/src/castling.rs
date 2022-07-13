use crate::bitboard::Bitboard;
use crate::types::{CastlingSide, Color};

#[inline]
pub const fn offset(c: Color) -> usize {
    match c {
        Color::White => 56,
        Color::Black => 0,
    }
}

#[inline]
pub const fn pass(c: Color, s: CastlingSide) -> Bitboard {
    let x = match s {
        CastlingSide::King => 0x60,
        CastlingSide::Queen => 0x0e,
    };
    Bitboard::from_raw(x << offset(c))
}

#[inline]
pub const fn srcs(c: Color, s: CastlingSide) -> Bitboard {
    let x = match s {
        CastlingSide::King => 0x90,
        CastlingSide::Queen => 0x11,
    };
    Bitboard::from_raw(x << offset(c))
}

pub const ALL_SRCS: Bitboard = Bitboard::from_raw(0x91 | (0x91 << 56));
