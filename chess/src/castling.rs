use crate::bitboard::Bitboard;
use crate::types::{CastlingSide, Color};

#[inline]
pub const fn pass(c: Color, s: CastlingSide) -> Bitboard {
    let x = match s {
        CastlingSide::King => 0x60,
        CastlingSide::Queen => 0x0e,
    };
    Bitboard::from_raw(match c {
        Color::White => x << 56,
        Color::Black => x,
    })
}

#[inline]
pub const fn srcs(c: Color, s: CastlingSide) -> Bitboard {
    let x = match s {
        CastlingSide::King => 0x90,
        CastlingSide::Queen => 0x11,
    };
    Bitboard::from_raw(match c {
        Color::White => x << 56,
        Color::Black => x,
    })
}

pub const ALL_SRCS: Bitboard = Bitboard::from_raw(0x91 | (0x91 << 56));
