//! Named ranks and offsets on the chess board

use crate::types::{Color, Rank};

/// Rank on which the pieces of color `c` stand while performing castling
#[inline]
pub const fn castling_rank(c: Color) -> Rank {
    match c {
        Color::White => Rank::R1,
        Color::Black => Rank::R8,
    }
}

/// Source rank for pawns of color `c` to perform a double move
#[inline]
pub const fn double_move_src_rank(c: Color) -> Rank {
    match c {
        Color::White => Rank::R2,
        Color::Black => Rank::R7,
    }
}

/// Destination rank for pawns of color `c` to perform a double move
#[inline]
pub const fn double_move_dst_rank(c: Color) -> Rank {
    match c {
        Color::White => Rank::R4,
        Color::Black => Rank::R5,
    }
}

/// Source rank for pawns of color `c` to perform a promote
#[inline]
pub const fn promote_src_rank(c: Color) -> Rank {
    match c {
        Color::White => Rank::R7,
        Color::Black => Rank::R2,
    }
}

/// Destination rank for pawns of color `c` to perform a promote
#[inline]
pub const fn promote_dst_rank(c: Color) -> Rank {
    match c {
        Color::White => Rank::R8,
        Color::Black => Rank::R1,
    }
}

/// Source rank for pawns of color `c` to perform an enpassant
///
/// Also the rank on which the opponent's pawn captured by enpassant is located.
#[inline]
pub const fn enpassant_src_rank(c: Color) -> Rank {
    match c {
        Color::White => Rank::R5,
        Color::Black => Rank::R4,
    }
}

/// Destination rank for pawns of color `c` to perform an enpassant
#[inline]
pub const fn enpassant_dst_rank(c: Color) -> Rank {
    match c {
        Color::White => Rank::R6,
        Color::Black => Rank::R3,
    }
}

/// Offset for pawn of color `c` when it performs a simple move
#[inline]
pub const fn pawn_forward_delta(c: Color) -> isize {
    match c {
        Color::White => -8,
        Color::Black => 8,
    }
}

/// Offset for pawn of color `c` when it performs a left capture
#[inline]
pub const fn pawn_left_delta(c: Color) -> isize {
    match c {
        Color::White => -9,
        Color::Black => 7,
    }
}

/// Offset for pawn of color `c` when it performs a right capture
#[inline]
pub const fn pawn_right_delta(c: Color) -> isize {
    match c {
        Color::White => -7,
        Color::Black => 9,
    }
}
