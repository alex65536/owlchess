use crate::types::{Color, Rank};

#[inline]
pub const fn castling_rank(c: Color) -> Rank {
    match c {
        Color::White => Rank::R1,
        Color::Black => Rank::R8,
    }
}

#[inline]
pub const fn double_move_src_rank(c: Color) -> Rank {
    match c {
        Color::White => Rank::R2,
        Color::Black => Rank::R7,
    }
}

#[inline]
pub const fn double_move_dst_rank(c: Color) -> Rank {
    match c {
        Color::White => Rank::R4,
        Color::Black => Rank::R5,
    }
}

#[inline]
pub const fn promote_src_rank(c: Color) -> Rank {
    match c {
        Color::White => Rank::R7,
        Color::Black => Rank::R2,
    }
}

#[inline]
pub const fn promote_dst_rank(c: Color) -> Rank {
    match c {
        Color::White => Rank::R8,
        Color::Black => Rank::R1,
    }
}

#[inline]
pub const fn enpassant_src_rank(c: Color) -> Rank {
    match c {
        Color::White => Rank::R5,
        Color::Black => Rank::R4,
    }
}

#[inline]
pub const fn enpassant_dst_rank(c: Color) -> Rank {
    match c {
        Color::White => Rank::R6,
        Color::Black => Rank::R3,
    }
}

#[inline]
pub const fn pawn_forward_delta(c: Color) -> isize {
    match c {
        Color::White => -8,
        Color::Black => 8,
    }
}

#[inline]
pub const fn pawn_left_delta(c: Color) -> isize {
    match c {
        Color::White => -9,
        Color::Black => 7,
    }
}

#[inline]
pub const fn pawn_right_delta(c: Color) -> isize {
    match c {
        Color::White => -7,
        Color::Black => 9,
    }
}
