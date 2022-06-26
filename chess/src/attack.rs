use crate::bitboard::Bitboard;
use crate::types::{Color, Coord};

#[inline]
const fn bb(val: u64) -> Bitboard {
    Bitboard::from_raw(val)
}

include!(concat!(env!("OUT_DIR"), "/near_attacks.rs"));

struct MagicEntry {
    mask: Bitboard,
    post_mask: Bitboard,
    lookup: *const Bitboard,
}

unsafe impl Sync for MagicEntry {}

include!(concat!(env!("OUT_DIR"), "/magic.rs"));

#[inline]
pub fn king(coord: Coord) -> Bitboard {
    unsafe { *KING_ATTACKS.get_unchecked(coord.index()) }
}

#[inline]
pub fn knight(coord: Coord) -> Bitboard {
    unsafe { *KNIGHT_ATTACKS.get_unchecked(coord.index()) }
}

#[inline]
pub fn pawn(color: Color, coord: Coord) -> Bitboard {
    match color {
        Color::White => unsafe { *WHITE_PAWN_ATTACKS.get_unchecked(coord.index()) },
        Color::Black => unsafe { *BLACK_PAWN_ATTACKS.get_unchecked(coord.index()) },
    }
}

#[inline]
pub fn rook(coord: Coord, occupied: Bitboard) -> Bitboard {
    unsafe {
        let entry = MAGIC_ROOK.get_unchecked(coord.index());
        let magic = *MAGIC_CONSTS_ROOK.get_unchecked(coord.index());
        let shift = *MAGIC_SHIFTS_ROOK.get_unchecked(coord.index());
        let idx = (occupied & entry.mask).as_raw().wrapping_mul(magic) >> shift;
        *entry.lookup.add(idx as usize) & entry.post_mask
    }
}

#[inline]
pub fn bishop(coord: Coord, occupied: Bitboard) -> Bitboard {
    unsafe {
        let entry = MAGIC_BISHOP.get_unchecked(coord.index());
        let magic = *MAGIC_CONSTS_BISHOP.get_unchecked(coord.index());
        let shift = *MAGIC_SHIFTS_BISHOP.get_unchecked(coord.index());
        let idx = (occupied & entry.mask).as_raw().wrapping_mul(magic) >> shift;
        *entry.lookup.add(idx as usize) & entry.post_mask
    }
}
