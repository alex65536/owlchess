use crate::types::{CastlingRights, CastlingSide, Cell, Color, Coord};

include!(concat!(env!("OUT_DIR"), "/zobrist.rs"));

pub fn pieces(cell: Cell, coord: Coord) -> u64 {
    unsafe {
        *PIECES
            .get_unchecked(cell.index())
            .get_unchecked(coord.index())
    }
}

pub fn enpassant(coord: Coord) -> u64 {
    unsafe { *ENPASSANT.get_unchecked(coord.index()) }
}

pub fn castling(rights: CastlingRights) -> u64 {
    unsafe { *CASTLING.get_unchecked(rights.index()) }
}

pub fn castling_delta(color: Color, side: CastlingSide) -> u64 {
    match side {
        CastlingSide::Queen => unsafe { *CASTLING_QUEENSIDE.get_unchecked(color as u8 as usize) },
        CastlingSide::King => unsafe { *CASTLING_KINGSIDE.get_unchecked(color as u8 as usize) },
    }
}
