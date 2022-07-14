use crate::bitboard::Bitboard;
use crate::types::Coord;

#[inline]
const fn bb(val: u64) -> Bitboard {
    Bitboard::from_raw(val)
}

include!(concat!(env!("OUT_DIR"), "/between.rs"));

#[inline]
fn sort(src: Coord, dst: Coord) -> (Coord, Coord) {
    if src.index() < dst.index() {
        (src, dst)
    } else {
        (dst, src)
    }
}

#[inline]
pub fn bishop_strict(src: Coord, dst: Coord) -> Bitboard {
    let (src, dst) = sort(src, dst);
    unsafe { *BISHOP_GT.get_unchecked(src.index()) & *BISHOP_LT.get_unchecked(dst.index()) }
}

#[inline]
pub fn rook_strict(src: Coord, dst: Coord) -> Bitboard {
    let (src, dst) = sort(src, dst);
    unsafe { *ROOK_GT.get_unchecked(src.index()) & *ROOK_LT.get_unchecked(dst.index()) }
}

#[inline]
pub fn is_bishop_valid(src: Coord, dst: Coord) -> bool {
    unsafe { BISHOP_NE.get_unchecked(src.index()).has(dst) }
}

#[inline]
pub fn is_rook_valid(src: Coord, dst: Coord) -> bool {
    unsafe { ROOK_NE.get_unchecked(src.index()).has(dst) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Coord, File, Rank};

    #[test]
    fn test_bishop() {
        let b4 = Coord::from_parts(File::B, Rank::R4);
        let e7 = Coord::from_parts(File::E, Rank::R7);
        let res = Bitboard::EMPTY
            .with2(File::C, Rank::R5)
            .with2(File::D, Rank::R6);
        assert_eq!(bishop_strict(b4, e7), res);
        assert_eq!(bishop_strict(e7, b4), res);

        let f3 = Coord::from_parts(File::F, Rank::R3);
        let c6 = Coord::from_parts(File::C, Rank::R6);
        let res = Bitboard::EMPTY
            .with2(File::E, Rank::R4)
            .with2(File::D, Rank::R5);
        assert_eq!(bishop_strict(f3, c6), res);
        assert_eq!(bishop_strict(c6, f3), res);
    }

    #[test]
    fn test_rook() {
        let b4 = Coord::from_parts(File::B, Rank::R4);
        let e4 = Coord::from_parts(File::E, Rank::R4);
        let res = Bitboard::EMPTY
            .with2(File::C, Rank::R4)
            .with2(File::D, Rank::R4);
        assert_eq!(rook_strict(b4, e4), res);
        assert_eq!(rook_strict(e4, b4), res);

        let d3 = Coord::from_parts(File::D, Rank::R3);
        let d6 = Coord::from_parts(File::D, Rank::R6);
        let res = Bitboard::EMPTY
            .with2(File::D, Rank::R4)
            .with2(File::D, Rank::R5);
        assert_eq!(rook_strict(d3, d6), res);
        assert_eq!(rook_strict(d6, d3), res);
    }
}
