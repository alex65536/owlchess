use crate::types::Coord;
use derive_more::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};
use std::iter::IntoIterator;

#[derive(
    Debug,
    Copy,
    Clone,
    PartialEq,
    Eq,
    Hash,
    BitAnd,
    BitAndAssign,
    BitOr,
    BitOrAssign,
    BitXor,
    BitXorAssign,
    Not,
)]
pub struct Bitboard(u64);

impl Bitboard {
    pub const EMPTY: Bitboard = Bitboard(0);

    pub fn from_raw(val: u64) -> Bitboard {
        Bitboard(val)
    }

    pub fn from_coord(coord: Coord) -> Bitboard {
        Bitboard(1_u64 << coord.index())
    }

    pub fn with(self, coord: Coord) -> Bitboard {
        Bitboard(self.0 | (1_u64 << coord.index()))
    }

    pub fn without(self, coord: Coord) -> Bitboard {
        Bitboard(self.0 & !(1_u64 << coord.index()))
    }

    pub fn as_raw(&self) -> u64 {
        self.0
    }

    pub fn popcount(&self) -> u32 {
        self.0.count_ones()
    }

    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }
}

impl From<Bitboard> for u64 {
    fn from(b: Bitboard) -> u64 {
        b.0
    }
}

pub struct Iter(u64);

impl Iterator for Iter {
    type Item = Coord;

    fn next(&mut self) -> Option<Coord> {
        if self.0 == 0 {
            return None;
        }
        let bit = self.0.trailing_zeros();
        self.0 &= self.0.wrapping_sub(1_u64);
        dbg!(bit);
        unsafe { Some(Coord::from_index_unchecked(bit as u8)) }
    }
}

impl IntoIterator for Bitboard {
    type Item = Coord;
    type IntoIter = Iter;

    fn into_iter(self) -> Iter {
        Iter(self.0)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::types::{Coord, File, Rank};

    #[test]
    fn test_iter() {
        let bb = Bitboard::EMPTY
            .with(Coord::from_parts(File::A, Rank::R4))
            .with(Coord::from_parts(File::E, Rank::R2))
            .with(Coord::from_parts(File::F, Rank::R3));
        assert_eq!(
            bb.into_iter().collect::<Vec<_>>(),
            vec![
                Coord::from_parts(File::A, Rank::R4),
                Coord::from_parts(File::F, Rank::R3),
                Coord::from_parts(File::E, Rank::R2)
            ],
        );
    }

    #[test]
    fn test_bitops() {
        let ca = Coord::from_parts(File::A, Rank::R4);
        let cb = Coord::from_parts(File::E, Rank::R2);
        let cc = Coord::from_parts(File::F, Rank::R3);

        let bb1 = Bitboard::EMPTY.with(ca).with(cb);
        let bb2 = Bitboard::EMPTY.with(cb).with(cc);
        assert_eq!(bb1 & bb2, Bitboard::EMPTY.with(cb));
        assert_eq!(bb1 | bb2, Bitboard::EMPTY.with(ca).with(cb).with(cc));
        assert_eq!(bb1 ^ bb2, Bitboard::EMPTY.with(ca).with(cc));

        assert_eq!((!bb1).into_iter().count(), 62);
        assert_eq!((!bb1).popcount(), 62);
    }
}
