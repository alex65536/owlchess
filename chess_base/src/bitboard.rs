use crate::types::Coord;
use derive_more::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};
use std::fmt;
use std::iter::IntoIterator;

#[derive(
    Default,
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

    pub const fn from_raw(val: u64) -> Bitboard {
        Bitboard(val)
    }

    pub const fn from_coord(coord: Coord) -> Bitboard {
        Bitboard(1_u64 << coord.index())
    }

    pub const fn with(self, coord: Coord) -> Bitboard {
        Bitboard(self.0 | (1_u64 << coord.index()))
    }

    pub const fn without(self, coord: Coord) -> Bitboard {
        Bitboard(self.0 & !(1_u64 << coord.index()))
    }

    pub const fn shl(self, by: usize) -> Bitboard {
        Bitboard(self.0 << by)
    }

    pub const fn shr(self, by: usize) -> Bitboard {
        Bitboard(self.0 >> by)
    }

    pub fn deposit_bits(&self, mut x: u64) -> Bitboard {
        let mut res: u64 = 0;
        let mut msk = self.0;
        while msk != 0 {
            let bit = msk & msk.wrapping_neg();
            if (x & 1) != 0 {
                res |= bit;
            }
            msk ^= bit;
            x >>= 1;
        }
        Bitboard(res)
    }

    pub fn set(&mut self, coord: Coord) {
        *self = self.with(coord);
    }

    pub fn unset(&mut self, coord: Coord) {
        *self = self.without(coord);
    }

    pub const fn has(&self, coord: Coord) -> bool {
        ((self.0 >> coord.index()) & 1) != 0
    }

    pub const fn as_raw(&self) -> u64 {
        self.0
    }

    pub const fn popcount(&self) -> u32 {
        self.0.count_ones()
    }

    pub const fn is_empty(&self) -> bool {
        self.0 == 0
    }

    pub const fn is_nonempty(&self) -> bool {
        self.0 != 0
    }
}

impl From<Bitboard> for u64 {
    fn from(b: Bitboard) -> u64 {
        b.0
    }
}

impl From<u64> for Bitboard {
    fn from(u: u64) -> Bitboard {
        Bitboard(u)
    }
}

impl fmt::Debug for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "Bitboard({})", self)
    }
}

impl fmt::Display for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        let v = self.0.reverse_bits();
        write!(
            f,
            "{:08b}/{:08b}/{:08b}/{:08b}/{:08b}/{:08b}/{:08b}/{:08b}",
            (v >> 56) & 0xff,
            (v >> 48) & 0xff,
            (v >> 40) & 0xff,
            (v >> 32) & 0xff,
            (v >> 24) & 0xff,
            (v >> 16) & 0xff,
            (v >> 8) & 0xff,
            v & 0xff,
        )
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
        unsafe { Some(Coord::from_index_unchecked(bit as usize)) }
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
mod tests {
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

    #[test]
    fn test_format() {
        let bb = Bitboard::EMPTY
            .with(Coord::from_parts(File::A, Rank::R4))
            .with(Coord::from_parts(File::E, Rank::R2))
            .with(Coord::from_parts(File::F, Rank::R3))
            .with(Coord::from_parts(File::H, Rank::R8));
        assert_eq!(
            bb.to_string(),
            "00000001/00000000/00000000/00000000/10000000/00000100/00001000/00000000"
        );
    }
}
