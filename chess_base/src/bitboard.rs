//! Bitboard-related stuff

use crate::types::Coord;
use derive_more::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};
use std::fmt;
use std::iter::IntoIterator;

/// Bitmask on chess board
///
/// Bitboard is just a set of squares on the chess board. It is represented as a single `u64`,
/// each bit of which corresponds to a single square on the board. The bits correspond to squares
/// in the same order as the squares are numbered. See [`Coord::index`](crate::types::Coord::index)
/// for details. So, the square with index `0` corresponds to the lowest bit, and the square with
/// index `63` corresponds to the highest bit.
///
/// As you can see, operations with bitboards are extremely fast and lightweight. Thus, bitboards
/// are extensively used, for example, in move generation or pattern detection.
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
    /// Empty bitboard, containing no squares
    pub const EMPTY: Bitboard = Bitboard(0);

    /// Full bitboard, containing all the squares
    pub const FULL: Bitboard = Bitboard(u64::MAX);

    /// Wraps a raw `u64` into bitboard
    ///
    /// This function does the same as [`Bitboard::from()`](#method.from-1), except that this one is `const`.
    #[inline]
    pub const fn from_raw(val: u64) -> Bitboard {
        Bitboard(val)
    }

    /// Makes a bitboard containing a single square `Coord`
    #[inline]
    pub const fn from_coord(coord: Coord) -> Bitboard {
        Bitboard(1_u64 << coord.index())
    }

    /// Adds square `coord` to the bitboard
    ///
    /// If the provided square is already present, the bitboard is returned unchanged.
    #[inline]
    pub const fn with(self, coord: Coord) -> Bitboard {
        Bitboard(self.0 | (1_u64 << coord.index()))
    }

    /// Removes square `coord` from the bitboard
    ///
    /// If the provided bitboard doesn't have square `coord`, it is returned unchanged.
    #[inline]
    pub const fn without(self, coord: Coord) -> Bitboard {
        Bitboard(self.0 & !(1_u64 << coord.index()))
    }

    /// Performs a left bitwise shift of the inner value
    #[inline]
    pub const fn shl(self, by: usize) -> Bitboard {
        Bitboard(self.0 << by)
    }

    /// Performs a right bitwise shift of the inner value
    #[inline]
    pub const fn shr(self, by: usize) -> Bitboard {
        Bitboard(self.0 >> by)
    }

    /// Creates a new bitboard from the lower bits of `x`, using `self` as mask
    ///
    /// The operation is similar to [PDEP](https://www.felixcloutier.com/x86/pdep) instruction
    /// on x86. The lowest bits from `x` are deposited to the places of ones in `self`. All other
    /// bits remain set to zero.
    ///
    /// Note that the implemetation currently uses good old loops and is not optimized to use
    /// PDEP instruction directly.
    ///
    /// # Example
    ///
    /// ```
    /// # use owlchess_base::bitboard::Bitboard;
    /// #
    /// let b = Bitboard::from(0b11011001);
    /// let x = 0b10110;
    /// // Suppose that the bits are numbered from 0 in order from lower to higher.
    /// // The 0th bit of `x` is placed to the 0th bit of `b`.
    /// // The 1st bit of `x` is placed to the 3rd bit of `b`.
    /// // The 2nd bit of `x` is placed to the 4th bit of `b`.
    /// // The 3rd bit of `x` is placed to the 6th bit of `b`.
    /// // The 4th bit of `x` is placed to the 7th bit of `b`.
    /// assert_eq!(b.deposit_bits(x), Bitboard::from(0b10011000));
    ///
    /// let x = 0b10110110;
    /// // `b` has only five bits set, so the bits from 5th to 7th in `x` are just ignored.
    /// assert_eq!(b.deposit_bits(x), Bitboard::from(0b10011000));
    /// ```
    #[inline]
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

    /// Adds square `coord` to the bitboard
    ///
    /// If the provided square is already present, the bitboard is unchanged.
    ///
    /// This function does the same as [`with()`](Bitboard::with), but mutates the bitboard instead
    /// of returning a new one.
    #[inline]
    pub fn set(&mut self, coord: Coord) {
        *self = self.with(coord);
    }

    /// Removes square `coord` from the bitboard
    ///
    /// If the provided bitboard doesn't have square `coord`, it is unchanged.
    ///
    /// This function does the same as [`without()`](Bitboard::without), but mutates the bitboard instead
    /// of returning a new one.
    #[inline]
    pub fn unset(&mut self, coord: Coord) {
        *self = self.without(coord);
    }

    /// Returns `true` if the bitboard has square `coord`
    #[inline]
    pub const fn has(&self, coord: Coord) -> bool {
        ((self.0 >> coord.index()) & 1) != 0
    }

    /// Unwraps the bitboard into raw `u64`
    ///
    /// This function does the same as [`u64::from()`](#method.from), except that this one is `const`.
    #[inline]
    pub const fn as_raw(&self) -> u64 {
        self.0
    }

    /// Returns the number of bits in the bitboard
    #[inline]
    pub const fn len(&self) -> u32 {
        self.0.count_ones()
    }

    /// Returns `true` if the bitboard doesn't contain any squares
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.0 == 0
    }

    /// The opposite of [`is_empty()`](Bitboard::is_empty())
    #[inline]
    pub const fn is_nonempty(&self) -> bool {
        self.0 != 0
    }
}

impl From<Bitboard> for u64 {
    #[inline]
    fn from(b: Bitboard) -> u64 {
        b.0
    }
}

impl From<u64> for Bitboard {
    #[inline]
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

/// Iterator over the squares containing in a [`Bitboard`]
///
/// The squares are iterated in the increasing order of [`index()`](crate::types::Coord::index).
///
/// # Example
///
/// ```
/// # use owlchess_base::{bitboard::Bitboard, types::Coord};
/// #
/// let b = Bitboard::from(0b1100_1011);
/// let mut iter = b.into_iter();
/// assert_eq!(iter.next(), Some(Coord::from_index(0)));
/// assert_eq!(iter.next(), Some(Coord::from_index(1)));
/// assert_eq!(iter.next(), Some(Coord::from_index(3)));
/// assert_eq!(iter.next(), Some(Coord::from_index(6)));
/// assert_eq!(iter.next(), Some(Coord::from_index(7)));
/// assert_eq!(iter.next(), None);
/// ```
#[derive(Clone)]
pub struct Iter(u64);

impl Iterator for Iter {
    type Item = Coord;

    #[inline]
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

    #[inline]
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
        assert_eq!((!bb1).len(), 62);
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
