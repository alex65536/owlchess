use std::fmt::Display;
use std::hint;
use std::str::FromStr;
use thiserror::Error;

#[derive(Error, Debug, Copy, Clone, PartialEq, Eq)]
pub enum CoordError {
    #[error("unexpected file char `{0}`")]
    UnexpectedFile(char),
    #[error("unexpected rank char `{0}`")]
    UnexpectedRank(char),
    #[error("invalid string length")]
    BadLength,
}

#[derive(Error, Debug, Copy, Clone, PartialEq, Eq)]
pub enum CellError {
    #[error("unexpected cell char `{0}`")]
    UnexpectedCell(char),
    #[error("invalid string length")]
    BadLength,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
#[repr(u8)]
pub enum File {
    A = 0,
    B = 1,
    C = 2,
    D = 3,
    E = 4,
    F = 5,
    G = 6,
    H = 7,
}

impl File {
    pub const fn index(&self) -> u8 {
        *self as u8
    }

    pub const unsafe fn from_index_unchecked(val: u8) -> Self {
        match val {
            0 => File::A,
            1 => File::B,
            2 => File::C,
            3 => File::D,
            4 => File::E,
            5 => File::F,
            6 => File::G,
            7 => File::H,
            _ => hint::unreachable_unchecked(),
        }
    }

    pub const fn from_index(val: u8) -> Self {
        assert!(val < 8, "file index must be between 0 and 7");
        unsafe { Self::from_index_unchecked(val) }
    }

    pub fn iter() -> impl Iterator<Item = Self> {
        (0_u8..8_u8).map(|x| unsafe { Self::from_index_unchecked(x) })
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
#[repr(u8)]
pub enum Rank {
    R8 = 0,
    R7 = 1,
    R6 = 2,
    R5 = 3,
    R4 = 4,
    R3 = 5,
    R2 = 6,
    R1 = 7,
}

impl Rank {
    pub const fn index(&self) -> u8 {
        *self as u8
    }

    pub const unsafe fn from_index_unchecked(val: u8) -> Self {
        match val {
            0 => Rank::R8,
            1 => Rank::R7,
            2 => Rank::R6,
            3 => Rank::R5,
            4 => Rank::R4,
            5 => Rank::R3,
            6 => Rank::R2,
            7 => Rank::R1,
            _ => hint::unreachable_unchecked(),
        }
    }

    pub const fn from_index(val: u8) -> Self {
        assert!(val < 8, "rank index must be between 0 and 7");
        unsafe { Self::from_index_unchecked(val) }
    }

    pub fn iter() -> impl Iterator<Item = Self> {
        (0_u8..8_u8).map(|x| unsafe { Self::from_index_unchecked(x) })
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Coord(u8);

impl Coord {
    pub const fn from_index(val: u8) -> Coord {
        assert!(val < 64, "coord must be between 0 and 63");
        Coord(val)
    }

    pub const unsafe fn from_index_unchecked(val: u8) -> Coord {
        Coord(val)
    }

    pub const fn from_parts(file: File, rank: Rank) -> Coord {
        Coord(((rank as u8) << 3) | file as u8)
    }

    pub const fn file(&self) -> File {
        unsafe { File::from_index_unchecked(self.0 & 7) }
    }

    pub const fn rank(&self) -> Rank {
        unsafe { Rank::from_index_unchecked(self.0 >> 3) }
    }

    pub const fn index(&self) -> u8 {
        self.0
    }

    pub const fn flipped_x(self) -> Coord {
        Coord(self.0 ^ 56)
    }

    pub const fn flipped_y(self) -> Coord {
        Coord(self.0 ^ 7)
    }

    pub const fn diag1(&self) -> u8 {
        self.file().index() + self.rank().index()
    }

    pub const fn diag2(&self) -> u8 {
        7 - self.file().index() + self.rank().index()
    }

    pub fn iter() -> impl Iterator<Item = Self> {
        (0_u8..64_u8).map(Coord)
    }
}

impl Display for Coord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        write!(
            f,
            "{}{}",
            (b'a' + self.file() as u8) as char,
            (b'8' - self.rank() as u8) as char
        )
    }
}

impl FromStr for Coord {
    type Err = CoordError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() != 2 {
            return Err(CoordError::BadLength);
        }
        let bytes = s.as_bytes();
        let file = match bytes[0] {
            b @ b'a'..=b'h' => unsafe { File::from_index_unchecked(b - b'a') },
            b => return Err(CoordError::UnexpectedFile(b as char)),
        };
        let rank = match bytes[1] {
            b @ b'1'..=b'8' => unsafe { Rank::from_index_unchecked(b'8' - b) },
            b => return Err(CoordError::UnexpectedRank(b as char)),
        };
        Ok(Coord::from_parts(file, rank))
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Piece {
    Pawn = 0,
    King = 1,
    Knight = 2,
    Bishop = 3,
    Rook = 4,
    Queen = 5,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Cell(u8);

impl Cell {
    pub const EMPTY: Cell = Cell(0);
    pub const MAX_INDEX: u8 = 13;

    pub const fn is_empty(&self) -> bool {
        self.0 == 0
    }

    pub const unsafe fn from_index_unchecked(val: u8) -> Cell {
        Cell(val)
    }

    pub const fn from_index(val: u8) -> Cell {
        assert!(val < Self::MAX_INDEX, "index too large");
        Cell(val)
    }

    pub const fn index(&self) -> u8 {
        self.0
    }

    pub const fn from_parts(c: Color, p: Piece) -> Cell {
        Cell(match c {
            Color::White => 1 + p as u8,
            Color::Black => 7 + p as u8,
        })
    }

    pub const fn color(&self) -> Option<Color> {
        match self.0 {
            0 => None,
            1..=6 => Some(Color::White),
            _ => Some(Color::Black),
        }
    }

    pub const fn piece(&self) -> Option<Piece> {
        match self.0 {
            0 => None,
            1 | 7 => Some(Piece::Pawn),
            2 | 8 => Some(Piece::King),
            3 | 9 => Some(Piece::Knight),
            4 | 10 => Some(Piece::Bishop),
            5 | 11 => Some(Piece::Rook),
            6 | 12 => Some(Piece::Queen),
            _ => unsafe { hint::unreachable_unchecked() },
        }
    }

    pub fn iter() -> impl Iterator<Item = Self> {
        (0..Self::MAX_INDEX).map(|x| unsafe { Self::from_index_unchecked(x) })
    }

    pub fn as_char(&self) -> char {
        b".PKNBRQpknbrq"[self.0 as usize] as char
    }
}

impl Display for Cell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        write!(f, "{}", self.as_char())
    }
}

impl FromStr for Cell {
    type Err = CellError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() != 1 {
            return Err(CellError::BadLength);
        }
        let b = s.as_bytes()[0];
        if b == b'.' {
            return Ok(Cell::EMPTY);
        }
        let color = if b.is_ascii_uppercase() {
            Color::White
        } else {
            Color::Black
        };
        let piece = match b.to_ascii_lowercase() {
            b'p' => Piece::Pawn,
            b'k' => Piece::King,
            b'n' => Piece::Knight,
            b'b' => Piece::Bishop,
            b'r' => Piece::Rook,
            b'q' => Piece::Queen,
            _ => return Err(CellError::UnexpectedCell(b as char)),
        };
        Ok(Cell::from_parts(color, piece))
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CastlingSide {
    Queen = 0,
    King = 1,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct CastlingRights(u8);

impl CastlingRights {
    const fn to_index(c: Color, s: CastlingSide) -> u8 {
        ((c as u8) << 1) | s as u8
    }

    pub const EMPTY: CastlingRights = CastlingRights(0);
    pub const FULL: CastlingRights = CastlingRights(15);

    pub const fn has(&self, c: Color, s: CastlingSide) -> bool {
        ((self.0 >> Self::to_index(c, s)) & 1) != 0
    }

    pub fn flip(&mut self, c: Color, s: CastlingSide) {
        self.0 ^= 1_u8 << Self::to_index(c, s)
    }

    pub const fn with(self, c: Color, s: CastlingSide) -> CastlingRights {
        CastlingRights(self.0 | (1_u8 << Self::to_index(c, s)))
    }

    pub fn set(&mut self, c: Color, s: CastlingSide) {
        *self = self.with(c, s)
    }

    pub fn unset(&mut self, c: Color, s: CastlingSide) {
        self.0 &= !(1_u8 << Self::to_index(c, s))
    }

    pub const fn from_raw(val: u8) -> CastlingRights {
        assert!(val < 16, "raw castling rights must be between 0 and 15");
        CastlingRights(val)
    }

    pub const fn as_raw(&self) -> u8 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file() {
        for (idx, file) in File::iter().enumerate() {
            assert_eq!(file.index(), idx as u8);
            assert_eq!(File::from_index(idx as u8), file);
        }
    }

    #[test]
    fn test_rank() {
        for (idx, rank) in Rank::iter().enumerate() {
            assert_eq!(rank.index(), idx as u8);
            assert_eq!(Rank::from_index(idx as u8), rank);
        }
    }

    #[test]
    fn test_coord() {
        let mut coords = Vec::new();
        for rank in Rank::iter() {
            for file in File::iter() {
                let coord = Coord::from_parts(file, rank);
                assert_eq!(coord.file(), file);
                assert_eq!(coord.rank(), rank);
                coords.push(coord);
            }
        }
        assert_eq!(coords, Coord::iter().collect::<Vec<_>>());
    }

    #[test]
    fn test_cell() {
        assert_eq!(Cell::EMPTY.color(), None);
        assert_eq!(Cell::EMPTY.piece(), None);
        let mut cells = vec![Cell::EMPTY];
        for color in [Color::White, Color::Black] {
            for piece in [
                Piece::Pawn,
                Piece::King,
                Piece::Knight,
                Piece::Bishop,
                Piece::Rook,
                Piece::Queen,
            ] {
                let cell = Cell::from_parts(color, piece);
                assert_eq!(cell.color(), Some(color));
                assert_eq!(cell.piece(), Some(piece));
                cells.push(cell);
            }
        }
        assert_eq!(cells, Cell::iter().collect::<Vec<_>>());
    }

    #[test]
    fn test_castling() {
        let empty = CastlingRights::EMPTY;
        assert!(!empty.has(Color::White, CastlingSide::Queen));
        assert!(!empty.has(Color::White, CastlingSide::King));
        assert!(!empty.has(Color::Black, CastlingSide::Queen));
        assert!(!empty.has(Color::Black, CastlingSide::King));

        let full = CastlingRights::FULL;
        assert!(full.has(Color::White, CastlingSide::Queen));
        assert!(full.has(Color::White, CastlingSide::King));
        assert!(full.has(Color::Black, CastlingSide::Queen));
        assert!(full.has(Color::Black, CastlingSide::King));

        let mut rights = CastlingRights::EMPTY;
        rights.set(Color::White, CastlingSide::King);
        assert!(!rights.has(Color::White, CastlingSide::Queen));
        assert!(rights.has(Color::White, CastlingSide::King));
        assert!(!rights.has(Color::Black, CastlingSide::Queen));
        assert!(!rights.has(Color::Black, CastlingSide::King));

        rights.unset(Color::White, CastlingSide::King);
        rights.flip(Color::Black, CastlingSide::Queen);
        assert!(!rights.has(Color::White, CastlingSide::Queen));
        assert!(!rights.has(Color::White, CastlingSide::King));
        assert!(rights.has(Color::Black, CastlingSide::Queen));
        assert!(!rights.has(Color::Black, CastlingSide::King));
    }

    #[test]
    fn test_coord_str() {
        assert_eq!(
            Coord::from_parts(File::B, Rank::R4).to_string(),
            "b4".to_string()
        );
        assert_eq!(
            Coord::from_parts(File::A, Rank::R1).to_string(),
            "a1".to_string()
        );
        assert_eq!(
            Coord::from_str("a1"),
            Ok(Coord::from_parts(File::A, Rank::R1))
        );
        assert_eq!(
            Coord::from_str("b4"),
            Ok(Coord::from_parts(File::B, Rank::R4))
        );
        assert!(Coord::from_str("h9").is_err());
        assert!(Coord::from_str("i4").is_err());
    }

    #[test]
    fn test_cell_str() {
        for cell in Cell::iter() {
            let s = cell.to_string();
            assert_eq!(Cell::from_str(&s), Ok(cell));
        }
    }
}
