use std::fmt::{self, Display};
use std::hint;
use std::str::FromStr;
use thiserror::Error;

#[derive(Error, Debug, Copy, Clone, PartialEq, Eq)]
pub enum CoordParseError {
    #[error("unexpected file char {0:?}")]
    UnexpectedFileChar(char),
    #[error("unexpected rank char {0:?}")]
    UnexpectedRankChar(char),
    #[error("invalid string length")]
    BadLength,
}

#[derive(Error, Debug, Copy, Clone, PartialEq, Eq)]
pub enum CellParseError {
    #[error("unexpected cell char {0:?}")]
    UnexpectedChar(char),
    #[error("invalid string length")]
    BadLength,
}

#[derive(Error, Debug, Copy, Clone, PartialEq, Eq)]
pub enum ColorParseError {
    #[error("unexpected color char {0:?}")]
    UnexpectedChar(char),
    #[error("invalid string length")]
    BadLength,
}

#[derive(Error, Debug, Copy, Clone, PartialEq, Eq)]
pub enum CastlingRightsParseError {
    #[error("unexpected char {0:?}")]
    UnexpectedChar(char),
    #[error("duplicate char {0:?}")]
    DuplicateChar(char),
    #[error("unexpected empty string")]
    EmptyString,
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
    pub const fn index(&self) -> usize {
        *self as u8 as usize
    }

    pub const unsafe fn from_index_unchecked(val: usize) -> Self {
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

    pub const fn from_index(val: usize) -> Self {
        assert!(val < 8, "file index must be between 0 and 7");
        unsafe { Self::from_index_unchecked(val) }
    }

    pub fn iter() -> impl Iterator<Item = Self> {
        (0..8).map(|x| unsafe { Self::from_index_unchecked(x) })
    }

    unsafe fn from_char_unchecked(c: char) -> Self {
        File::from_index_unchecked((u32::from(c) - u32::from('a')) as usize)
    }

    pub fn from_char(c: char) -> Option<Self> {
        match c {
            'a'..='h' => Some(unsafe { Self::from_char_unchecked(c) }),
            _ => None,
        }
    }

    pub fn as_char(&self) -> char {
        (b'a' + *self as u8) as char
    }
}

impl fmt::Display for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{}", self.as_char())
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
    pub const fn index(&self) -> usize {
        *self as u8 as usize
    }

    pub const unsafe fn from_index_unchecked(val: usize) -> Self {
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

    pub const fn from_index(val: usize) -> Self {
        assert!(val < 8, "rank index must be between 0 and 7");
        unsafe { Self::from_index_unchecked(val) }
    }

    pub fn iter() -> impl Iterator<Item = Self> {
        (0..8).map(|x| unsafe { Self::from_index_unchecked(x) })
    }

    unsafe fn from_char_unchecked(c: char) -> Self {
         Rank::from_index_unchecked((u32::from('8') - u32::from(c)) as usize)
    }

    pub fn from_char(c: char) -> Option<Self> {
        match c {
            '1'..='8' => Some(unsafe { Self::from_char_unchecked(c) }),
            _ => None,
        }
    }

    pub fn as_char(&self) -> char {
        (b'8' - *self as u8) as char
    }
}

impl fmt::Display for Rank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{}", self.as_char())
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Coord(u8);

impl Coord {
    pub const fn from_index(val: usize) -> Coord {
        assert!(val < 64, "coord must be between 0 and 63");
        Coord(val as u8)
    }

    pub const unsafe fn from_index_unchecked(val: usize) -> Coord {
        Coord(val as u8)
    }

    pub const fn from_parts(file: File, rank: Rank) -> Coord {
        Coord(((rank as u8) << 3) | file as u8)
    }

    pub const fn file(&self) -> File {
        unsafe { File::from_index_unchecked((self.0 & 7) as usize) }
    }

    pub const fn rank(&self) -> Rank {
        unsafe { Rank::from_index_unchecked((self.0 >> 3) as usize) }
    }

    pub const fn index(&self) -> usize {
        self.0 as usize
    }

    pub const fn flipped_rank(self) -> Coord {
        Coord(self.0 ^ 56)
    }

    pub const fn flipped_file(self) -> Coord {
        Coord(self.0 ^ 7)
    }

    pub const fn diag1(&self) -> usize {
        self.file().index() + self.rank().index()
    }

    pub const fn diag2(&self) -> usize {
        7 - self.rank().index() + self.file().index()
    }

    pub const fn add(self, delta: isize) -> Coord {
        Coord::from_index(self.index().wrapping_add(delta as usize))
    }

    pub const unsafe fn add_unchecked(self, delta: isize) -> Coord {
        Coord::from_index_unchecked(self.index().wrapping_add(delta as usize))
    }

    pub fn try_shift(self, delta_file: isize, delta_rank: isize) -> Option<Coord> {
        let new_file = self.file().index().wrapping_add(delta_file as usize);
        let new_rank = self.rank().index().wrapping_add(delta_rank as usize);
        if new_file >= 8 || new_rank >= 8 {
            return None;
        }
        unsafe {
            Some(Coord::from_parts(
                File::from_index_unchecked(new_file),
                Rank::from_index_unchecked(new_rank),
            ))
        }
    }

    pub fn iter() -> impl Iterator<Item = Self> {
        (0_u8..64_u8).map(Coord)
    }
}

impl fmt::Debug for Coord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        if self.0 < 64 {
            return write!(f, "Coord({})", self);
        }
        write!(f, "Coord(?{:?})", self.0)
    }
}

impl fmt::Display for Coord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{}{}", self.file().as_char(), self.rank().as_char())
    }
}

impl FromStr for Coord {
    type Err = CoordParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() != 2 {
            return Err(CoordParseError::BadLength);
        }
        let bytes = s.as_bytes();
        let (file_ch, rank_ch) = (bytes[0] as char, bytes[1] as char);
        Ok(Coord::from_parts(
            File::from_char(file_ch).ok_or(CoordParseError::UnexpectedFileChar(file_ch))?,
            Rank::from_char(rank_ch).ok_or(CoordParseError::UnexpectedRankChar(rank_ch))?,
        ))
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    pub const fn inv(&self) -> Color {
        match *self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    pub fn as_char(&self) -> char {
        match *self {
            Color::White => 'w',
            Color::Black => 'b',
        }
    }

    pub fn from_char(c: char) -> Option<Color> {
        match c {
            'w' => Some(Color::White),
            'b' => Some(Color::Black),
            _ => None,
        }
    }
}

impl Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{}", self.as_char())
    }
}

impl FromStr for Color {
    type Err = ColorParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() != 1 {
            return Err(ColorParseError::BadLength);
        }
        let ch = s.as_bytes()[0] as char;
        Color::from_char(s.as_bytes()[0] as char).ok_or(ColorParseError::UnexpectedChar(ch))
    }
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

#[derive(Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Cell(u8);

impl Cell {
    pub const EMPTY: Cell = Cell(0);
    pub const MAX_INDEX: usize = 13;

    pub const fn is_empty(&self) -> bool {
        self.0 == 0
    }

    pub const fn is_occupied(&self) -> bool {
        self.0 != 0
    }

    pub const unsafe fn from_index_unchecked(val: usize) -> Cell {
        Cell(val as u8)
    }

    pub const fn from_index(val: usize) -> Cell {
        assert!(val < Self::MAX_INDEX, "index too large");
        Cell(val as u8)
    }

    pub const fn index(&self) -> usize {
        self.0 as usize
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

    pub fn as_utf8_char(&self) -> char {
        [
            '.', '♙', '♔', '♘', '♗', '♖', '♕', '♟', '♚', '♞', '♝', '♜', '♛',
        ][self.0 as usize]
    }

    pub fn from_char(c: char) -> Option<Self> {
        if c == '.' {
            return Some(Cell::EMPTY);
        }
        let color = if c.is_ascii_uppercase() {
            Color::White
        } else {
            Color::Black
        };
        let piece = match c.to_ascii_lowercase() {
            'p' => Piece::Pawn,
            'k' => Piece::King,
            'n' => Piece::Knight,
            'b' => Piece::Bishop,
            'r' => Piece::Rook,
            'q' => Piece::Queen,
            _ => return None,
        };
        Some(Cell::from_parts(color, piece))
    }
}

impl fmt::Debug for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        if (self.0 as usize) < Self::MAX_INDEX {
            return write!(f, "Cell({})", self.as_char());
        }
        write!(f, "Cell(?{:?})", self.0)
    }
}

impl fmt::Display for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{}", self.as_char())
    }
}

impl FromStr for Cell {
    type Err = CellParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() != 1 {
            return Err(CellParseError::BadLength);
        }
        let ch = s.as_bytes()[0] as char;
        Cell::from_char(s.as_bytes()[0] as char).ok_or(CellParseError::UnexpectedChar(ch))
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CastlingSide {
    Queen = 0,
    King = 1,
}

#[derive(Default, Copy, Clone, PartialEq, Eq, Hash)]
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

    pub fn unset_color(&mut self, c: Color) {
        self.unset(c, CastlingSide::King);
        self.unset(c, CastlingSide::Queen);
    }

    pub const fn from_index(val: usize) -> CastlingRights {
        assert!(val < 16, "raw castling rights must be between 0 and 15");
        CastlingRights(val as u8)
    }

    pub const fn index(&self) -> usize {
        self.0 as usize
    }
}

impl fmt::Debug for CastlingRights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        if self.0 < 16 {
            return write!(f, "CastlingRights({})", self);
        }
        write!(f, "CastlingRights(?{:?})", self.0)
    }
}

impl fmt::Display for CastlingRights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        if *self == Self::EMPTY {
            return write!(f, "-");
        }
        if self.has(Color::White, CastlingSide::King) {
            write!(f, "K")?;
        }
        if self.has(Color::White, CastlingSide::Queen) {
            write!(f, "Q")?;
        }
        if self.has(Color::Black, CastlingSide::King) {
            write!(f, "k")?;
        }
        if self.has(Color::Black, CastlingSide::Queen) {
            write!(f, "q")?;
        }
        Ok(())
    }
}

impl FromStr for CastlingRights {
    type Err = CastlingRightsParseError;

    fn from_str(s: &str) -> Result<CastlingRights, Self::Err> {
        type Error = CastlingRightsParseError;
        if s == "-" {
            return Ok(CastlingRights::EMPTY);
        }
        if s.is_empty() {
            return Err(Error::EmptyString);
        }
        let mut res = CastlingRights::EMPTY;
        for b in s.bytes() {
            let (color, side) = match b {
                b'K' => (Color::White, CastlingSide::King),
                b'Q' => (Color::White, CastlingSide::Queen),
                b'k' => (Color::Black, CastlingSide::King),
                b'q' => (Color::Black, CastlingSide::Queen),
                _ => return Err(Error::UnexpectedChar(b as char)),
            };
            if res.has(color, side) {
                return Err(Error::DuplicateChar(b as char));
            }
            res.set(color, side);
        }
        Ok(res)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DrawKind {
    Stalemate,
    InsufficientMaterial,
    Moves75,
    Repeat5,
    Moves50,
    Repeat3,
    Agreement,
    Unknown,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum WinKind {
    Checkmate,
    TimeForfeit,
    EngineError,
    Resign,
    Unknown,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Outcome {
    White(WinKind),
    Black(WinKind),
    Draw(DrawKind),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum OutcomeFilter {
    Force,
    Strict,
    Relaxed,
}

impl Outcome {
    pub fn winner(&self) -> Option<Color> {
        match self {
            Self::White(_) => Some(Color::White),
            Self::Black(_) => Some(Color::Black),
            Self::Draw(_) => None,
        }
    }

    pub fn win(color: Color, kind: WinKind) -> Outcome {
        match color {
            Color::White => Self::White(kind),
            Color::Black => Self::Black(kind),
        }
    }

    pub fn is_force(&self) -> bool {
        matches!(
            *self,
            Self::White(WinKind::Checkmate)
                | Self::Black(WinKind::Checkmate)
                | Self::Draw(DrawKind::Stalemate)
        )
    }

    pub fn is_auto(&self, filter: OutcomeFilter) -> bool {
        if self.is_force() {
            return true;
        }
        if matches!(filter, OutcomeFilter::Strict | OutcomeFilter::Relaxed)
            && matches!(
                *self,
                Self::Draw(DrawKind::InsufficientMaterial | DrawKind::Moves75 | DrawKind::Repeat5)
            )
        {
            return true;
        }
        matches!(filter, OutcomeFilter::Relaxed)
            && matches!(*self, Self::Draw(DrawKind::Moves50 | DrawKind::Repeat3))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file() {
        for (idx, file) in File::iter().enumerate() {
            assert_eq!(file.index(), idx);
            assert_eq!(File::from_index(idx), file);
        }
    }

    #[test]
    fn test_rank() {
        for (idx, rank) in Rank::iter().enumerate() {
            assert_eq!(rank.index(), idx);
            assert_eq!(Rank::from_index(idx), rank);
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
        assert_eq!(empty.to_string(), "-");
        assert_eq!(CastlingRights::from_str("-"), Ok(empty));

        let full = CastlingRights::FULL;
        assert!(full.has(Color::White, CastlingSide::Queen));
        assert!(full.has(Color::White, CastlingSide::King));
        assert!(full.has(Color::Black, CastlingSide::Queen));
        assert!(full.has(Color::Black, CastlingSide::King));
        assert_eq!(full.to_string(), "KQkq");
        assert_eq!(CastlingRights::from_str("KQkq"), Ok(full));

        let mut rights = CastlingRights::EMPTY;
        rights.set(Color::White, CastlingSide::King);
        assert!(!rights.has(Color::White, CastlingSide::Queen));
        assert!(rights.has(Color::White, CastlingSide::King));
        assert!(!rights.has(Color::Black, CastlingSide::Queen));
        assert!(!rights.has(Color::Black, CastlingSide::King));
        assert_eq!(rights.to_string(), "K");
        assert_eq!(CastlingRights::from_str("K"), Ok(rights));

        rights.unset(Color::White, CastlingSide::King);
        rights.flip(Color::Black, CastlingSide::Queen);
        assert!(!rights.has(Color::White, CastlingSide::Queen));
        assert!(!rights.has(Color::White, CastlingSide::King));
        assert!(rights.has(Color::Black, CastlingSide::Queen));
        assert!(!rights.has(Color::Black, CastlingSide::King));
        assert_eq!(rights.to_string(), "q");
        assert_eq!(CastlingRights::from_str("q"), Ok(rights));
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
