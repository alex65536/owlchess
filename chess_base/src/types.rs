//! Core chess types

use derive_more::Display;
use std::{fmt, hint, str::FromStr};
use thiserror::Error;

/// Error when parsing [`Coord`] from string
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum CoordParseError {
    /// Unexpected character for file coordinate
    #[error("unexpected file char {0:?}")]
    UnexpectedFileChar(char),
    /// Unexpected character for rank coordinate
    #[error("unexpected rank char {0:?}")]
    UnexpectedRankChar(char),
    /// Invalid string length
    #[error("invalid string length")]
    BadLength,
}

/// Error when parsing [`Cell`] from string
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum CellParseError {
    /// Unexpected character
    #[error("unexpected cell char {0:?}")]
    UnexpectedChar(char),
    /// Invalid string length
    #[error("invalid string length")]
    BadLength,
}

/// Error when parsing [`Color`] from string
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ColorParseError {
    /// Unexpected character
    #[error("unexpected color char {0:?}")]
    UnexpectedChar(char),
    /// Invalid string length
    #[error("invalid string length")]
    BadLength,
}

/// Error when parsing [`CastlingRights`] from string
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum CastlingRightsParseError {
    /// Unexpected character
    #[error("unexpected char {0:?}")]
    UnexpectedChar(char),
    /// Duplicate character
    #[error("duplicate char {0:?}")]
    DuplicateChar(char),
    /// The string is empty
    #[error("the string is empty")]
    EmptyString,
}

/// File (i. e. a vertical line) on a chess board
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
    /// Returns a numeric index of the current file
    ///
    /// The files are numbered from left to right, i.e. file A has index 0, and file H has index 7.
    #[inline]
    pub const fn index(&self) -> usize {
        *self as u8 as usize
    }

    /// Converts a file index to [`File`]
    ///
    /// # Safety
    ///
    /// The behavior is undefined when `val` is not in range `[0; 8)`.
    #[inline]
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

    /// Converts a file index to [`File`]
    ///
    /// # Panics
    ///
    /// The function panics when `val` is not in range `[0; 8)`.
    #[inline]
    pub const fn from_index(val: usize) -> Self {
        assert!(val < 8, "file index must be between 0 and 7");
        unsafe { Self::from_index_unchecked(val) }
    }

    /// Returns an iterator over all the files, in ascending order of their indices
    #[inline]
    pub fn iter() -> impl Iterator<Item = Self> {
        (0..8).map(|x| unsafe { Self::from_index_unchecked(x) })
    }

    #[inline]
    unsafe fn from_char_unchecked(c: char) -> Self {
        File::from_index_unchecked((u32::from(c) - u32::from('a')) as usize)
    }

    /// Creates a file from its character representation, if it's valid
    ///
    /// If `c` is a valid character representation of file, then the corresponding file is returned.
    /// Otherwise, returns `None`.
    ///
    /// Note that the only valid character representations are lowercase Latin letters from `'a``
    /// to `'h'` inclusively.
    ///
    /// # Example
    ///
    /// ```
    /// # use owlchess_base::types::File;
    /// #
    /// assert_eq!(File::from_char('a'), Some(File::A));
    /// assert_eq!(File::from_char('e'), Some(File::E));
    /// assert_eq!(File::from_char('q'), None);
    /// assert_eq!(File::from_char('A'), None);
    /// ```
    #[inline]
    pub fn from_char(c: char) -> Option<Self> {
        match c {
            'a'..='h' => Some(unsafe { Self::from_char_unchecked(c) }),
            _ => None,
        }
    }

    /// Converts a file into its character representation
    ///
    /// # Example
    ///
    /// ```
    /// # use owlchess_base::types::File;
    /// #
    /// assert_eq!(File::A.as_char(), 'a');
    /// assert_eq!(File::E.as_char(), 'e');
    /// ```
    #[inline]
    pub fn as_char(&self) -> char {
        (b'a' + *self as u8) as char
    }
}

impl fmt::Display for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{}", self.as_char())
    }
}

/// Rank (i. e. a horizontal line) on a chess board
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
    /// Returns a numeric index of the current rank
    ///
    /// The ranks are numbered from top to bottom, i.e. rank 8 has index 0, and rank 1 has index 7.
    #[inline]
    pub const fn index(&self) -> usize {
        *self as u8 as usize
    }

    /// Converts a rank index to [`Rank`]
    ///
    /// # Safety
    ///
    /// The behavior is undefined when `val` is not in range `[0; 8)`.
    #[inline]
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

    /// Converts a rank index to [`Rank`]
    ///
    /// # Panics
    ///
    /// The function panics when `val` is not in range `[0; 8)`.
    #[inline]
    pub const fn from_index(val: usize) -> Self {
        assert!(val < 8, "rank index must be between 0 and 7");
        unsafe { Self::from_index_unchecked(val) }
    }

    /// Returns an iterator over all the ranks, in ascending order of their indices
    #[inline]
    pub fn iter() -> impl Iterator<Item = Self> {
        (0..8).map(|x| unsafe { Self::from_index_unchecked(x) })
    }

    #[inline]
    unsafe fn from_char_unchecked(c: char) -> Self {
        Rank::from_index_unchecked((u32::from('8') - u32::from(c)) as usize)
    }

    /// Creates a rank from its character representation, if it's valid
    ///
    /// If `c` is a valid character representation of rank, then the corresponding rank is returned.
    /// Otherwise, returns `None`.
    ///
    /// # Example
    ///
    /// ```
    /// # use owlchess_base::types::Rank;
    /// #
    /// assert_eq!(Rank::from_char('1'), Some(Rank::R1));
    /// assert_eq!(Rank::from_char('5'), Some(Rank::R5));
    /// assert_eq!(Rank::from_char('9'), None);
    /// assert_eq!(Rank::from_char('A'), None);
    /// ```
    #[inline]
    pub fn from_char(c: char) -> Option<Self> {
        match c {
            '1'..='8' => Some(unsafe { Self::from_char_unchecked(c) }),
            _ => None,
        }
    }
    /// Converts a rank into its character representation
    ///
    /// # Example
    ///
    /// ```
    /// # use owlchess_base::types::Rank;
    /// #
    /// assert_eq!(Rank::R1.as_char(), '1');
    /// assert_eq!(Rank::R5.as_char(), '5');
    /// ```
    #[inline]
    pub fn as_char(&self) -> char {
        (b'8' - *self as u8) as char
    }
}

impl fmt::Display for Rank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{}", self.as_char())
    }
}

/// Coordinate of a square
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Coord(u8);

impl Coord {
    /// Creates a coordinate from its index
    ///
    /// See [`Coord::index()`] for the details about index assignment.
    ///
    /// # Panics
    ///
    /// This function panics if `val` is not a valid index.
    #[inline]
    pub const fn from_index(val: usize) -> Coord {
        assert!(val < 64, "coord must be between 0 and 63");
        Coord(val as u8)
    }

    /// Creates a coordinate from its index
    ///
    /// See [`Coord::index()`] for the details about index assignment.
    ///
    /// # Safety
    ///
    /// The behavior is undefined if `val` is not a valid index.
    #[inline]
    pub const unsafe fn from_index_unchecked(val: usize) -> Coord {
        Coord(val as u8)
    }

    /// Creates a square coordinate from the given file and rank
    #[inline]
    pub const fn from_parts(file: File, rank: Rank) -> Coord {
        Coord(((rank as u8) << 3) | file as u8)
    }

    /// Returns the file on which the square is located
    #[inline]
    pub const fn file(&self) -> File {
        unsafe { File::from_index_unchecked((self.0 & 7) as usize) }
    }

    /// Returns the rank on which the square is located
    #[inline]
    pub const fn rank(&self) -> Rank {
        unsafe { Rank::from_index_unchecked((self.0 >> 3) as usize) }
    }

    /// Returns the index of the square
    ///
    /// The indices are assigned in a big-endian rank-file manner:
    ///
    /// ```notrust
    /// 8 |  0  1  2  3  4  5  6  7
    /// 7 |  8  9 10 11 12 13 14 15
    /// 6 | 16 17 18 19 20 21 22 23
    /// 5 | 24 25 26 27 28 29 30 31
    /// 4 | 32 33 34 35 36 37 38 39
    /// 3 | 40 41 42 43 44 45 46 47
    /// 2 | 48 49 50 51 52 53 54 55
    /// 1 | 56 57 58 59 60 61 62 63
    /// --+------------------------
    ///   |  a  b  c  d  e  f  g  h
    /// ```
    #[inline]
    pub const fn index(&self) -> usize {
        self.0 as usize
    }

    /// Flips the square vertically
    ///
    /// # Example
    ///
    /// ```
    /// # use owlchess_base::types::{File, Rank, Coord};
    /// #
    /// let c3 = Coord::from_parts(File::C, Rank::R3);
    /// let c6 = Coord::from_parts(File::C, Rank::R6);
    /// assert_eq!(c3.flipped_rank(), c6);
    /// assert_eq!(c6.flipped_rank(), c3);
    /// ```
    #[inline]
    pub const fn flipped_rank(self) -> Coord {
        Coord(self.0 ^ 56)
    }

    /// Flips the square horizontally
    ///
    /// # Example
    ///
    /// ```
    /// # use owlchess_base::types::{File, Rank, Coord};
    /// #
    /// let c3 = Coord::from_parts(File::C, Rank::R3);
    /// let f3 = Coord::from_parts(File::F, Rank::R3);
    /// assert_eq!(c3.flipped_file(), f3);
    /// assert_eq!(f3.flipped_file(), c3);
    /// ```
    #[inline]
    pub const fn flipped_file(self) -> Coord {
        Coord(self.0 ^ 7)
    }

    /// Returns the index of diagonal on which the square is located
    ///
    /// Diagonals have indices in range `[0; 15)`, where h1-h1 diagonal has index 0
    /// and a8-a8 diagonal has index 14. The main diagonal a1-h8 has index 7.
    #[inline]
    pub const fn diag(&self) -> usize {
        self.file().index() + self.rank().index()
    }

    /// Returns the index of antidiagonal on which the square is located
    ///
    /// Antidiagonals have indices in range `[0; 15)`, where a1-a1 antidiagonal has index 0
    /// and h8-h8 antidiagonal has index 14. The main antidiagonal a8-h1 has index 7.
    #[inline]
    pub const fn antidiag(&self) -> usize {
        7 - self.rank().index() + self.file().index()
    }

    /// Adds `delta` to the index of the coordinate
    ///
    /// # Panics
    ///
    /// The function panics if the index is invalid (i.e. not in range `[0; 64)`) after
    /// such addition.
    #[inline]
    pub const fn add(self, delta: isize) -> Coord {
        Coord::from_index(self.index().wrapping_add(delta as usize))
    }

    /// Adds `delta` to the index of the coordinate
    ///
    /// # Safety
    ///
    /// The behavior is undefined if the index is invalid (i.e. not in range `[0; 64)`)
    /// after such addition.
    #[inline]
    pub const unsafe fn add_unchecked(self, delta: isize) -> Coord {
        Coord::from_index_unchecked(self.index().wrapping_add(delta as usize))
    }

    /// Adds `delta_file` to the file index and `delta_rank` to the rank index.
    /// If either file index or rank index becomes invalid, returns `None`, otherwise
    /// the new file and rank are gathered into a new [`Coord`], which is returned
    ///
    /// # Example
    ///
    /// ```
    /// # use owlchess_base::types::{File, Rank, Coord};
    /// #
    /// let b3 = Coord::from_parts(File::B, Rank::R3);
    /// let e4 = Coord::from_parts(File::E, Rank::R4);
    /// assert_eq!(b3.shift(-3, 1), None);
    /// assert_eq!(b3.shift(3, -1), Some(e4));
    /// ```
    #[inline]
    pub fn shift(self, delta_file: isize, delta_rank: isize) -> Option<Coord> {
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

    /// Iterates over all possible coordinates in ascending order of their indices
    #[inline]
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

/// Color of chess pieces (either white or black)
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    /// Returns the opposite color
    #[inline]
    pub const fn inv(&self) -> Color {
        match *self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    /// Returns a character representation of the color
    ///
    /// The character representation is `"w"` for white, and `"b"` for black.
    #[inline]
    pub fn as_char(&self) -> char {
        match *self {
            Color::White => 'w',
            Color::Black => 'b',
        }
    }

    /// Creates a color from its character representation
    ///
    /// If `c` is not a valid character representation of color, returns `None`.
    #[inline]
    pub fn from_char(c: char) -> Option<Color> {
        match c {
            'w' => Some(Color::White),
            'b' => Some(Color::Black),
            _ => None,
        }
    }

    /// Returns a full string representation of the color (either `"white"` or `"black"`)
    #[inline]
    pub fn as_long_str(&self) -> &'static str {
        match *self {
            Color::White => "white",
            Color::Black => "black",
        }
    }
}

impl fmt::Display for Color {
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

/// Kind of chess pieces (without regard to piece color)
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

impl Piece {
    /// Number of different possible indices of [`Piece`]
    ///
    /// It exceeds maximum possible index by one.
    pub const COUNT: usize = 6;

    /// Returns a numeric index of the current piece
    #[inline]
    pub const fn index(&self) -> usize {
        *self as u8 as usize
    }

    /// Converts a piece index to [`Piece`]
    ///
    /// # Safety
    ///
    /// The behavior is undefined if the index is invalid (i.e. it is greater or equal than [`Piece::COUNT`])
    #[inline]
    pub const unsafe fn from_index_unchecked(val: usize) -> Self {
        match val {
            0 => Self::Pawn,
            1 => Self::King,
            2 => Self::Knight,
            3 => Self::Bishop,
            4 => Self::Rook,
            5 => Self::Queen,
            _ => hint::unreachable_unchecked(),
        }
    }

    /// Converts a piece index to [`Piece`]
    ///
    /// # Panics
    ///
    /// The function panics if the index is invalid (i.e. it is greater or equal than [`Piece::COUNT`])
    #[inline]
    pub const fn from_index(val: usize) -> Self {
        assert!(val < Self::COUNT, "piece index must be between 0 and 5");
        unsafe { Self::from_index_unchecked(val) }
    }

    /// Returns an iterator over all the pieces, in ascending order of their indices
    #[inline]
    pub fn iter() -> impl Iterator<Item = Self> {
        (0..Self::COUNT).map(|x| unsafe { Self::from_index_unchecked(x) })
    }
}

/// Contents of square on a chess board
///
/// A square can be either empty or contain a piece of some given color.
///
/// This type is one compact and is only one byte long to facilitate compact chess board
/// representation.
#[derive(Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Cell(u8);

impl Cell {
    /// [`Cell`] without any pieces
    pub const EMPTY: Cell = Cell(0);

    /// Number of different possible indices of [`Cell`]
    ///
    /// It exceeds maximum possible index by one.
    pub const COUNT: usize = 13;

    /// Returns `true` if the cell doesn't contain any pieces
    #[inline]
    pub const fn is_free(&self) -> bool {
        self.0 == 0
    }

    /// Returns `true` if the cell contains a piece
    #[inline]
    pub const fn is_occupied(&self) -> bool {
        self.0 != 0
    }

    /// Creates a cell from its index
    ///
    /// # Safety
    ///
    /// The behavior is undefined if the index is invalid (i.e. it is greater or equal than [`Cell::COUNT`])
    #[inline]
    pub const unsafe fn from_index_unchecked(val: usize) -> Cell {
        Cell(val as u8)
    }

    /// Creates a cell from its index
    ///
    /// # Panics
    ///
    /// The function panics if the index is invalid (i.e. it is greater or equal than [`Cell::COUNT`])
    #[inline]
    pub const fn from_index(val: usize) -> Cell {
        assert!(val < Self::COUNT, "index too large");
        Cell(val as u8)
    }

    /// Returns the index of the cell
    ///
    /// Cell indices are stable between updates, and changing the index of some given cell
    /// is considered API breakage.
    #[inline]
    pub const fn index(&self) -> usize {
        self.0 as usize
    }

    /// Creates a cell with a piece `p` of color `c`
    #[inline]
    pub const fn from_parts(c: Color, p: Piece) -> Cell {
        Cell(match c {
            Color::White => 1 + p as u8,
            Color::Black => 7 + p as u8,
        })
    }

    /// Returns the color of the piece on the cell
    ///
    /// If the cell is empty, returns `None`.
    #[inline]
    pub const fn color(&self) -> Option<Color> {
        match self.0 {
            0 => None,
            1..=6 => Some(Color::White),
            _ => Some(Color::Black),
        }
    }

    /// Returns the kind of the piece on the cell
    ///
    /// If the cell is empty, returns `None`.
    #[inline]
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

    /// Iterates over all possible cells in ascending order of their indices
    #[inline]
    pub fn iter() -> impl Iterator<Item = Self> {
        (0..Self::COUNT).map(|x| unsafe { Self::from_index_unchecked(x) })
    }

    /// Returns a character representation of the cell
    ///
    /// Unlike [`Cell::as_utf8_char`], the representation is an ASCII character.
    #[inline]
    pub fn as_char(&self) -> char {
        b".PKNBRQpknbrq"[self.0 as usize] as char
    }

    /// Converts a cell to a corresponding Unicode character
    #[inline]
    pub fn as_utf8_char(&self) -> char {
        [
            '.', '♙', '♔', '♘', '♗', '♖', '♕', '♟', '♚', '♞', '♝', '♜', '♛',
        ][self.0 as usize]
    }

    /// Creates a cell from its character representation
    ///
    /// If `c` is not a valid character representation of a cell, returns `None`. Note that
    /// only ASCII character repesentations as returned by [`Cell::as_char`] are accepted.
    #[inline]
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
        if (self.0 as usize) < Self::COUNT {
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

/// Castling side (either queenside or kingside)
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CastlingSide {
    /// Queenside castling (a.k.a. O-O-O)
    Queen = 0,
    /// Kingside castling (a.k.a. O-O)
    King = 1,
}

/// Flags specifying allowed castling sides for both white and black
#[derive(Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct CastlingRights(u8);

impl CastlingRights {
    #[inline]
    const fn to_index(c: Color, s: CastlingSide) -> u8 {
        ((c as u8) << 1) | s as u8
    }

    #[inline]
    const fn to_color_mask(c: Color) -> u8 {
        3 << ((c as u8) << 1)
    }

    /// Empty castling rights (i.e. castling is not allowed at all)
    pub const EMPTY: CastlingRights = CastlingRights(0);

    /// Full castling rights (i.e. all possible castlings are allowed)
    pub const FULL: CastlingRights = CastlingRights(15);

    /// Returns `true` if color `c` is able to perform castling to side `s`
    #[inline]
    pub const fn has(&self, c: Color, s: CastlingSide) -> bool {
        ((self.0 >> Self::to_index(c, s)) & 1) != 0
    }

    /// Returns `true` if color `c` is able to perform castling to at least one of
    /// the sides.
    #[inline]
    pub const fn has_color(&self, c: Color) -> bool {
        (self.0 & Self::to_color_mask(c)) != 0
    }

    /// Adds `s` to allowed castling sides for color `c`
    #[inline]
    pub const fn with(self, c: Color, s: CastlingSide) -> CastlingRights {
        CastlingRights(self.0 | (1_u8 << Self::to_index(c, s)))
    }

    /// Removes `s` to allowed castling sides for color `c`
    #[inline]
    pub const fn without(self, c: Color, s: CastlingSide) -> CastlingRights {
        CastlingRights(self.0 & !(1_u8 << Self::to_index(c, s)))
    }

    /// Adds `s` to allowed castling sides for color `c`
    ///
    /// Unlike [`CastlingRights::with`], mutates the current object instead of returning
    /// a new one.
    #[inline]
    pub fn set(&mut self, c: Color, s: CastlingSide) {
        *self = self.with(c, s)
    }

    /// Removes `s` to allowed castling sides for color `c`
    ///
    /// Unlike [`CastlingRights::without`], mutates the current object instead of returning
    /// a new one.
    #[inline]
    pub fn unset(&mut self, c: Color, s: CastlingSide) {
        *self = self.without(c, s)
    }

    /// Removes all the castling rights for color `c`
    #[inline]
    pub fn unset_color(&mut self, c: Color) {
        self.unset(c, CastlingSide::King);
        self.unset(c, CastlingSide::Queen);
    }

    /// Creates [`CastlingRights`] from index
    ///
    /// # Panics
    ///
    /// The function panics if `val` is an invalid index.
    #[inline]
    pub const fn from_index(val: usize) -> CastlingRights {
        assert!(val < 16, "raw castling rights must be between 0 and 15");
        CastlingRights(val as u8)
    }

    /// Converts [`CastlingRights`] into an index
    ///
    /// Indices are stable between updates, and changing the index of some given [`CastlingRights`]
    /// instance is considered API breakage.
    #[inline]
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

/// Reason for game finish with draw
#[non_exhaustive]
#[derive(Debug, Display, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DrawReason {
    /// Draw by stalemate
    #[display(fmt = "stalemate")]
    Stalemate,
    /// Draw by insufficient material
    #[display(fmt = "insufficient material")]
    InsufficientMaterial,
    /// Draw by 75 moves
    ///
    /// This one is mandatory, in contrast with draw by 50 moves.
    #[display(fmt = "75 move rule")]
    Moves75,
    /// Draw by five-fold repetition
    ///
    /// This one is mandatory, in contrast with draw by threefold repetition.
    #[display(fmt = "fivefold repetition")]
    Repeat5,
    /// Draw by 50 moves
    ///
    /// According to FIDE rules, one can claim a draw if no player captures a piece or
    /// makes a pawn move during the last 50 moves, but is not obligated to do so.
    #[display(fmt = "50 move rule")]
    Moves50,
    /// Draw by threefold repetition
    ///
    /// In case of threefold repetition, one can claim a draw but is not obligated to do so.
    #[display(fmt = "threefold repetition")]
    Repeat3,
    /// Draw by agreement
    #[display(fmt = "draw by agreement")]
    Agreement,
    /// Reason is unknown
    #[display(fmt = "draw by unknown reason")]
    Unknown,
}

/// Reason for game finish with win
#[non_exhaustive]
#[derive(Debug, Display, Copy, Clone, PartialEq, Eq, Hash)]
pub enum WinReason {
    /// Game ends with checkmate
    #[display(fmt = "checkmate")]
    Checkmate,
    /// Opponent forfeits on time
    #[display(fmt = "opponent forfeits on time")]
    TimeForfeit,
    /// Opponent made an invalid move
    #[display(fmt = "opponent made an invalid move")]
    InvalidMove,
    /// Opponent is a chess engine and it either violated the protocol or crashed
    #[display(fmt = "opponent is a buggy chess engine")]
    EngineError,
    /// Opponent resigns
    #[display(fmt = "opponent resigns")]
    Resign,
    /// Opponent abandons the game
    #[display(fmt = "opponent abandons the game")]
    Abandon,
    /// Reason is unknown
    #[display(fmt = "unknown reason")]
    Unknown,
}

/// Outcome of the finished game
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Outcome {
    /// Win (either by White or by Black)
    Win {
        /// Winning side
        side: Color,
        /// Reason
        reason: WinReason,
    },
    /// Draw
    Draw(DrawReason),
}

/// Filter to group various types of outcomes
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum OutcomeFilter {
    /// Only outcomes with no legal moves are considered (i.e. checkmate and stalemate)
    Force,
    /// Only outcomes which are mandatorily applied by FIDE rules are considered
    ///
    /// This includes the following:
    /// - checkmate
    /// - stalemate
    /// - draw by insufficient material
    /// - draw by 75 moves
    /// - draw by five-fold repetition
    Strict,
    /// All the outcomes in [`Strict`](OutcomeFilter::Strict) plus the outcomes where
    /// player can claim a draw
    ///
    /// This additionally includes:
    /// - draw by 50 moves
    /// - draw by threefold repetitions
    Relaxed,
}

impl Outcome {
    /// Extracts the winner from the outcome
    ///
    /// If this is a draw outcome, then `None` is returned
    #[inline]
    pub fn winner(&self) -> Option<Color> {
        match self {
            Self::Win { side, .. } => Some(*side),
            Self::Draw(_) => None,
        }
    }

    /// Returns `true` if the outcome occured because one of the sides didn't have a legal move
    ///
    /// Similar to [`Outcome::passes(OutcomeFilter::Force)`](Outcome::passes)
    #[inline]
    pub fn is_force(&self) -> bool {
        matches!(
            *self,
            Self::Win {
                reason: WinReason::Checkmate,
                ..
            } | Self::Draw(DrawReason::Stalemate)
        )
    }

    /// Returns `true` if the outcome passes filter `filter`
    ///
    /// See [`OutcomeFilter`] docs for the details about each filter
    #[inline]
    pub fn passes(&self, filter: OutcomeFilter) -> bool {
        if self.is_force() {
            return true;
        }
        if matches!(filter, OutcomeFilter::Strict | OutcomeFilter::Relaxed)
            && matches!(
                *self,
                Self::Draw(
                    DrawReason::InsufficientMaterial | DrawReason::Moves75 | DrawReason::Repeat5
                )
            )
        {
            return true;
        }
        matches!(filter, OutcomeFilter::Relaxed)
            && matches!(*self, Self::Draw(DrawReason::Moves50 | DrawReason::Repeat3))
    }
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Draw(reason) => reason.fmt(f),
            Self::Win { side, reason } => match reason {
                WinReason::Checkmate => write!(f, "{} checkmates", side.as_long_str()),
                WinReason::TimeForfeit => {
                    write!(f, "{} forfeits on time", side.inv().as_long_str())
                }
                WinReason::InvalidMove => {
                    write!(f, "{} made an invalid move", side.inv().as_long_str())
                }
                WinReason::EngineError => {
                    write!(f, "{} is a buggy chess engine", side.inv().as_long_str())
                }
                WinReason::Resign => write!(f, "{} resigns", side.inv().as_long_str()),
                WinReason::Abandon => write!(f, "{} abandons the game", side.inv().as_long_str()),
                WinReason::Unknown => write!(f, "{} wins by unknown reason", side.as_long_str()),
            },
        }
    }
}

/// Short status of the game (either running of finished)
#[derive(Debug, Display, Copy, Clone, PartialEq, Eq, Hash)]
pub enum GameStatus {
    /// White wins
    #[display(fmt = "1-0")]
    White,
    /// Black wins
    #[display(fmt = "0-1")]
    Black,
    /// Draw
    #[display(fmt = "1/2-1/2")]
    Draw,
    /// Game is still running
    #[display(fmt = "*")]
    Running,
}

impl From<Option<Outcome>> for GameStatus {
    #[inline]
    fn from(src: Option<Outcome>) -> Self {
        match src {
            Some(Outcome::Win {
                side: Color::White, ..
            }) => Self::White,
            Some(Outcome::Win {
                side: Color::Black, ..
            }) => Self::Black,
            Some(Outcome::Draw(_)) => Self::Draw,
            None => Self::Running,
        }
    }
}

impl From<Outcome> for GameStatus {
    #[inline]
    fn from(src: Outcome) -> Self {
        Self::from(Some(src))
    }
}

impl From<&Outcome> for GameStatus {
    #[inline]
    fn from(src: &Outcome) -> Self {
        Self::from(Some(*src))
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
    fn test_piece() {
        for (idx, piece) in Piece::iter().enumerate() {
            assert_eq!(piece.index(), idx);
            assert_eq!(Piece::from_index(idx), piece);
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
        assert!(!empty.has_color(Color::White));
        assert!(!empty.has(Color::Black, CastlingSide::Queen));
        assert!(!empty.has(Color::Black, CastlingSide::King));
        assert!(!empty.has_color(Color::Black));
        assert_eq!(empty.to_string(), "-");
        assert_eq!(CastlingRights::from_str("-"), Ok(empty));

        let full = CastlingRights::FULL;
        assert!(full.has(Color::White, CastlingSide::Queen));
        assert!(full.has(Color::White, CastlingSide::King));
        assert!(full.has_color(Color::White));
        assert!(full.has(Color::Black, CastlingSide::Queen));
        assert!(full.has(Color::Black, CastlingSide::King));
        assert!(full.has_color(Color::Black));
        assert_eq!(full.to_string(), "KQkq");
        assert_eq!(CastlingRights::from_str("KQkq"), Ok(full));

        let mut rights = CastlingRights::EMPTY;
        rights.set(Color::White, CastlingSide::King);
        assert!(!rights.has(Color::White, CastlingSide::Queen));
        assert!(rights.has(Color::White, CastlingSide::King));
        assert!(rights.has_color(Color::White));
        assert!(!rights.has(Color::Black, CastlingSide::Queen));
        assert!(!rights.has(Color::Black, CastlingSide::King));
        assert!(!rights.has_color(Color::Black));
        assert_eq!(rights.to_string(), "K");
        assert_eq!(CastlingRights::from_str("K"), Ok(rights));

        rights.unset(Color::White, CastlingSide::King);
        rights.set(Color::Black, CastlingSide::Queen);
        assert!(!rights.has(Color::White, CastlingSide::Queen));
        assert!(!rights.has(Color::White, CastlingSide::King));
        assert!(!rights.has_color(Color::White));
        assert!(rights.has(Color::Black, CastlingSide::Queen));
        assert!(!rights.has(Color::Black, CastlingSide::King));
        assert!(rights.has_color(Color::Black));
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
