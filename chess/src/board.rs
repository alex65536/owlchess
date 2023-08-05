//! Board and related things

use crate::bitboard::Bitboard;
use crate::moves::Make;
use crate::types::{
    self, CastlingRights, CastlingSide, Cell, Color, Coord, DrawReason, File, Outcome, Piece, Rank,
    WinReason,
};
use crate::{bitboard_consts, geometry, movegen, zobrist};

use std::fmt::{self, Display};
use std::hash::{Hash, Hasher};
use std::num::ParseIntError;
use std::str::FromStr;

use thiserror::Error;

/// Board validation error
#[derive(Debug, Clone, Error, Eq, PartialEq)]
pub enum ValidateError {
    /// Invalid enpassant coordinate specified (i.e. it is located on an invalid rank)
    #[error("invalid enpassant position {0}")]
    InvalidEnpassant(Coord),
    /// Too many pieces of given color
    ///
    /// No more than 16 pieces of each color is allowed.
    #[error("too many pieces of color {0:?}")]
    TooManyPieces(Color),
    /// One of the sides doesn't have a king
    #[error("no king of color {0:?}")]
    NoKing(Color),
    /// One of the sides has more than one king
    #[error("more than one king of color {0:?}")]
    TooManyKings(Color),
    /// There is a pawn on the 1th or on the 8th rank
    #[error("invalid pawn position {0}")]
    InvalidPawn(Coord),
    /// Opponent's king is under attack
    #[error("opponent's king is attacked")]
    OpponentKingAttacked,
}

/// Error parsing the first part of FEN (i.e. the positions of pieces on the board)
#[derive(Debug, Clone, Error, Eq, PartialEq)]
pub enum CellsParseError {
    /// Rank is too large
    #[error("too many items in rank {0}")]
    RankOverflow(Rank),
    /// Rank is too small
    #[error("not enough items in rank {0}")]
    RankUnderflow(Rank),
    /// Too many ranks
    #[error("too many ranks")]
    Overflow,
    /// Not enough ranks
    #[error("not enough ranks")]
    Underflow,
    /// Unexpected character
    #[error("unexpected char {0:?}")]
    UnexpectedChar(char),
}

/// Error parsing [`RawBoard`] from FEN
#[derive(Debug, Clone, Error, Eq, PartialEq)]
pub enum RawFenParseError {
    /// FEN contains non-ASCII characters
    #[error("non-ASCII data in FEN")]
    NonAscii,
    /// FEN doesn't have board part
    #[error("board not specified")]
    NoBoard,
    /// Error parsing board from FEN
    #[error("bad board: {0}")]
    Board(#[from] CellsParseError),
    /// FEN doesn't have move side part
    #[error("no move side")]
    NoMoveSide,
    /// Error parsing move side from FEN
    #[error("bad move side: {0}")]
    MoveSide(#[from] types::ColorParseError),
    /// FEN doesn't have castling rights part
    #[error("no castling rights")]
    NoCastling,
    /// Error parsing castling rights from FEN
    #[error("bad castling rights: {0}")]
    Castling(#[from] types::CastlingRightsParseError),
    /// FEN doesn't have enpassant part
    #[error("no enpassant")]
    NoEnpassant,
    /// Error parsing enpassant from FEN
    #[error("bad enpassant: {0}")]
    Enpassant(#[from] types::CoordParseError),
    /// Enpassant rank is invalid
    #[error("invalid enpassant rank {0}")]
    InvalidEnpassantRank(Rank),
    /// Error parsing move counter
    #[error("bad move counter: {0}")]
    MoveCounter(ParseIntError),
    /// Error parsing move number
    #[error("bad move number: {0}")]
    MoveNumber(ParseIntError),
    /// FEN contains extra data
    #[error("extra data in FEN")]
    ExtraData,
}

/// Error parsing [`Board`] from FEN
#[derive(Debug, Clone, Error, Eq, PartialEq)]
pub enum FenParseError {
    /// Board cannot be parsed
    #[error("cannot parse fen: {0}")]
    Fen(#[from] RawFenParseError),
    /// Board was parsed, but it's invalid
    #[error("invalid position: {0}")]
    Valid(#[from] ValidateError),
}

/// Raw chess board
///
/// Raw board contains all the necessary information about the chess position. But, unlike [`Board`],
/// it is not validated and may contain an invalid position.
///
/// Raw board can be used to build or edit the position programmatically. After changing the necessary
/// fields, it must be converted to [`Board`] via [`Board::try_from()`].
///
/// # Example
///
/// ```
/// # use owlchess::{RawBoard, Board, File, Rank, Color, Piece, Cell, CastlingRights};
/// #
/// let mut raw = RawBoard {
///     cells: [Default::default(); 64],
///     side: Color::White,
///     castling: CastlingRights::EMPTY,
///     ep_source: None,
///     move_counter: 10,
///     move_number: 42,
/// };
/// raw.put2(File::B, Rank::R2, Cell::from_parts(Color::White, Piece::King));
/// raw.put2(File::D, Rank::R5, Cell::from_parts(Color::Black, Piece::King));
///
/// let board: Board = raw.try_into().unwrap();
/// assert_eq!(board.as_fen(), "8/8/8/3k4/8/8/1K6/8 w - - 10 42");
/// ```
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct RawBoard {
    /// Contents of the board
    ///
    /// The indices in this array are the indices of coordinates. You might probably want to use
    /// the functions like [`RawBoard::get()`] or [`RawBoard::put()`] instead of indexing this array
    /// directly.
    pub cells: [Cell; 64],
    /// Side to move
    pub side: Color,
    /// Castling rights
    pub castling: CastlingRights,
    /// En passant source square
    ///
    /// It is be equal to `None` if no enpassant is allowed. Otherwise, it contains
    /// the square with the pawn which can be captured by enpassant.
    ///
    /// If you want to obtain the destination square for the possible enpassant, see
    /// [`RawBoard::ep_dest()`].
    pub ep_source: Option<Coord>,
    /// Number of half-moves without pawn moves or captures
    ///
    /// It is required for draw by 50 or 75 moves.
    pub move_counter: u16,
    /// Move number
    ///
    /// Note that this is move number, not half-move number. It is incremented after each
    /// move by Black.
    pub move_number: u16,
}

impl RawBoard {
    /// Returns an empty `RawBoard`
    ///
    /// Does the same as [`RawBoard::default()`], except that this function is `const`.
    #[inline]
    pub const fn empty() -> RawBoard {
        RawBoard {
            cells: [Cell::EMPTY; 64],
            side: Color::White,
            castling: CastlingRights::EMPTY,
            ep_source: None,
            move_counter: 0,
            move_number: 1,
        }
    }

    /// Returns a board with the initial position
    pub fn initial() -> RawBoard {
        let mut res = RawBoard {
            cells: [Cell::EMPTY; 64],
            side: Color::White,
            castling: CastlingRights::FULL,
            ep_source: None,
            move_counter: 0,
            move_number: 1,
        };
        for file in File::iter() {
            res.put2(file, Rank::R2, Cell::from_parts(Color::White, Piece::Pawn));
            res.put2(file, Rank::R7, Cell::from_parts(Color::Black, Piece::Pawn));
        }
        for (color, rank) in [(Color::White, Rank::R1), (Color::Black, Rank::R8)] {
            res.put2(File::A, rank, Cell::from_parts(color, Piece::Rook));
            res.put2(File::B, rank, Cell::from_parts(color, Piece::Knight));
            res.put2(File::C, rank, Cell::from_parts(color, Piece::Bishop));
            res.put2(File::D, rank, Cell::from_parts(color, Piece::Queen));
            res.put2(File::E, rank, Cell::from_parts(color, Piece::King));
            res.put2(File::F, rank, Cell::from_parts(color, Piece::Bishop));
            res.put2(File::G, rank, Cell::from_parts(color, Piece::Knight));
            res.put2(File::H, rank, Cell::from_parts(color, Piece::Rook));
        }
        res
    }

    /// Parses a board from FEN
    ///
    /// Does the same as [`RawBoard::from_str`]. It is recommended to use this function instead of
    /// `from_str()` for better readability.
    #[inline]
    pub fn from_fen(fen: &str) -> Result<RawBoard, RawFenParseError> {
        RawBoard::from_str(fen)
    }

    /// Returns the contents of the square with coordinate `c`
    #[inline]
    pub fn get(&self, c: Coord) -> Cell {
        unsafe { *self.cells.get_unchecked(c.index()) }
    }

    /// Returns the contents of the square with file `file` and rank `rank`
    #[inline]
    pub fn get2(&self, file: File, rank: Rank) -> Cell {
        self.get(Coord::from_parts(file, rank))
    }

    /// Puts `cell` to the square with coordinate `c`
    #[inline]
    pub fn put(&mut self, c: Coord, cell: Cell) {
        unsafe {
            *self.cells.get_unchecked_mut(c.index()) = cell;
        }
    }

    /// Puts `cell` to the square with file `file` and rank `rank`
    #[inline]
    pub fn put2(&mut self, file: File, rank: Rank, cell: Cell) {
        self.put(Coord::from_parts(file, rank), cell);
    }

    /// Returns Zobrist hash of the board
    ///
    /// Note that Zobrist hash doesn't contain move counter and move number, so it can be used
    /// to detect draw by repetitions.
    ///
    /// By contrast, implementation of [`Hash`] trait includes move counter and move number into
    /// the hash.
    #[inline]
    pub fn zobrist_hash(&self) -> u64 {
        let mut hash = if self.side == Color::White {
            zobrist::MOVE_SIDE
        } else {
            0
        };
        if let Some(p) = self.ep_source {
            hash ^= zobrist::enpassant(p);
        }
        hash ^= zobrist::castling(self.castling);
        for (i, cell) in self.cells.iter().enumerate() {
            if cell.is_occupied() {
                hash ^= zobrist::pieces(*cell, Coord::from_index(i));
            }
        }
        hash
    }

    /// Returns `None` if no enpassant is allowed. Otherwise, returns the destination
    /// square for the possible enpassant.
    ///
    /// If the rank of `self.ep_source` is not a valid rank for enpassant source (i.e not
    /// equal to [`Rank::R5`] for White and [`Rank::R4`] for Black), then the function just
    /// returns the result as if the rank was valid.
    #[inline]
    pub fn ep_dest(&self) -> Option<Coord> {
        let p = self.ep_source?;
        Some(Coord::from_parts(
            p.file(),
            geometry::enpassant_dst_rank(self.side),
        ))
    }

    /// Wraps the board to allow pretty-printing with the given style `Style`
    ///
    /// The resulting wrapper implements [`fmt::Display`], so can be used with
    /// `write!()`, `println!()`, or `ToString::to_string`.
    ///
    /// # Example
    ///
    /// ```
    /// # use owlchess::{RawBoard, board::PrettyStyle};
    /// #
    /// let r = RawBoard::initial();
    ///
    /// let res = r#"
    /// 8|rnbqkbnr
    /// 7|pppppppp
    /// 6|........
    /// 5|........
    /// 4|........
    /// 3|........
    /// 2|PPPPPPPP
    /// 1|RNBQKBNR
    /// -+--------
    /// W|abcdefgh
    /// "#;
    /// assert_eq!(r.pretty(PrettyStyle::Ascii).to_string().trim(), res.trim());
    ///
    /// let res = r#"
    /// 8│♜♞♝♛♚♝♞♜
    /// 7│♟♟♟♟♟♟♟♟
    /// 6│........
    /// 5│........
    /// 4│........
    /// 3│........
    /// 2│♙♙♙♙♙♙♙♙
    /// 1│♖♘♗♕♔♗♘♖
    /// ─┼────────
    /// ○│abcdefgh
    /// "#;
    /// assert_eq!(r.pretty(PrettyStyle::Utf8).to_string().trim(), res.trim());
    /// ```
    #[inline]
    pub fn pretty(&self, style: PrettyStyle) -> Pretty<'_> {
        Pretty { raw: self, style }
    }

    /// Converts the board into a FEN string
    ///
    /// Does the same as `RawBoard::to_string()`. It is recommended to use this function instead of
    /// `to_string()` for better readability.
    #[inline]
    pub fn as_fen(&self) -> String {
        self.to_string()
    }
}

impl Default for RawBoard {
    #[inline]
    fn default() -> RawBoard {
        RawBoard::empty()
    }
}

/// Board that contains a valid position
///
/// This board always contains a valid chess position. It is used for literally every chess operation:
/// move generation, making and validating moves, verifying for check and checkmate.
///
/// It contains a [`RawBoard`] alongside with auxilliary structures to make all the chess operations
/// faster.
///
/// # Safety
///
/// The board must be always valid (i. e. `Ok(b.clone()) == b.raw().try_into()` must always hold). The
/// only allowed exception is attack on the opponent's king after making a semi-legal move. In this case,
/// you must call [`Board::is_opponent_king_attacked()`] and undo the offending before doing anything else.
/// Alternatively, you can just drop the invalid board. Other actions with the board when the opponent's
/// king is under attack are considered undefined behavior.
#[derive(Debug, Clone)]
pub struct Board {
    pub(crate) r: RawBoard,
    pub(crate) hash: u64,
    pub(crate) white: Bitboard,
    pub(crate) black: Bitboard,
    pub(crate) all: Bitboard,
    pub(crate) pieces: [Bitboard; Cell::COUNT],
}

impl Board {
    /// Returns a board with the initial position
    pub fn initial() -> Board {
        RawBoard::initial().try_into().unwrap()
    }

    /// Parses a board from FEN
    ///
    /// Does the same as [`Board::from_str`]. It is recommended to use this function instead of
    /// `from_str()` for better readability.
    pub fn from_fen(fen: &str) -> Result<Board, FenParseError> {
        Board::from_str(fen)
    }

    /// Returns a view over the raw board
    #[inline]
    pub fn raw(&self) -> &RawBoard {
        &self.r
    }

    /// Returns the contents of the square with coordinate `c`
    #[inline]
    pub fn get(&self, c: Coord) -> Cell {
        self.r.get(c)
    }

    /// Returns the contents of the square with file `file` and rank `rank`
    #[inline]
    pub fn get2(&self, file: File, rank: Rank) -> Cell {
        self.r.get2(file, rank)
    }

    /// Returns side to move
    #[inline]
    pub fn side(&self) -> Color {
        self.r.side
    }

    /// Returns the bitboard over all the pieces with color `c`
    ///
    /// This function is quite fast, as it doesn't compute anything and just returns the
    /// stored value.
    #[inline]
    pub fn color(&self, c: Color) -> Bitboard {
        if c == Color::White {
            self.white
        } else {
            self.black
        }
    }

    #[inline]
    pub(crate) fn color_mut(&mut self, c: Color) -> &mut Bitboard {
        if c == Color::White {
            &mut self.white
        } else {
            &mut self.black
        }
    }

    /// Returns the bitboard over all the cells equal to `c`
    ///
    /// **Note**: when `c` is an empty cell, the function just returns an empty bitboard,
    /// not the bitboard over all the empty cells.
    ///
    /// This function is quite fast, as it doesn't compute anything and just returns the
    /// stored value.
    #[inline]
    pub fn piece(&self, c: Cell) -> Bitboard {
        unsafe { *self.pieces.get_unchecked(c.index()) }
    }

    /// Returns the bitboard over all the pieces of color `c` and kind `p`
    ///
    /// This function is quite fast, as it doesn't compute anything and just returns the
    /// stored value.
    #[inline]
    pub fn piece2(&self, c: Color, p: Piece) -> Bitboard {
        self.piece(Cell::from_parts(c, p))
    }

    #[inline]
    pub(crate) fn piece_diag(&self, c: Color) -> Bitboard {
        self.piece2(c, Piece::Bishop) | self.piece2(c, Piece::Queen)
    }

    #[inline]
    pub(crate) fn piece_line(&self, c: Color) -> Bitboard {
        self.piece2(c, Piece::Rook) | self.piece2(c, Piece::Queen)
    }

    #[inline]
    pub(crate) fn piece_mut(&mut self, c: Cell) -> &mut Bitboard {
        unsafe { self.pieces.get_unchecked_mut(c.index()) }
    }

    /// Returns the position of the king of color `c`
    #[inline]
    pub fn king_pos(&self, c: Color) -> Coord {
        self.piece(Cell::from_parts(c, Piece::King))
            .into_iter()
            .next()
            .unwrap()
    }

    /// Returns the Zobrist hash of the position
    ///
    /// This function is quite fast, as it doesn't compute anything and just returns the
    /// stored value.
    ///
    /// Note that Zobrist hash doesn't contain move counter and move number, so it can be used
    /// to detect draw by repetitions.
    ///
    /// By contrast, implementation of [`Hash`] trait includes move counter and move number into
    /// the hash.
    ///
    /// Unlike [`RawBoard::zobrist_hash`], this function just returns the precomputed value and
    /// doesn't try to  recalculate the hash from scratch.
    #[inline]
    pub fn zobrist_hash(&self) -> u64 {
        self.hash
    }

    /// Convenience alias for [`moves::Make::make`](crate::moves::Make::make)
    pub fn make_move<M: Make>(&self, m: M) -> Result<Self, M::Err> {
        m.make(self)
    }

    /// Returns `true` if the opponent's king is under attack
    ///
    /// If it is under attack, you must undo the offending move before doing anything else. See doc for
    /// [`Board`] for more details.
    #[inline]
    pub fn is_opponent_king_attacked(&self) -> bool {
        let c = self.r.side;
        movegen::is_cell_attacked(self, self.king_pos(c.inv()), c)
    }

    /// Returns `true` if the current side has at least one legal move
    #[inline]
    pub fn has_legal_moves(&self) -> bool {
        movegen::has_legal_moves(self)
    }

    /// Returns `true` if the current side is in check
    #[inline]
    pub fn is_check(&self) -> bool {
        let c = self.r.side;
        movegen::is_cell_attacked(self, self.king_pos(c), c.inv())
    }

    /// Returns all the pieces that give check currently
    #[inline]
    pub fn checkers(&self) -> Bitboard {
        let c = self.r.side;
        movegen::cell_attackers(self, self.king_pos(c), c.inv())
    }

    /// Returns `true` if the position is guaranteed to be drawn because of insufficient material, regardless
    /// of the players' moves
    ///
    /// Currently, such positions include:
    ///
    /// - king vs king
    /// - king + knight vs king
    /// - kings and bishops of the same color
    ///
    /// Note that king + knight vs king + knight is not considered a draw, as one of the sides can intentionally
    /// corner itself, allowing its opponent to win.
    fn is_insufficient_material(&self) -> bool {
        let all_without_kings = self.all
            ^ (self.piece2(Color::White, Piece::King) | self.piece2(Color::Black, Piece::King));

        // If we have pieces on both white and black squares, then no draw occurs. This cutoff
        // optimizes the function in most positions.
        if (all_without_kings & bitboard_consts::CELLS_WHITE).is_nonempty()
            && (all_without_kings & bitboard_consts::CELLS_BLACK).is_nonempty()
        {
            return false;
        }

        // Two kings only
        if all_without_kings.is_empty() {
            return true;
        }

        // King vs king + knight
        let knights =
            self.piece2(Color::White, Piece::Knight) | self.piece2(Color::Black, Piece::Knight);
        if all_without_kings == knights && knights.len() == 1 {
            return true;
        }

        // Kings and bishops of the same cell color. Note that we checked above that all the pieces
        // have the same cell color, so we just need to ensure that all the pieces are bishops.
        let bishops =
            self.piece2(Color::White, Piece::Bishop) | self.piece2(Color::Black, Piece::Bishop);
        if all_without_kings == bishops {
            return true;
        }

        false
    }

    /// Calculates the current outcome on the board
    ///
    /// This function ignores draws by repetition, as [`Board`] doesn't remember the previous
    /// positions. You need to use [`MoveChain`](crate::chain::MoveChain) if you want to consider
    /// such draws.
    ///
    /// This function can be computationally expensive, as it calls [`movegen::has_legal_moves`].
    ///
    /// When calculating outcomes, outcomes passing [`OutcomeFilter::Force`](crate::types::OutcomeFilter::Force)
    /// are the most prioritized, and outcomes not passing [`OutcomeFilter::Strict`](crate::types::OutcomeFilter::Strict)
    /// are the least prioritized.
    #[inline]
    pub fn calc_outcome(&self) -> Option<Outcome> {
        // First, we verify for checkmate or stalemate, as force outcome take precedence over
        // non-force ones.
        if !self.has_legal_moves() {
            return if self.is_check() {
                Some(Outcome::Win {
                    side: self.r.side.inv(),
                    reason: WinReason::Checkmate,
                })
            } else {
                Some(Outcome::Draw(DrawReason::Stalemate))
            };
        }

        if let Some(draw) = self.calc_draw_simple() {
            return Some(Outcome::Draw(draw));
        }

        None
    }

    /// Calculates the current outcome on the board, considering only draws by insufficient material,
    /// by 50 and 75 move rules.
    ///
    /// For the details about outcome priority, see docs for [`Board::calc_outcome()`]
    #[inline]
    pub fn calc_draw_simple(&self) -> Option<DrawReason> {
        // Check for insufficient material
        if self.is_insufficient_material() {
            return Some(DrawReason::InsufficientMaterial);
        }

        // Check for 50/75 move rule. Note that check for 50 move rule must
        // come after all other ones, because it is non-strict.
        if self.r.move_counter >= 150 {
            return Some(DrawReason::Moves75);
        }
        if self.r.move_counter >= 100 {
            return Some(DrawReason::Moves50);
        }

        None
    }

    /// Wraps the board to allow pretty-printing with the given style `Style`
    ///
    /// The resulting wrapper implements [`fmt::Display`], so can be used with
    /// `write!()`, `println!()`, or `ToString::to_string`.
    ///
    /// See docs for [`RawBoard::pretty()`] for more usage details.
    #[inline]
    pub fn pretty(&self, style: PrettyStyle) -> Pretty<'_> {
        self.r.pretty(style)
    }

    /// Converts the board into a FEN string
    ///
    /// Does the same as `Board::to_string()`. It is recommended to use this function instead of
    /// `to_string()` for better readability.
    #[inline]
    pub fn as_fen(&self) -> String {
        self.to_string()
    }
}

impl PartialEq for Board {
    #[inline]
    fn eq(&self, other: &Board) -> bool {
        self.r == other.r
    }
}

impl Eq for Board {}

impl Hash for Board {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.r.hash(state)
    }
}

impl TryFrom<RawBoard> for Board {
    type Error = ValidateError;

    fn try_from(mut raw: RawBoard) -> Result<Board, ValidateError> {
        // Check enpassant
        if let Some(p) = raw.ep_source {
            // Check InvalidEnpassant
            if p.rank() != geometry::enpassant_src_rank(raw.side) {
                return Err(ValidateError::InvalidEnpassant(p));
            }

            // Reset enpassant if either there is no pawn or the cell on the pawn's path is occupied
            let pp = p.add(geometry::pawn_forward_delta(raw.side));
            if raw.get(p) != Cell::from_parts(raw.side.inv(), Piece::Pawn)
                || raw.get(pp) != Cell::EMPTY
            {
                raw.ep_source = None;
            }
        }

        // Reset bad castling flags
        for color in [Color::White, Color::Black] {
            let rank = geometry::castling_rank(color);
            if raw.get2(File::E, rank) != Cell::from_parts(color, Piece::King) {
                raw.castling.unset(color, CastlingSide::Queen);
                raw.castling.unset(color, CastlingSide::King);
            }
            if raw.get2(File::A, rank) != Cell::from_parts(color, Piece::Rook) {
                raw.castling.unset(color, CastlingSide::Queen);
            }
            if raw.get2(File::H, rank) != Cell::from_parts(color, Piece::Rook) {
                raw.castling.unset(color, CastlingSide::King);
            }
        }

        // Calculate bitboards
        let mut white = Bitboard::EMPTY;
        let mut black = Bitboard::EMPTY;
        let mut pieces = [Bitboard::EMPTY; Cell::COUNT];
        for (idx, cell) in raw.cells.iter().enumerate() {
            let coord = Coord::from_index(idx);
            if let Some(color) = cell.color() {
                match color {
                    Color::White => white.set(coord),
                    Color::Black => black.set(coord),
                };
                pieces[cell.index()].set(coord);
            }
        }

        // Check TooManyPieces, NoKing, TooManyKings
        if white.len() > 16 {
            return Err(ValidateError::TooManyPieces(Color::White));
        }
        if black.len() > 16 {
            return Err(ValidateError::TooManyPieces(Color::Black));
        }
        let white_king = pieces[Cell::from_parts(Color::White, Piece::King).index()];
        let black_king = pieces[Cell::from_parts(Color::White, Piece::King).index()];
        if white_king.is_empty() {
            return Err(ValidateError::NoKing(Color::White));
        }
        if black_king.is_empty() {
            return Err(ValidateError::NoKing(Color::Black));
        }
        if white_king.len() > 1 {
            return Err(ValidateError::TooManyKings(Color::White));
        }
        if black_king.len() > 1 {
            return Err(ValidateError::TooManyKings(Color::Black));
        }

        // Check InvalidPawn
        let pawns = pieces[Cell::from_parts(Color::White, Piece::Pawn).index()]
            | pieces[Cell::from_parts(Color::Black, Piece::Pawn).index()];
        const BAD_PAWN_POSES: Bitboard = Bitboard::from_raw(0xff000000000000ff);
        let bad_pawns = pawns & BAD_PAWN_POSES;
        if bad_pawns.is_nonempty() {
            return Err(ValidateError::InvalidPawn(
                bad_pawns.into_iter().next().unwrap(),
            ));
        }

        // Check OpponentKingAttacked
        let res = Board {
            r: raw,
            hash: raw.zobrist_hash(),
            white,
            black,
            all: white | black,
            pieces,
        };
        if res.is_opponent_king_attacked() {
            return Err(ValidateError::OpponentKingAttacked);
        }

        Ok(res)
    }
}

impl TryFrom<&RawBoard> for Board {
    type Error = ValidateError;

    fn try_from(raw: &RawBoard) -> Result<Board, ValidateError> {
        (*raw).try_into()
    }
}

/// Style for [`RawBoard::pretty()`] and [`Board::pretty()`]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum PrettyStyle {
    /// Print pieces and frames as ASCII characters
    Ascii,
    /// Print pieces and frames as fancy Unicode characters
    Utf8,
}

/// Wrapper to pretty-print the board
///
/// See docs for [`RawBoard::pretty()`] for more details.
pub struct Pretty<'a> {
    raw: &'a RawBoard,
    style: PrettyStyle,
}

fn parse_cells(s: &str) -> Result<[Cell; 64], CellsParseError> {
    type Error = CellsParseError;

    let mut file = 0_usize;
    let mut rank = 0_usize;
    let mut pos = 0_usize;
    let mut cells = [Cell::EMPTY; 64];
    for b in s.bytes() {
        match b {
            b'1'..=b'8' => {
                let add = (b - b'0') as usize;
                if file + add > 8 {
                    return Err(CellsParseError::RankOverflow(Rank::from_index(rank)));
                }
                file += add;
                pos += add;
            }
            b'/' => {
                if file < 8 {
                    return Err(Error::RankUnderflow(Rank::from_index(rank)));
                }
                rank += 1;
                file = 0;
                if rank >= 8 {
                    return Err(Error::Overflow);
                }
            }
            _ => {
                if file >= 8 {
                    return Err(Error::RankOverflow(Rank::from_index(rank)));
                }
                cells[pos] = Cell::from_char(b as char).ok_or(Error::UnexpectedChar(b as char))?;
                file += 1;
                pos += 1;
            }
        };
    }

    if file < 8 {
        return Err(Error::RankUnderflow(Rank::from_index(rank)));
    }
    if rank < 7 {
        return Err(Error::Underflow);
    }
    assert_eq!(file, 8);
    assert_eq!(rank, 7);
    assert_eq!(pos, 64);

    Ok(cells)
}

fn parse_ep_source(s: &str, side: Color) -> Result<Option<Coord>, RawFenParseError> {
    if s == "-" {
        return Ok(None);
    }
    let enpassant = Coord::from_str(s)?;
    if enpassant.rank() != geometry::enpassant_dst_rank(side) {
        return Err(RawFenParseError::InvalidEnpassantRank(enpassant.rank()));
    }
    Ok(Some(Coord::from_parts(
        enpassant.file(),
        geometry::enpassant_src_rank(side),
    )))
}

impl FromStr for RawBoard {
    type Err = RawFenParseError;

    fn from_str(s: &str) -> Result<RawBoard, Self::Err> {
        type Error = RawFenParseError;

        if !s.is_ascii() {
            return Err(Error::NonAscii);
        }
        let mut iter = s.split(' ').fuse();

        let cells = parse_cells(iter.next().ok_or(Error::NoBoard)?)?;
        let side = Color::from_str(iter.next().ok_or(Error::NoMoveSide)?)?;
        let castling = CastlingRights::from_str(iter.next().ok_or(Error::NoCastling)?)?;
        let ep_source = parse_ep_source(iter.next().ok_or(Error::NoEnpassant)?, side)?;
        let move_counter = match iter.next() {
            Some(s) => u16::from_str(s).map_err(Error::MoveCounter)?,
            None => 0,
        };
        let move_number = match iter.next() {
            Some(s) => u16::from_str(s).map_err(Error::MoveNumber)?,
            None => 1,
        };

        if iter.next().is_some() {
            return Err(Error::ExtraData);
        }

        Ok(RawBoard {
            cells,
            side,
            castling,
            ep_source,
            move_counter,
            move_number,
        })
    }
}

impl FromStr for Board {
    type Err = FenParseError;

    fn from_str(s: &str) -> Result<Board, Self::Err> {
        Ok(RawBoard::from_str(s)?.try_into()?)
    }
}

fn format_cells(cells: &[Cell; 64], f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
    for rank in Rank::iter() {
        if rank.index() != 0 {
            write!(f, "/")?;
        }
        let mut empty = 0;
        for file in File::iter() {
            let cell = cells[Coord::from_parts(file, rank).index()];
            if cell.is_free() {
                empty += 1;
                continue;
            }
            if empty != 0 {
                write!(f, "{}", (b'0' + empty) as char)?;
                empty = 0;
            }
            write!(f, "{}", cell)?;
        }
        if empty != 0 {
            write!(f, "{}", (b'0' + empty) as char)?;
        }
    }
    Ok(())
}

impl Display for RawBoard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        format_cells(&self.cells, f)?;
        write!(f, " {} {}", self.side, self.castling)?;
        match self.ep_dest() {
            Some(p) => write!(f, " {}", p)?,
            None => write!(f, " -")?,
        };
        write!(f, " {} {}", self.move_counter, self.move_number)?;
        Ok(())
    }
}

impl Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        self.r.fmt(f)
    }
}

trait StyleTable {
    const HORZ_FRAME: char;
    const VERT_FRAME: char;
    const ANGLE_FRAME: char;
    const WHITE_INDICATOR: char;
    const BLACK_INDICATOR: char;

    fn cell(c: Cell) -> char;

    fn indicator(c: Color) -> char {
        match c {
            Color::White => Self::WHITE_INDICATOR,
            Color::Black => Self::BLACK_INDICATOR,
        }
    }

    fn fmt(r: &RawBoard, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        for rank in Rank::iter() {
            write!(f, "{}{}", rank, Self::VERT_FRAME)?;
            for file in File::iter() {
                write!(f, "{}", Self::cell(r.get2(file, rank)))?;
            }
            writeln!(f)?;
        }
        write!(f, "{}{}", Self::HORZ_FRAME, Self::ANGLE_FRAME)?;
        for _ in File::iter() {
            write!(f, "{}", Self::HORZ_FRAME)?;
        }
        writeln!(f)?;
        write!(f, "{}{}", Self::indicator(r.side), Self::VERT_FRAME)?;
        for file in File::iter() {
            write!(f, "{}", file)?;
        }
        writeln!(f)?;
        Ok(())
    }
}

struct AsciiStyleTable;
struct Utf8StyleTable;

impl StyleTable for AsciiStyleTable {
    const HORZ_FRAME: char = '-';
    const VERT_FRAME: char = '|';
    const ANGLE_FRAME: char = '+';
    const WHITE_INDICATOR: char = 'W';
    const BLACK_INDICATOR: char = 'B';

    fn cell(c: Cell) -> char {
        c.as_char()
    }
}

impl StyleTable for Utf8StyleTable {
    const HORZ_FRAME: char = '─';
    const VERT_FRAME: char = '│';
    const ANGLE_FRAME: char = '┼';
    const WHITE_INDICATOR: char = '○';
    const BLACK_INDICATOR: char = '●';

    fn cell(c: Cell) -> char {
        c.as_utf8_char()
    }
}

impl<'a> Display for Pretty<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self.style {
            PrettyStyle::Ascii => AsciiStyleTable::fmt(self.raw, f),
            PrettyStyle::Utf8 => Utf8StyleTable::fmt(self.raw, f),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DrawReason, Outcome, WinReason};
    use std::mem;

    #[test]
    fn test_size() {
        assert_eq!(mem::size_of::<RawBoard>(), 72);
        assert_eq!(mem::size_of::<Board>(), 208);
    }

    #[test]
    fn test_initial() {
        const INI_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

        assert_eq!(RawBoard::initial().to_string(), INI_FEN);
        assert_eq!(Board::initial().to_string(), INI_FEN);
        assert_eq!(RawBoard::from_str(INI_FEN), Ok(RawBoard::initial()));
        assert_eq!(Board::from_str(INI_FEN), Ok(Board::initial()));
    }

    #[test]
    fn test_midgame() {
        const FEN: &str =
            "1rq1r1k1/1p3ppp/pB3n2/3ppP2/Pbb1P3/1PN2B2/2P2QPP/R1R4K w - - 1 21";

        let board = Board::from_fen(FEN).unwrap();
        assert_eq!(board.as_fen(), FEN);
        assert_eq!(
            board.get2(File::B, Rank::R4),
            Cell::from_parts(Color::Black, Piece::Bishop)
        );
        assert_eq!(
            board.get2(File::F, Rank::R2),
            Cell::from_parts(Color::White, Piece::Queen)
        );
        assert_eq!(
            board.king_pos(Color::White),
            Coord::from_parts(File::H, Rank::R1)
        );
        assert_eq!(
            board.king_pos(Color::Black),
            Coord::from_parts(File::G, Rank::R8)
        );
        assert_eq!(board.raw().side, Color::White);
        assert_eq!(board.raw().castling, CastlingRights::EMPTY);
        assert_eq!(board.raw().ep_source, None);
        assert_eq!(board.raw().move_counter, 1);
        assert_eq!(board.raw().move_number, 21);
    }

    #[test]
    fn test_fixes() {
        const FEN: &str =
            "r1bq1b1r/ppppkppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK1R1 w KQkq c6 6 5";

        let raw = RawBoard::from_fen(FEN).unwrap();
        assert_eq!(raw.castling, CastlingRights::FULL);
        assert_eq!(raw.ep_source, Some(Coord::from_parts(File::C, Rank::R5)));
        assert_eq!(raw.ep_dest(), Some(Coord::from_parts(File::C, Rank::R6)));
        assert_eq!(raw.as_fen(), FEN);

        let board: Board = raw.try_into().unwrap();
        assert_eq!(
            board.raw().castling,
            CastlingRights::EMPTY.with(Color::White, CastlingSide::Queen)
        );
        assert_eq!(board.raw().ep_source, None);
        assert_eq!(board.raw().ep_dest(), None);
        assert_eq!(
            board.as_fen(),
            "r1bq1b1r/ppppkppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK1R1 w Q - 6 5"
        );
    }

    #[test]
    fn test_incomplete() {
        assert_eq!(
            RawBoard::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"),
            Err(RawFenParseError::NoMoveSide)
        );

        assert_eq!(
            RawBoard::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w"),
            Err(RawFenParseError::NoCastling)
        );

        assert_eq!(
            RawBoard::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq"),
            Err(RawFenParseError::NoEnpassant)
        );

        let raw =
            RawBoard::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -").unwrap();
        assert_eq!(raw.move_counter, 0);
        assert_eq!(raw.move_number, 1);

        let raw =
            RawBoard::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 10").unwrap();
        assert_eq!(raw.move_counter, 10);
        assert_eq!(raw.move_number, 1);
    }

    #[test]
    fn test_outcome() {
        let b = Board::initial();
        assert_eq!(b.calc_outcome(), None);

        let b = Board::from_fen("rn1qkbnr/ppp2B1p/3p2p1/4N3/4P3/2N5/PPPP1PPP/R1BbK2R b KQkq - 0 6")
            .unwrap();
        assert_eq!(b.calc_outcome(), None);

        let b = Board::from_fen("rn1q1bnr/ppp1kB1p/3p2p1/3NN3/4P3/8/PPPP1PPP/R1BbK2R b KQ - 2 7")
            .unwrap();
        assert!(!b.has_legal_moves());
        assert_eq!(
            b.calc_outcome(),
            Some(Outcome::Win {
                side: Color::White,
                reason: WinReason::Checkmate
            })
        );

        let b = Board::from_fen("7K/8/5n2/5n2/8/8/7k/8 w - - 0 1").unwrap();
        assert!(!b.has_legal_moves());
        assert_eq!(b.calc_outcome(), Some(Outcome::Draw(DrawReason::Stalemate)));

        let b = Board::from_fen("7K/8/5n2/8/8/8/7k/8 w - - 0 1").unwrap();
        assert_eq!(
            b.calc_outcome(),
            Some(Outcome::Draw(DrawReason::InsufficientMaterial))
        );

        let b = Board::from_fen("7K/8/5b2/8/8/8/7k/8 w - - 0 1").unwrap();
        assert_eq!(
            b.calc_outcome(),
            Some(Outcome::Draw(DrawReason::InsufficientMaterial))
        );

        let b = Board::from_fen("2K4k/8/8/8/B1B5/1B1B4/B1B5/1B1B4 w - - 0 1").unwrap();
        assert_eq!(
            b.calc_outcome(),
            Some(Outcome::Draw(DrawReason::InsufficientMaterial))
        );

        let b = Board::from_fen("BBK4k/8/8/8/8/8/8/8 w - - 0 1").unwrap();
        assert_eq!(b.calc_outcome(), None);

        let b = Board::from_fen("NNK4k/8/8/8/8/8/8/8 w - - 0 1").unwrap();
        assert_eq!(b.calc_outcome(), None);

        let b = Board::from_fen("NNK4k/8/8/8/8/8/8/8 w - - 99 80").unwrap();
        assert_eq!(b.calc_outcome(), None);

        let b = Board::from_fen("NNK4k/8/8/8/8/8/8/8 w - - 100 80").unwrap();
        assert_eq!(b.calc_outcome(), Some(Outcome::Draw(DrawReason::Moves50)));

        let b = Board::from_fen("NNK4k/8/8/8/8/8/8/8 w - - 150 90").unwrap();
        assert_eq!(b.calc_outcome(), Some(Outcome::Draw(DrawReason::Moves75)));
    }
}
