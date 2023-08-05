use super::{san, uci};
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::legal::{Checker, NilPrechecker};
use crate::types::{CastlingRights, CastlingSide, Cell, Color, Coord, File, Piece, Rank};
use crate::{attack, between, castling, generic, geometry, movegen, zobrist};

use std::fmt;
use std::str::FromStr;

use thiserror::Error;

/// Move kind
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MoveKind {
    /// Null move
    #[default]
    Null = 0,
    /// Non-pawn move or capture (except castling)
    Simple = 1,
    /// Kingside castling
    CastlingKingside = 2,
    /// Queenside castling
    CastlingQueenside = 3,
    /// Single pawn move (either non-capture or capture)
    PawnSimple = 4,
    /// Double pawn move
    PawnDouble = 5,
    /// Enpassant
    Enpassant = 6,
    /// Pawn promote to knight (either non-capture or capture)
    PromoteKnight = 7,
    /// Pawn promote to bishop (either non-capture or capture)
    PromoteBishop = 8,
    /// Pawn promote to rook (either non-capture or capture)
    PromoteRook = 9,
    /// Pawn promote to queen (either non-capture or capture)
    PromoteQueen = 10,
}

/// Target piece for promotion
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PromotePiece {
    Knight = 2,
    Bishop = 3,
    Rook = 4,
    Queen = 5,
}

/// Move output style
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Style {
    /// Output in SAN format, with capical Latin letters for pieces
    San,
    /// Output in SAN format, with Unicode chess symbols for pieces
    SanUtf8,
    /// Output in UCI format
    Uci,
}

impl From<PromotePiece> for Piece {
    #[inline]
    fn from(p: PromotePiece) -> Self {
        match p {
            PromotePiece::Knight => Piece::Knight,
            PromotePiece::Bishop => Piece::Bishop,
            PromotePiece::Rook => Piece::Rook,
            PromotePiece::Queen => Piece::Queen,
        }
    }
}

impl TryFrom<Piece> for PromotePiece {
    type Error = ();

    #[inline]
    fn try_from(p: Piece) -> Result<Self, Self::Error> {
        match p {
            Piece::Knight => Ok(PromotePiece::Knight),
            Piece::Bishop => Ok(PromotePiece::Bishop),
            Piece::Rook => Ok(PromotePiece::Rook),
            Piece::Queen => Ok(PromotePiece::Queen),
            _ => Err(()),
        }
    }
}

impl From<CastlingSide> for MoveKind {
    #[inline]
    fn from(side: CastlingSide) -> Self {
        match side {
            CastlingSide::King => Self::CastlingKingside,
            CastlingSide::Queen => Self::CastlingQueenside,
        }
    }
}

impl TryFrom<MoveKind> for CastlingSide {
    type Error = ();

    #[inline]
    fn try_from(kind: MoveKind) -> Result<Self, Self::Error> {
        match kind {
            MoveKind::CastlingKingside => Ok(Self::King),
            MoveKind::CastlingQueenside => Ok(Self::Queen),
            _ => Err(()),
        }
    }
}

impl From<PromotePiece> for MoveKind {
    #[inline]
    fn from(kind: PromotePiece) -> Self {
        match kind {
            PromotePiece::Knight => Self::PromoteKnight,
            PromotePiece::Bishop => Self::PromoteBishop,
            PromotePiece::Rook => Self::PromoteRook,
            PromotePiece::Queen => Self::PromoteQueen,
        }
    }
}

impl TryFrom<MoveKind> for PromotePiece {
    type Error = ();

    #[inline]
    fn try_from(kind: MoveKind) -> Result<Self, Self::Error> {
        match kind {
            MoveKind::PromoteKnight => Ok(Self::Knight),
            MoveKind::PromoteBishop => Ok(Self::Bishop),
            MoveKind::PromoteRook => Ok(Self::Rook),
            MoveKind::PromoteQueen => Ok(Self::Queen),
            _ => Err(()),
        }
    }
}

impl MoveKind {
    /// Returns the piece after promote is this move kind represents a promote
    ///
    /// Otherwise, returns `None`.
    #[inline]
    pub fn promote(self) -> Option<Piece> {
        let piece: PromotePiece = self.try_into().ok()?;
        Some(piece.into())
    }
}

/// Chess move
///
/// Represents a chess move which can be applied to the chess board.
///
/// Moves can have different degrees of validity:
///
/// - _Well-formed_. A move is considered well-formed if there exists a position in which this move
///   is semilegal. Note that such existence is only a sufficient condition, so you may create a
///   well-formed move which would not be semilegal in any position. It is determined by [`Move::is_well_formed()`]
///   whether the move is well-formed.
///
///   Null move is explicitly well-formed.
///
///   Note that using non-well-formed moves other than checking them via [`Move::is_well_formed()`] or
///   examining their fields via getters, is undefined behavior.
///
/// - _Semilegal_. A move is considered semi-legal if it's valid by the rules of chess, except that the king can
///   remain under attack after such move.
///
///   Null move is not considered semilegal (but see the [notes below](#null-move)).
///
/// - _Legal_. A move is considered legal if it's semilegal plus the king doesn't remain under attack. So, such move
///   is fully valid by the rules of check.
///
/// # Null move
///
/// Null move is a move that just flips the move side, without changing the position. Such move doesn't exist
/// is chess, but may be useful for chess engines, for example, to implement null move heuristics.
///
/// Note that this kind of moves is a special one.
///
/// As stated above, it is well-formed but not semilegal. So, it is not accepted by safe functions, but is accepted
/// as a semilegal move by unsafe functions (such as [`make_move_unchecked()`]).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Move {
    kind: MoveKind,
    src: Coord,
    dst: Coord,
    side: Option<Color>,
}

/// Error indicating that move is invalid
#[derive(Debug, Clone, Error, Eq, PartialEq)]
pub enum ValidateError {
    /// Move is not semi-legal
    #[error("move is not semi-legal")]
    NotSemiLegal,
    /// Move is not legal
    #[error("move is not legal")]
    NotLegal,
}

/// Error creating move
#[derive(Debug, Clone, Error, Eq, PartialEq)]
pub enum CreateError {
    /// Move is not well-formed
    #[error("move is not well-formed")]
    NotWellFormed,
}

impl Move {
    /// Null move
    pub const NULL: Move = Move::null();

    const fn null() -> Self {
        Self {
            kind: MoveKind::Null,
            src: Coord::from_index(0),
            dst: Coord::from_index(0),
            side: None,
        }
    }

    /// Creates a castling move made by `color` with side `side`
    #[inline]
    pub fn from_castling(color: Color, side: CastlingSide) -> Move {
        let rank = geometry::castling_rank(color);
        let src = Coord::from_parts(File::E, rank);
        let dst = match side {
            CastlingSide::King => Coord::from_parts(File::G, rank),
            CastlingSide::Queen => Coord::from_parts(File::C, rank),
        };
        Move {
            kind: MoveKind::from(side),
            src,
            dst,
            side: Some(color),
        }
    }

    /// Creates a new non-null move from its raw parts
    ///
    /// # Safety
    ///
    /// If the created move is not well formed, it is undefined behavior to do with it something other
    /// than checking it for well-formedness via [`Move::is_well_formed()`] or examining its fields via
    /// getters.
    #[inline]
    pub const unsafe fn new_unchecked(kind: MoveKind, src: Coord, dst: Coord, side: Color) -> Move {
        Move {
            kind,
            src,
            dst,
            side: Some(side),
        }
    }

    /// Creates a move from the UCI string `s` if `b` is the positon preceding this move
    ///
    /// The returned move is **not** guaranteed to be semilegal.
    #[inline]
    pub fn from_uci(s: &str, b: &Board) -> Result<Move, uci::BasicParseError> {
        Ok(uci::Move::from_str(s)?.into_move(b)?)
    }

    /// Same as [`Move::from_uci()`], but the returned move is guaranteed to be semilegal
    pub fn from_uci_semilegal(s: &str, b: &Board) -> Result<Move, uci::ParseError> {
        let res = uci::Move::from_str(s)?.into_move(b)?;
        res.semi_validate(b)?;
        Ok(res)
    }

    /// Same as [`Move::from_uci()`], but the returned move is guaranteed to be legal
    pub fn from_uci_legal(s: &str, b: &Board) -> Result<Move, uci::ParseError> {
        let res = uci::Move::from_str(s)?.into_move(b)?;
        res.validate(b)?;
        Ok(res)
    }

    /// Creates a move from the SAN string `s` if `b` is the positon preceding this move
    ///
    /// The returned move is guaranteed to be **legal**.
    #[inline]
    pub fn from_san(s: &str, b: &Board) -> Result<Move, san::ParseError> {
        Ok(san::Move::from_str(s)?.into_move(b)?)
    }

    /// Returns `true` if the move is semilegal
    pub fn is_semilegal(&self, b: &Board) -> bool {
        match b.r.side {
            Color::White => do_is_move_semilegal::<generic::White>(b, *self),
            Color::Black => do_is_move_semilegal::<generic::Black>(b, *self),
        }
    }

    /// Returns `true` if the move is legal
    ///
    /// # Safety
    ///
    /// The move must be semilegal, otherwise the behavior is undefined.
    pub unsafe fn is_legal_unchecked(&self, b: &Board) -> bool {
        Checker::new(b, NilPrechecker).is_legal(*self)
    }

    /// Validates whether this move is semilegal from position `b`
    #[inline]
    pub fn semi_validate(&self, b: &Board) -> Result<(), ValidateError> {
        if !self.is_semilegal(b) {
            return Err(ValidateError::NotSemiLegal);
        }
        Ok(())
    }

    /// Validates whether this move is legal from position `b`
    #[inline]
    pub fn validate(&self, b: &Board) -> Result<(), ValidateError> {
        self.semi_validate(b)?;
        match unsafe { self.is_legal_unchecked(b) } {
            true => Ok(()),
            false => Err(ValidateError::NotLegal),
        }
    }

    /// Creates a new non-null move from its raw parts and validates it for well-formedness
    pub fn new(kind: MoveKind, src: Coord, dst: Coord, side: Color) -> Result<Move, CreateError> {
        let mv = Move {
            kind,
            src,
            dst,
            side: Some(side),
        };
        mv.is_well_formed()
            .then_some(mv)
            .ok_or(CreateError::NotWellFormed)
    }

    /// Returns `true` if the move is well-formed
    ///
    /// If the move is not well-formed, you can only call getters and this function on it, other uses
    /// are undefined behaviour.
    pub fn is_well_formed(&self) -> bool {
        // `side` can be `None` only if it's null move
        if self.side.is_none() && self.kind != MoveKind::Null {
            return false;
        }
        let side = self.side.unwrap_or(Color::White);

        match self.kind {
            MoveKind::Null => {
                if *self != Move::NULL {
                    return false;
                }
            }
            MoveKind::Simple => {
                // No need to perform additional checks
            }
            MoveKind::CastlingKingside => {
                let rank = geometry::castling_rank(side);
                if self.src != Coord::from_parts(File::E, rank)
                    || self.dst != Coord::from_parts(File::G, rank)
                {
                    return false;
                }
            }
            MoveKind::CastlingQueenside => {
                let rank = geometry::castling_rank(side);
                if self.src != Coord::from_parts(File::E, rank)
                    || self.dst != Coord::from_parts(File::C, rank)
                {
                    return false;
                }
            }
            MoveKind::PawnSimple => {
                if self.src.file().index().abs_diff(self.dst.file().index()) > 1
                    || matches!(self.src.rank(), Rank::R1 | Rank::R8)
                    || matches!(self.dst.rank(), Rank::R1 | Rank::R8)
                {
                    return false;
                }
                match side {
                    Color::White => {
                        if self.src.rank().index() != self.dst.rank().index() + 1 {
                            return false;
                        }
                    }
                    Color::Black => {
                        if self.src.rank().index() + 1 != self.dst.rank().index() {
                            return false;
                        }
                    }
                };
            }
            MoveKind::PawnDouble => {
                if self.src.file() != self.dst.file()
                    || self.src.rank() != geometry::double_move_src_rank(side)
                    || self.dst.rank() != geometry::double_move_dst_rank(side)
                {
                    return false;
                }
            }
            MoveKind::Enpassant => {
                if self.src.rank() != geometry::enpassant_src_rank(side)
                    || self.dst.rank() != geometry::enpassant_dst_rank(side)
                    || self.src.file().index().abs_diff(self.dst.file().index()) != 1
                {
                    return false;
                }
            }
            MoveKind::PromoteKnight
            | MoveKind::PromoteBishop
            | MoveKind::PromoteRook
            | MoveKind::PromoteQueen => {
                if self.src.rank() != geometry::promote_src_rank(side)
                    || self.dst.rank() != geometry::promote_dst_rank(side)
                    || self.src.file().index().abs_diff(self.dst.file().index()) > 1
                {
                    return false;
                }
            }
        };

        true
    }

    /// Returns the move kind
    #[inline]
    pub const fn kind(&self) -> MoveKind {
        self.kind
    }

    /// Returns the move source square
    ///
    /// For null move, this function returns a square with index 0.
    #[inline]
    pub const fn src(&self) -> Coord {
        self.src
    }

    /// Returns the move destination square
    ///
    /// For null move, this function returns a square with index 0.
    #[inline]
    pub const fn dst(&self) -> Coord {
        self.dst
    }

    /// Returns the side which makes this move
    #[inline]
    pub const fn side(&self) -> Option<Color> {
        self.side
    }

    /// Converts this move into a parsed UCI representation
    #[inline]
    pub fn uci(&self) -> uci::Move {
        (*self).into()
    }

    /// Converts this move into a parsed SAN representation in given position `b`
    ///
    /// This function returns an error if the move is not legal in the given position.
    #[inline]
    pub fn san(&self, b: &Board) -> Result<san::Move, ValidateError> {
        san::Move::from_move(*self, b)
    }

    /// Returns the wrapper which helps to format the move with the given style `style`
    ///
    /// The resulting wrapper implements [`fmt::Display`], so can be used with
    /// `write!()`, `println!()`, or `ToString::to_string`.
    ///
    /// # Example
    ///
    /// ```
    /// # use owlchess::{Move, Board, MoveKind, File, Rank, Coord, Color, moves::Style};
    /// #
    /// let b = Board::initial();
    /// let g1 = Coord::from_parts(File::G, Rank::R1);
    /// let f3 = Coord::from_parts(File::F, Rank::R3);
    /// let mv = Move::new(MoveKind::Simple, g1, f3, Color::White).unwrap();
    /// assert_eq!(mv.styled(&b, Style::Uci).unwrap().to_string(), "g1f3".to_string());
    /// assert_eq!(mv.styled(&b, Style::San).unwrap().to_string(), "Nf3".to_string());
    /// assert_eq!(mv.styled(&b, Style::SanUtf8).unwrap().to_string(), "♘f3".to_string());
    /// ```
    pub fn styled(&self, b: &Board, style: Style) -> Result<StyledMove, ValidateError> {
        match style {
            Style::Uci => Ok(StyledMove(Styled::Uci((*self).into()))),
            Style::San => Ok(StyledMove(Styled::San(
                san::Move::from_move(*self, b)?,
                san::Style::Algebraic,
            ))),
            Style::SanUtf8 => Ok(StyledMove(Styled::San(
                san::Move::from_move(*self, b)?,
                san::Style::Utf8,
            ))),
        }
    }
}

impl Default for Move {
    #[inline]
    fn default() -> Self {
        Move::NULL
    }
}

impl fmt::Display for Move {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        self.uci().fmt(f)
    }
}

enum Styled {
    Uci(uci::Move),
    San(san::Move, san::Style),
}

/// Wrapper to format the move with the given style
///
/// See [`Move::styled()`] doc for details.
pub struct StyledMove(Styled);

impl fmt::Display for StyledMove {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self.0 {
            Styled::Uci(mv) => mv.fmt(f),
            Styled::San(mv, sty) => mv.styled(sty).fmt(f),
        }
    }
}

/// Metadata necessary to undo the applied move
#[derive(Debug, Copy, Clone)]
pub struct RawUndo {
    hash: u64,
    dst_cell: Cell,
    castling: CastlingRights,
    ep_source: Option<Coord>,
    move_counter: u16,
}

trait MakeMoveImpl {
    const COLOR: Color;
}

fn update_castling(b: &mut Board, change: Bitboard) {
    if (change & castling::ALL_SRCS).is_empty() {
        return;
    }

    let mut castling = b.r.castling;
    for (c, s) in [
        (Color::White, CastlingSide::Queen),
        (Color::White, CastlingSide::King),
        (Color::Black, CastlingSide::Queen),
        (Color::Black, CastlingSide::King),
    ] {
        if (change & castling::srcs(c, s)).is_nonempty() {
            castling.unset(c, s);
        }
    }

    if castling != b.r.castling {
        b.hash ^= zobrist::castling(b.r.castling);
        b.r.castling = castling;
        b.hash ^= zobrist::castling(b.r.castling);
    }
}

#[inline]
fn do_make_pawn_double<C: generic::Color>(b: &mut Board, mv: Move, change: Bitboard, inv: bool) {
    let pawn = Cell::from_parts(C::COLOR, Piece::Pawn);
    if inv {
        b.r.put(mv.src, pawn);
        b.r.put(mv.dst, Cell::EMPTY);
    } else {
        b.r.put(mv.src, Cell::EMPTY);
        b.r.put(mv.dst, pawn);
        b.hash ^= zobrist::pieces(pawn, mv.src) ^ zobrist::pieces(pawn, mv.dst);
    }
    *b.color_mut(C::COLOR) ^= change;
    *b.piece_mut(pawn) ^= change;
    if !inv {
        b.r.ep_source = Some(mv.dst);
        b.hash ^= zobrist::enpassant(mv.dst);
    }
}

#[inline]
fn do_make_enpassant<C: generic::Color>(b: &mut Board, mv: Move, change: Bitboard, inv: bool) {
    let taken_pos = unsafe {
        mv.dst
            .add_unchecked(-geometry::pawn_forward_delta(C::COLOR))
    };
    let taken = Bitboard::from_coord(taken_pos);
    let our_pawn = Cell::from_parts(C::COLOR, Piece::Pawn);
    let their_pawn = Cell::from_parts(C::COLOR.inv(), Piece::Pawn);
    if inv {
        b.r.put(mv.src, our_pawn);
        b.r.put(mv.dst, Cell::EMPTY);
        b.r.put(taken_pos, their_pawn);
    } else {
        b.r.put(mv.src, Cell::EMPTY);
        b.r.put(mv.dst, our_pawn);
        b.r.put(taken_pos, Cell::EMPTY);
        b.hash ^= zobrist::pieces(our_pawn, mv.src)
            ^ zobrist::pieces(our_pawn, mv.dst)
            ^ zobrist::pieces(their_pawn, taken_pos);
    }
    *b.color_mut(C::COLOR) ^= change;
    *b.piece_mut(our_pawn) ^= change;
    *b.color_mut(C::COLOR.inv()) ^= taken;
    *b.piece_mut(their_pawn) ^= taken;
}

#[inline]
fn do_make_castling_kingside<C: generic::Color>(b: &mut Board, inv: bool) {
    let king = Cell::from_parts(C::COLOR, Piece::King);
    let rook = Cell::from_parts(C::COLOR, Piece::Rook);
    let rank = geometry::castling_rank(C::COLOR);
    if inv {
        b.r.put2(File::E, rank, king);
        b.r.put2(File::F, rank, Cell::EMPTY);
        b.r.put2(File::G, rank, Cell::EMPTY);
        b.r.put2(File::H, rank, rook);
    } else {
        b.r.put2(File::E, rank, Cell::EMPTY);
        b.r.put2(File::F, rank, rook);
        b.r.put2(File::G, rank, king);
        b.r.put2(File::H, rank, Cell::EMPTY);
        b.hash ^= zobrist::castling_delta(C::COLOR, CastlingSide::King);
    }
    *b.color_mut(C::COLOR) ^= Bitboard::from_raw(0xf0 << C::CASTLING_OFFSET);
    *b.piece_mut(rook) ^= Bitboard::from_raw(0xa0 << C::CASTLING_OFFSET);
    *b.piece_mut(king) ^= Bitboard::from_raw(0x50 << C::CASTLING_OFFSET);
    if !inv {
        b.hash ^= zobrist::castling(b.r.castling);
        b.r.castling.unset_color(C::COLOR);
        b.hash ^= zobrist::castling(b.r.castling);
    }
}

#[inline]
fn do_make_castling_queenside<C: generic::Color>(b: &mut Board, inv: bool) {
    let king = Cell::from_parts(C::COLOR, Piece::King);
    let rook = Cell::from_parts(C::COLOR, Piece::Rook);
    let rank = geometry::castling_rank(C::COLOR);
    if inv {
        b.r.put2(File::A, rank, rook);
        b.r.put2(File::C, rank, Cell::EMPTY);
        b.r.put2(File::D, rank, Cell::EMPTY);
        b.r.put2(File::E, rank, king);
    } else {
        b.r.put2(File::A, rank, Cell::EMPTY);
        b.r.put2(File::C, rank, king);
        b.r.put2(File::D, rank, rook);
        b.r.put2(File::E, rank, Cell::EMPTY);
        b.hash ^= zobrist::castling_delta(C::COLOR, CastlingSide::Queen);
    }
    *b.color_mut(C::COLOR) ^= Bitboard::from_raw(0x1d << C::CASTLING_OFFSET);
    *b.piece_mut(rook) ^= Bitboard::from_raw(0x09 << C::CASTLING_OFFSET);
    *b.piece_mut(king) ^= Bitboard::from_raw(0x14 << C::CASTLING_OFFSET);
    if !inv {
        b.hash ^= zobrist::castling(b.r.castling);
        b.r.castling.unset_color(C::COLOR);
        b.hash ^= zobrist::castling(b.r.castling);
    }
}

fn do_make_move<C: generic::Color>(b: &mut Board, mv: Move) -> RawUndo {
    let src_cell = b.get(mv.src);
    let dst_cell = b.get(mv.dst);
    let undo = RawUndo {
        hash: b.hash,
        dst_cell,
        castling: b.r.castling,
        ep_source: b.r.ep_source,
        move_counter: b.r.move_counter,
    };
    let src = Bitboard::from_coord(mv.src);
    let dst = Bitboard::from_coord(mv.dst);
    let change = src | dst;
    if let Some(p) = b.r.ep_source {
        b.hash ^= zobrist::enpassant(p);
        b.r.ep_source = None;
    }
    match mv.kind {
        MoveKind::Simple | MoveKind::PawnSimple => {
            b.r.put(mv.src, Cell::EMPTY);
            b.r.put(mv.dst, src_cell);
            b.hash ^= zobrist::pieces(src_cell, mv.src)
                ^ zobrist::pieces(src_cell, mv.dst)
                ^ zobrist::pieces(dst_cell, mv.dst);
            *b.color_mut(C::COLOR) ^= change;
            *b.piece_mut(src_cell) ^= change;
            *b.color_mut(C::COLOR.inv()) &= !dst;
            *b.piece_mut(dst_cell) &= !dst;
            if mv.kind == MoveKind::Simple {
                update_castling(b, change);
            }
        }
        MoveKind::PawnDouble => {
            do_make_pawn_double::<C>(b, mv, change, false);
        }
        MoveKind::PromoteKnight
        | MoveKind::PromoteBishop
        | MoveKind::PromoteRook
        | MoveKind::PromoteQueen => {
            let promote = Cell::from_parts(C::COLOR, mv.kind.promote().unwrap());
            let pawn = Cell::from_parts(C::COLOR, Piece::Pawn);
            b.r.put(mv.src, Cell::EMPTY);
            b.r.put(mv.dst, promote);
            b.hash ^= zobrist::pieces(src_cell, mv.src)
                ^ zobrist::pieces(promote, mv.dst)
                ^ zobrist::pieces(dst_cell, mv.dst);
            *b.color_mut(C::COLOR) ^= change;
            *b.piece_mut(pawn) ^= src;
            *b.piece_mut(promote) ^= dst;
            *b.color_mut(C::COLOR.inv()) &= !dst;
            *b.piece_mut(dst_cell) &= !dst;
            update_castling(b, change);
        }
        MoveKind::CastlingKingside => {
            do_make_castling_kingside::<C>(b, false);
        }
        MoveKind::CastlingQueenside => {
            do_make_castling_queenside::<C>(b, false);
        }
        MoveKind::Null => {
            // Do nothing.
        }
        MoveKind::Enpassant => {
            do_make_enpassant::<C>(b, mv, change, false);
        }
    }

    if dst_cell != Cell::EMPTY || src_cell == Cell::from_parts(C::COLOR, Piece::Pawn) {
        b.r.move_counter = 0;
    } else {
        b.r.move_counter += 1;
    }
    b.r.side = C::COLOR.inv();
    b.hash ^= zobrist::MOVE_SIDE;
    if C::COLOR == Color::Black {
        b.r.move_number += 1;
    }
    b.all = b.white | b.black;

    undo
}

fn do_unmake_move<C: generic::Color>(b: &mut Board, mv: Move, u: RawUndo) {
    let src = Bitboard::from_coord(mv.src);
    let dst = Bitboard::from_coord(mv.dst);
    let change = src | dst;
    let src_cell = b.get(mv.dst);
    let dst_cell = u.dst_cell;

    match mv.kind {
        MoveKind::Simple | MoveKind::PawnSimple => {
            b.r.put(mv.src, src_cell);
            b.r.put(mv.dst, dst_cell);
            *b.color_mut(C::COLOR) ^= change;
            *b.piece_mut(src_cell) ^= change;
            if dst_cell.is_occupied() {
                *b.color_mut(C::COLOR.inv()) |= dst;
                *b.piece_mut(dst_cell) |= dst;
            }
        }
        MoveKind::PawnDouble => {
            do_make_pawn_double::<C>(b, mv, change, true);
        }
        MoveKind::PromoteKnight
        | MoveKind::PromoteBishop
        | MoveKind::PromoteRook
        | MoveKind::PromoteQueen => {
            let pawn = Cell::from_parts(C::COLOR, Piece::Pawn);
            b.r.put(mv.src, pawn);
            b.r.put(mv.dst, dst_cell);
            *b.color_mut(C::COLOR) ^= change;
            *b.piece_mut(pawn) ^= src;
            *b.piece_mut(src_cell) ^= dst;
            if dst_cell.is_occupied() {
                *b.color_mut(C::COLOR.inv()) |= dst;
                *b.piece_mut(dst_cell) |= dst;
            }
        }
        MoveKind::CastlingKingside => {
            do_make_castling_kingside::<C>(b, true);
        }
        MoveKind::CastlingQueenside => {
            do_make_castling_queenside::<C>(b, true);
        }
        MoveKind::Null => {
            // Do nothing
        }
        MoveKind::Enpassant => {
            do_make_enpassant::<C>(b, mv, change, true);
        }
    }

    b.hash = u.hash;
    b.r.castling = u.castling;
    b.r.ep_source = u.ep_source;
    b.r.move_counter = u.move_counter;
    b.r.side = C::COLOR;
    if C::COLOR == Color::Black {
        b.r.move_number -= 1;
    }
    b.all = b.white | b.black;
}

/// Makes the move `mv` on the board `b`
///
/// To allow unmaking the move, a `RawUndo` instance is returned. See [`unmake_move_unchecked()`] for the
/// details on how to unmake a move.
///
/// # Safety
///
/// The move must be either semilegal or null, otherwise the behavior is undefined.
///
/// If the king is under attack after the move (i.e. the board becomes invalid), it must be immediately
/// rolled back via [`unmake_move_unchecked()`]. Doing anything other with the board before that,
/// except unmaking the move or calling [`Board::is_opponent_king_attacked()`] is undefined behavior.
///
/// See docs for [`Board`] for more details.
pub unsafe fn make_move_unchecked(b: &mut Board, mv: Move) -> RawUndo {
    match b.r.side {
        Color::White => do_make_move::<generic::White>(b, mv),
        Color::Black => do_make_move::<generic::Black>(b, mv),
    }
}

/// Unmakes the move `mv` on the board `b`
///
/// # Safety
///
/// If there exists a valid position `b_old` such that `make_move_unchecked(&mut b_old, mv)`
/// doesn't result in undefined behavior, returns a value equal to `u`, and `b_old == b` holds
/// after such operation, then this `b_old` is returned. Otherwise, the behavior is undefined.
///
/// In simpler words, you may invoke this function only from the position occured after the
/// corresponing call to [`make_move_unchecked()`] or [`Make::make_raw()`](super::Make::make_raw).
///
/// Note that `b` can be an invalid position, so you can roll back an illegal move. See docs for
/// [`Board`] or [`make_move_unchecked()`] for more details.
pub unsafe fn unmake_move_unchecked(b: &mut Board, mv: Move, u: RawUndo) {
    match b.r.side {
        Color::White => do_unmake_move::<generic::Black>(b, mv, u),
        Color::Black => do_unmake_move::<generic::White>(b, mv, u),
    }
}

fn is_bishop_semilegal(src: Coord, dst: Coord, all: Bitboard) -> bool {
    between::is_bishop_valid(src, dst) && (between::bishop_strict(src, dst) & all).is_empty()
}

fn is_rook_semilegal(src: Coord, dst: Coord, all: Bitboard) -> bool {
    between::is_rook_valid(src, dst) && (between::rook_strict(src, dst) & all).is_empty()
}

fn do_is_move_semilegal<C: generic::Color>(b: &Board, mv: Move) -> bool {
    if mv.side == Some(C::COLOR.inv()) {
        return false;
    }
    let src_cell = b.get(mv.src);
    let pawn = Cell::from_parts(C::COLOR, Piece::Pawn);
    match mv.kind {
        MoveKind::Simple => {
            let dst = Bitboard::from_coord(mv.dst);
            if src_cell.color() != Some(C::COLOR) || (dst & b.color(C::COLOR)).is_nonempty() {
                return false;
            }
            match src_cell.piece() {
                Some(Piece::Bishop) => is_bishop_semilegal(mv.src, mv.dst, b.all),
                Some(Piece::Rook) => is_rook_semilegal(mv.src, mv.dst, b.all),
                Some(Piece::Queen) => {
                    is_bishop_semilegal(mv.src, mv.dst, b.all)
                        || is_rook_semilegal(mv.src, mv.dst, b.all)
                }
                Some(Piece::Knight) => (attack::knight(mv.src) & dst).is_nonempty(),
                Some(Piece::King) => (attack::king(mv.src) & dst).is_nonempty(),
                Some(Piece::Pawn) => false,
                _ => unreachable!(),
            }
        }
        MoveKind::PawnSimple
        | MoveKind::PromoteKnight
        | MoveKind::PromoteBishop
        | MoveKind::PromoteRook
        | MoveKind::PromoteQueen => {
            if src_cell != pawn {
                return false;
            }
            let dst_cell = b.r.get(mv.dst);
            if let Some(c) = dst_cell.color() {
                c == C::COLOR.inv() && mv.dst.file() != mv.src.file()
            } else {
                mv.dst.file() == mv.src.file()
            }
        }
        MoveKind::PawnDouble => {
            let must_empty = match C::COLOR {
                Color::White => Bitboard::from_raw(0x0101 << (mv.src.index() - 16)),
                Color::Black => Bitboard::from_raw(0x010100 << mv.src.index()),
            };
            src_cell == pawn && (b.all & must_empty).is_empty()
        }
        MoveKind::CastlingKingside => {
            b.r.castling.has(C::COLOR, CastlingSide::King)
                && (b.all & castling::pass(C::COLOR, CastlingSide::King)).is_empty()
                && !movegen::do_is_cell_attacked::<C::Inv>(b, mv.src)
                && !movegen::do_is_cell_attacked::<C::Inv>(b, unsafe { mv.src.add_unchecked(1) })
        }
        MoveKind::CastlingQueenside => {
            b.r.castling.has(C::COLOR, CastlingSide::Queen)
                && (b.all & castling::pass(C::COLOR, CastlingSide::Queen)).is_empty()
                && !movegen::do_is_cell_attacked::<C::Inv>(b, mv.src)
                && !movegen::do_is_cell_attacked::<C::Inv>(b, unsafe { mv.src.add_unchecked(-1) })
        }
        MoveKind::Enpassant => {
            if src_cell != pawn {
                return false;
            }
            if let Some(p) = b.r.ep_source {
                unsafe {
                    return (p == mv.src.add_unchecked(1) || p == mv.src.add_unchecked(-1))
                        && mv.dst == p.add_unchecked(geometry::pawn_forward_delta(C::COLOR));
                }
            }
            false
        }
        MoveKind::Null => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::moves::make::{self, Make};
    use std::mem;

    #[test]
    fn test_size() {
        assert_eq!(mem::size_of::<Move>(), 4);
    }

    #[test]
    fn test_style() {
        let b = Board::initial();
        let mv = Move::from_uci("g1f3", &b).unwrap();
        assert_eq!(mv.styled(&b, Style::Uci).unwrap().to_string(), "g1f3");
        assert_eq!(mv.styled(&b, Style::San).unwrap().to_string(), "Nf3");
        assert_eq!(mv.styled(&b, Style::SanUtf8).unwrap().to_string(), "♘f3");
        assert_eq!(mv.uci().to_string(), "g1f3");
        assert_eq!(mv.san(&b).unwrap().to_string(), "Nf3");
        assert_eq!(
            mv.san(&b).unwrap().styled(san::Style::Utf8).to_string(),
            "♘f3"
        );
    }

    #[test]
    fn test_simple() {
        let mut b = Board::initial();
        for (mv_str, fen_str) in [
            (
                "e2e4",
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            ),
            (
                "b8c6",
                "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
            ),
            (
                "g1f3",
                "r1bqkbnr/pppppppp/2n5/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 2",
            ),
            (
                "e7e5",
                "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq e6 0 3",
            ),
            (
                "f1b5",
                "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 3",
            ),
            (
                "g8f6",
                "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4",
            ),
            (
                "e1g1",
                "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 3 4",
            ),
            (
                "f6e4",
                "r1bqkb1r/pppp1ppp/2n5/1B2p3/4n3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 0 5",
            ),
        ] {
            let m = Move::from_uci_semilegal(mv_str, &b).unwrap();
            b = b.make_move(m).unwrap();
            assert_eq!(b.as_fen(), fen_str);
            assert_eq!(b.raw().try_into(), Ok(b.clone()));
        }
    }

    #[test]
    fn test_promote() {
        let mut b = Board::from_fen("1b1b1K2/2P5/8/8/7k/8/8/8 w - - 0 1").unwrap();
        let b_copy = b.clone();

        for (mv_str, fen_str) in [
            ("c7c8q", "1bQb1K2/8/8/8/7k/8/8/8 b - - 0 1"),
            ("c7b8n", "1N1b1K2/8/8/8/7k/8/8/8 b - - 0 1"),
            ("c7d8r", "1b1R1K2/8/8/8/7k/8/8/8 b - - 0 1"),
        ] {
            let (m, u) = make::Uci(mv_str).make_raw(&mut b).unwrap();
            assert_eq!(b.as_fen(), fen_str);
            assert_eq!(b.raw().try_into(), Ok(b.clone()));
            unsafe { unmake_move_unchecked(&mut b, m, u) };
            assert_eq!(b, b_copy);
        }
    }

    #[test]
    fn test_undo() {
        let mut b =
            Board::from_fen("r1bqk2r/ppp2ppp/2np1n2/1Bb1p3/4P3/2PP1N2/PP3PPP/RNBQK2R w KQkq - 0 6")
                .unwrap();
        let b_copy = b.clone();

        for (mv_str, fen_str) in [
            (
                "e1g1",
                "r1bqk2r/ppp2ppp/2np1n2/1Bb1p3/4P3/2PP1N2/PP3PPP/RNBQ1RK1 b kq - 1 6",
            ),
            (
                "f3e5",
                "r1bqk2r/ppp2ppp/2np1n2/1Bb1N3/4P3/2PP4/PP3PPP/RNBQK2R b KQkq - 0 6",
            ),
            (
                "b2b4",
                "r1bqk2r/ppp2ppp/2np1n2/1Bb1p3/1P2P3/2PP1N2/P4PPP/RNBQK2R b KQkq b3 0 6",
            ),
            (
                "c3c4",
                "r1bqk2r/ppp2ppp/2np1n2/1Bb1p3/2P1P3/3P1N2/PP3PPP/RNBQK2R b KQkq - 0 6",
            ),
        ] {
            let (m, u) = make::Uci(mv_str).make_raw(&mut b).unwrap();
            assert_eq!(b.as_fen(), fen_str);
            assert_eq!(b.raw().try_into(), Ok(b.clone()));
            unsafe { unmake_move_unchecked(&mut b, m, u) };
            assert_eq!(b, b_copy);
        }
    }

    #[test]
    fn test_pawns() {
        let mut b = Board::from_fen("3K4/3p4/8/3PpP2/8/5p2/6P1/2k5 w - e6 0 1").unwrap();
        let b_copy = b.clone();

        for (mv_str, fen_str) in [
            ("g2g3", "3K4/3p4/8/3PpP2/8/5pP1/8/2k5 b - - 0 1"),
            ("g2g4", "3K4/3p4/8/3PpP2/6P1/5p2/8/2k5 b - g3 0 1"),
            ("g2f3", "3K4/3p4/8/3PpP2/8/5P2/8/2k5 b - - 0 1"),
            ("d5e6", "3K4/3p4/4P3/5P2/8/5p2/6P1/2k5 b - - 0 1"),
            ("f5e6", "3K4/3p4/4P3/3P4/8/5p2/6P1/2k5 b - - 0 1"),
        ] {
            let (m, u) = make::Uci(mv_str).make_raw(&mut b).unwrap();
            assert_eq!(b.as_fen(), fen_str);
            assert_eq!(b.raw().try_into(), Ok(b.clone()));
            unsafe { unmake_move_unchecked(&mut b, m, u) };
            assert_eq!(b, b_copy);
        }
    }

    #[test]
    fn test_legal() {
        let b =
            Board::from_fen("r1bqk2r/ppp2ppp/2np1n2/1Bb1p3/4P3/2PP1N2/PP3PPP/RNBQK2R w KQkq - 0 6")
                .unwrap();

        let m = Move::from_uci("e1c1", &b).unwrap();
        assert!(!m.is_semilegal(&b));
        assert_eq!(m.semi_validate(&b), Err(ValidateError::NotSemiLegal));

        let m = Move::from_uci("b5e8", &b).unwrap();
        assert!(!m.is_semilegal(&b));
        assert_eq!(m.semi_validate(&b), Err(ValidateError::NotSemiLegal));

        let m = Move::from_uci("a3a4", &b).unwrap();
        assert!(!m.is_semilegal(&b));
        assert_eq!(m.semi_validate(&b), Err(ValidateError::NotSemiLegal));

        let m = Move::from_uci("e1d1", &b).unwrap();
        assert!(!m.is_semilegal(&b));
        assert_eq!(m.semi_validate(&b), Err(ValidateError::NotSemiLegal));

        assert_eq!(
            Move::from_uci("c3c5", &b),
            Err(uci::BasicParseError::Create(CreateError::NotWellFormed))
        );
    }
}
