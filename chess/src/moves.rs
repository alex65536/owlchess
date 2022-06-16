use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::types::{
    CastlingRights, CastlingSide, Cell, Color, Coord, CoordParseError, File, Piece, Rank,
};
use crate::{attack, castling, generic, geometry, movegen, zobrist};

use std::fmt;
use std::str::FromStr;

use thiserror::Error;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MoveKind {
    Null = 0,
    Simple = 1,
    CastlingKingside = 2,
    CastlingQueenside = 3,
    PawnSimple = 4,
    PawnDouble = 5,
    Enpassant = 6,
    PromoteKnight = 7,
    PromoteBishop = 8,
    PromoteRook = 9,
    PromoteQueen = 10,
}

impl MoveKind {
    pub const fn castling_side(&self) -> Option<CastlingSide> {
        match *self {
            Self::CastlingKingside => Some(CastlingSide::King),
            Self::CastlingQueenside => Some(CastlingSide::Queen),
            _ => None,
        }
    }

    pub const fn promote_to(&self) -> Option<Piece> {
        match *self {
            Self::PromoteKnight => Some(Piece::Knight),
            Self::PromoteBishop => Some(Piece::Bishop),
            Self::PromoteRook => Some(Piece::Rook),
            Self::PromoteQueen => Some(Piece::Queen),
            _ => None,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Move {
    kind: MoveKind,
    src: Coord,
    dst: Coord,
    side: Option<Color>,
}

#[derive(Debug, Copy, Clone, Error, Eq, PartialEq)]
pub enum ValidateError {
    #[error("move is not sane")]
    NotSane,
    #[error("move is not semi-legal")]
    NotSemiLegal,
    #[error("move is not legal")]
    NotLegal,
}

#[derive(Debug, Copy, Clone, Error, Eq, PartialEq)]
pub enum CreateError {
    #[error("move is not well-formed")]
    NotWellFormed,
}

#[derive(Debug, Copy, Clone, Error, Eq, PartialEq)]
pub enum RawParseError {
    #[error("bad string length")]
    BadLength,
    #[error("bad source: {0}")]
    BadSrc(CoordParseError),
    #[error("bad destination: {0}")]
    BadDst(CoordParseError),
    #[error("bad promote char {0:?}")]
    BadPromote(char),
}

#[derive(Debug, Copy, Clone, Error, Eq, PartialEq)]
pub enum BasicParseError {
    #[error("cannot parse move: {0}")]
    Parse(#[from] RawParseError),
    #[error("cannot create move: {0}")]
    Create(#[from] CreateError),
}

#[derive(Debug, Copy, Clone, Error, Eq, PartialEq)]
pub enum ParseError {
    #[error("cannot parse move: {0}")]
    Parse(#[from] RawParseError),
    #[error("cannot create move: {0}")]
    Create(#[from] CreateError),
    #[error("invalid move: {0}")]
    Validate(#[from] ValidateError),
}

impl Move {
    pub const NULL: Move = Move {
        kind: MoveKind::Null,
        src: Coord::from_index(0),
        dst: Coord::from_index(0),
        side: None,
    };

    pub const unsafe fn new_unchecked(kind: MoveKind, src: Coord, dst: Coord, side: Color) -> Move {
        Move {
            kind,
            src,
            dst,
            side: Some(side),
        }
    }

    pub fn new(kind: MoveKind, src: Coord, dst: Coord, side: Color) -> Move {
        Self::try_new(kind, src, dst, side).expect("move is not well-formed")
    }

    pub fn from_str(s: &str, b: &Board) -> Result<Move, BasicParseError> {
        Ok(ParsedMove::from_str(s)?.into_move(b)?)
    }

    pub fn from_str_semilegal(s: &str, b: &Board) -> Result<Move, ParseError> {
        let res = ParsedMove::from_str(s)?.into_move(b)?;
        res.semi_validate(b)?;
        Ok(res)
    }

    pub fn from_str_legal(s: &str, b: &Board) -> Result<Move, ParseError> {
        let res = ParsedMove::from_str(s)?.into_move(b)?;
        res.validate(b)?;
        Ok(res)
    }

    pub fn semi_validate(&self, b: &Board) -> Result<(), ValidateError> {
        semi_validate(b, *self)
    }

    pub fn validate(&self, b: &Board) -> Result<(), ValidateError> {
        validate(b, *self)
    }

    pub fn try_new(
        kind: MoveKind,
        src: Coord,
        dst: Coord,
        side: Color,
    ) -> Result<Move, CreateError> {
        let mv = Move {
            kind,
            src,
            dst,
            side: Some(side),
        };
        mv.is_well_formed()
            .then(|| mv)
            .ok_or(CreateError::NotWellFormed)
    }

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

    pub const fn kind(&self) -> MoveKind {
        self.kind
    }

    pub const fn src(&self) -> Coord {
        self.src
    }

    pub const fn dst(&self) -> Coord {
        self.dst
    }

    pub const fn side(&self) -> Option<Color> {
        self.side
    }
}

impl Default for Move {
    fn default() -> Self {
        Move::NULL
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        if self.kind == MoveKind::Null {
            return write!(f, "0000");
        }
        write!(f, "{}{}", self.src, self.dst)?;
        match self.kind {
            MoveKind::PromoteKnight => write!(f, "n")?,
            MoveKind::PromoteBishop => write!(f, "b")?,
            MoveKind::PromoteRook => write!(f, "r")?,
            MoveKind::PromoteQueen => write!(f, "q")?,
            _ => {}
        };
        Ok(())
    }
}

pub struct ParsedMove {
    src: Coord,
    dst: Coord,
    kind: Option<MoveKind>,
}

impl ParsedMove {
    fn do_into_move<C: generic::Color>(&self, b: &Board) -> Result<Move, CreateError> {
        if self.kind == Some(MoveKind::Null) {
            return Ok(Move::NULL);
        }

        let kind = self.kind.unwrap_or_else(|| {
            // Pawn moves
            if b.get(self.src) == Cell::from_parts(C::COLOR, Piece::Pawn) {
                if self.src.rank() == geometry::double_move_src_rank(C::COLOR)
                    && self.dst.rank() == geometry::double_move_dst_rank(C::COLOR)
                {
                    return MoveKind::PawnDouble;
                }
                if self.src.file() != self.dst.file() && b.get(self.dst).is_empty() {
                    return MoveKind::Enpassant;
                }
                return MoveKind::PawnSimple;
            }

            // Castling
            if b.get(self.src) == Cell::from_parts(C::COLOR, Piece::King) {
                let rank = geometry::castling_rank(C::COLOR);
                if self.src == Coord::from_parts(File::E, rank) {
                    if self.dst == Coord::from_parts(File::G, rank) {
                        return MoveKind::CastlingKingside;
                    }
                    if self.dst == Coord::from_parts(File::C, rank) {
                        return MoveKind::CastlingQueenside;
                    }
                }
            }

            MoveKind::Simple
        });

        Move::try_new(kind, self.src, self.dst, C::COLOR)
    }

    pub fn into_move(self, b: &Board) -> Result<Move, CreateError> {
        match b.r.side {
            Color::White => self.do_into_move::<generic::White>(b),
            Color::Black => self.do_into_move::<generic::Black>(b),
        }
    }
}

impl FromStr for ParsedMove {
    type Err = RawParseError;

    fn from_str(s: &str) -> Result<ParsedMove, Self::Err> {
        if s == "0000" {
            return Ok(ParsedMove {
                src: Coord::from_index(0),
                dst: Coord::from_index(0),
                kind: Some(MoveKind::Null),
            });
        }
        if !matches!(s.len(), 4 | 5) {
            return Err(RawParseError::BadLength);
        }
        let src = Coord::from_str(&s[0..2]).map_err(RawParseError::BadSrc)?;
        let dst = Coord::from_str(&s[2..4]).map_err(RawParseError::BadDst)?;
        let kind = if s.len() == 5 {
            Some(match s.as_bytes()[4] {
                b'n' => MoveKind::PromoteKnight,
                b'b' => MoveKind::PromoteBishop,
                b'r' => MoveKind::PromoteRook,
                b'q' => MoveKind::PromoteQueen,
                b => return Err(RawParseError::BadPromote(b as char)),
            })
        } else {
            None
        };
        Ok(ParsedMove { src, dst, kind })
    }
}

#[derive(Debug, Copy, Clone)]
pub struct RawUndo {
    hash: u64,
    dst_cell: Cell,
    castling: CastlingRights,
    enpassant: Option<Coord>,
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
        b.r.enpassant = Some(mv.dst);
        b.hash ^= zobrist::enpassant(mv.dst);
    }
}

unsafe fn enpassant_pawn_pos_unchecked(c: Color, dst: Coord) -> Coord {
    match c {
        Color::White => dst.add_unchecked(8),
        Color::Black => dst.add_unchecked(-8),
    }
}

#[inline]
fn do_make_enpassant<C: generic::Color>(b: &mut Board, mv: Move, change: Bitboard, inv: bool) {
    let taken_pos = unsafe { enpassant_pawn_pos_unchecked(C::COLOR, mv.dst) };
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
        enpassant: b.r.enpassant,
        move_counter: b.r.move_counter,
    };
    let src = Bitboard::from_coord(mv.src);
    let dst = Bitboard::from_coord(mv.dst);
    let change = src | dst;
    if let Some(p) = b.r.enpassant {
        b.hash ^= zobrist::enpassant(p);
        b.r.enpassant = None;
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
            let promote = Cell::from_parts(C::COLOR, mv.kind.promote_to().unwrap());
            let pawn = Cell::from_parts(C::COLOR, Piece::Pawn);
            b.r.put(mv.src, promote);
            b.r.put(mv.dst, Cell::EMPTY);
            b.hash ^= zobrist::pieces(src_cell, mv.src)
                ^ zobrist::pieces(promote, mv.dst)
                ^ zobrist::pieces(dst_cell, mv.dst);
            *b.color_mut(C::COLOR) ^= change;
            *b.piece_mut(pawn) ^= src;
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
    b.r.enpassant = u.enpassant;
    b.r.move_counter = u.move_counter;
    b.r.side = C::COLOR;
    if C::COLOR == Color::Black {
        b.r.move_number -= 1;
    }
    b.all = b.white | b.black;
}

pub unsafe fn make_move_unchecked(b: &mut Board, mv: Move) -> RawUndo {
    match b.r.side {
        Color::White => do_make_move::<generic::White>(b, mv),
        Color::Black => do_make_move::<generic::Black>(b, mv),
    }
}

pub unsafe fn unmake_move_unchecked(b: &mut Board, mv: Move, u: RawUndo) {
    match b.r.side {
        Color::White => do_unmake_move::<generic::Black>(b, mv, u),
        Color::Black => do_unmake_move::<generic::White>(b, mv, u),
    }
}

pub unsafe fn make_semilegal_move_unchecked(
    b: &mut Board,
    mv: Move,
) -> Result<RawUndo, ValidateError> {
    let u = make_move_unchecked(b, mv);
    if b.is_opponent_king_attacked() {
        unmake_move_unchecked(b, mv, u);
        return Err(ValidateError::NotLegal);
    }
    Ok(u)
}

fn do_is_move_sane<C: generic::Color>(b: &Board, mv: Move) -> bool {
    let src_cell = b.get(mv.src);
    let pawn = Cell::from_parts(C::COLOR, Piece::Pawn);
    let dst = Bitboard::from_coord(mv.dst);
    let bad_dsts = b.color(C::COLOR) | b.piece2(C::COLOR.inv(), Piece::King);
    if let Some(c) = mv.side {
        if c != C::COLOR {
            return false;
        }
    }
    match mv.kind {
        MoveKind::Null => true,
        MoveKind::CastlingKingside => {
            b.r.castling.has(C::COLOR, CastlingSide::King)
                && (b.all & castling::pass(C::COLOR, CastlingSide::King)).is_empty()
        }
        MoveKind::CastlingQueenside => {
            b.r.castling.has(C::COLOR, CastlingSide::Queen)
                && (b.all & castling::pass(C::COLOR, CastlingSide::Queen)).is_empty()
        }
        MoveKind::Simple => {
            (dst & bad_dsts).is_empty() && src_cell.color() == Some(C::COLOR) && src_cell != pawn
        }
        kind => {
            // This is a pawn move
            if (dst & bad_dsts).is_nonempty() || src_cell != pawn {
                return false;
            }
            match kind {
                MoveKind::Enpassant => {
                    if let Some(p) = b.r.enpassant {
                        unsafe {
                            return (p == mv.src.add_unchecked(1) || p == mv.src.add_unchecked(-1))
                                && mv.dst
                                    == p.add_unchecked(geometry::pawn_forward_delta(C::COLOR));
                        }
                    }
                    false
                }
                MoveKind::PawnDouble => {
                    let must_empty = match C::COLOR {
                        Color::White => Bitboard::from_raw(0x0101 << (mv.src.index() - 16)),
                        Color::Black => Bitboard::from_raw(0x010100 << mv.src.index()),
                    };
                    (b.all & must_empty).is_empty()
                }
                MoveKind::PawnSimple
                | MoveKind::PromoteKnight
                | MoveKind::PromoteBishop
                | MoveKind::PromoteRook
                | MoveKind::PromoteQueen => true,
                MoveKind::Null
                | MoveKind::CastlingKingside
                | MoveKind::CastlingQueenside
                | MoveKind::Simple => unreachable!(),
            }
        }
    }
}

pub fn is_move_sane(b: &Board, mv: Move) -> bool {
    match b.r.side {
        Color::White => do_is_move_sane::<generic::White>(b, mv),
        Color::Black => do_is_move_sane::<generic::Black>(b, mv),
    }
}

fn do_is_move_semilegal<C: generic::Color>(b: &Board, mv: Move) -> bool {
    match mv.kind {
        MoveKind::Null => false,
        MoveKind::CastlingKingside => {
            let tmp = unsafe { mv.src.add_unchecked(1) };
            !movegen::do_is_cell_attacked::<C::Inv>(b, mv.src)
                && !movegen::do_is_cell_attacked::<C::Inv>(b, tmp)
        }
        MoveKind::CastlingQueenside => {
            let tmp = unsafe { mv.src.add_unchecked(-1) };
            !movegen::do_is_cell_attacked::<C::Inv>(b, mv.src)
                && !movegen::do_is_cell_attacked::<C::Inv>(b, tmp)
        }
        MoveKind::PawnSimple
        | MoveKind::PromoteKnight
        | MoveKind::PromoteBishop
        | MoveKind::PromoteRook
        | MoveKind::PromoteQueen => {
            let dst_cell = b.r.get(mv.dst);
            (mv.dst.file() == mv.src.file()) == dst_cell.is_empty()
        }
        MoveKind::Simple => {
            let src_cell = b.r.get(mv.src);
            let dst = Bitboard::from_coord(mv.dst);
            match src_cell.piece() {
                Some(Piece::King) => (attack::king(mv.src) & dst).is_nonempty(),
                Some(Piece::Knight) => (attack::knight(mv.src) & dst).is_nonempty(),
                Some(Piece::Bishop) => (attack::bishop(mv.src, b.all) & dst).is_nonempty(),
                Some(Piece::Rook) => (attack::rook(mv.src, b.all) & dst).is_nonempty(),
                Some(Piece::Queen) => {
                    (attack::bishop(mv.src, b.all) & dst).is_nonempty()
                        || (attack::rook(mv.src, b.all) & dst).is_nonempty()
                }
                _ => unreachable!(),
            }
        }
        MoveKind::Enpassant | MoveKind::PawnDouble => true,
    }
}

pub unsafe fn is_move_semilegal_unchecked(b: &Board, mv: Move) -> bool {
    match b.r.side {
        Color::White => do_is_move_semilegal::<generic::White>(b, mv),
        Color::Black => do_is_move_semilegal::<generic::Black>(b, mv),
    }
}

pub unsafe fn is_move_legal_unchecked(b: &Board, mv: Move) -> bool {
    let mut b_copy = b.clone();
    let _ = make_move_unchecked(&mut b_copy, mv);
    !b_copy.is_opponent_king_attacked()
}

pub fn semi_validate(b: &Board, mv: Move) -> Result<(), ValidateError> {
    if !is_move_sane(b, mv) {
        return Err(ValidateError::NotSane);
    }
    if unsafe { !is_move_semilegal_unchecked(b, mv) } {
        return Err(ValidateError::NotSemiLegal);
    }
    Ok(())
}

pub fn validate(b: &Board, mv: Move) -> Result<(), ValidateError> {
    semi_validate(b, mv)?;
    match unsafe { is_move_legal_unchecked(b, mv) } {
        true => Ok(()),
        false => Err(ValidateError::NotLegal),
    }
}

pub fn make_move_weak(b: &Board, mv: Move) -> Result<Board, ValidateError> {
    if !is_move_sane(b, mv) {
        return Err(ValidateError::NotSane);
    }
    let mut b_copy = b.clone();
    let _ = unsafe { make_move_unchecked(&mut b_copy, mv) };
    if b_copy.is_opponent_king_attacked() {
        return Err(ValidateError::NotLegal);
    }
    Ok(b_copy)
}

pub fn make_move(b: &Board, mv: Move) -> Result<Board, ValidateError> {
    semi_validate(b, mv)?;
    let mut b_copy = b.clone();
    let _ = unsafe { make_move_unchecked(&mut b_copy, mv) };
    if b_copy.is_opponent_king_attacked() {
        return Err(ValidateError::NotLegal);
    }
    Ok(b_copy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use std::mem;

    #[test]
    fn test_size() {
        assert_eq!(mem::size_of::<Move>(), 4);
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
            let m = Move::from_str_semilegal(mv_str, &b).unwrap();
            b = b.make_move(m).unwrap();
            assert_eq!(b.as_fen(), fen_str);
            assert_eq!(b.raw().try_into(), Ok(b.clone()));
        }
    }

    #[test]
    fn test_undo() {
        let mut b = Board::try_from_fen(
            "r1bqk2r/ppp2ppp/2np1n2/1Bb1p3/4P3/2PP1N2/PP3PPP/RNBQK2R w KQkq - 0 6",
        )
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
            let m = Move::from_str_semilegal(mv_str, &b).unwrap();
            let u = unsafe { make_semilegal_move_unchecked(&mut b, m).unwrap() };
            assert_eq!(b.as_fen(), fen_str);
            assert_eq!(b.raw().try_into(), Ok(b.clone()));
            unsafe { unmake_move_unchecked(&mut b, m, u) };
            assert_eq!(b, b_copy);
        }
    }

    #[test]
    fn test_pawns() {
        let mut b = Board::try_from_fen("3K4/3p4/8/3PpP2/8/5p2/6P1/2k5 w - e6 0 1").unwrap();
        let b_copy = b.clone();

        for (mv_str, fen_str) in [
            ("g2g3", "3K4/3p4/8/3PpP2/8/5pP1/8/2k5 b - - 0 1"),
            ("g2g4", "3K4/3p4/8/3PpP2/6P1/5p2/8/2k5 b - g3 0 1"),
            ("g2f3", "3K4/3p4/8/3PpP2/8/5P2/8/2k5 b - - 0 1"),
            ("d5e6", "3K4/3p4/4P3/5P2/8/5p2/6P1/2k5 b - - 0 1"),
            ("f5e6", "3K4/3p4/4P3/3P4/8/5p2/6P1/2k5 b - - 0 1"),
        ] {
            let m = Move::from_str_semilegal(mv_str, &b).unwrap();
            let u = unsafe { make_semilegal_move_unchecked(&mut b, m).unwrap() };
            assert_eq!(b.as_fen(), fen_str);
            assert_eq!(b.raw().try_into(), Ok(b.clone()));
            unsafe { unmake_move_unchecked(&mut b, m, u) };
            assert_eq!(b, b_copy);
        }
    }

    #[test]
    fn test_legal() {
        let b = Board::try_from_fen(
            "r1bqk2r/ppp2ppp/2np1n2/1Bb1p3/4P3/2PP1N2/PP3PPP/RNBQK2R w KQkq - 0 6",
        )
        .unwrap();

        let m = Move::from_str("e1c1", &b).unwrap();
        assert!(!is_move_sane(&b, m));
        assert_eq!(m.semi_validate(&b), Err(ValidateError::NotSane));

        let m = Move::from_str("b5e8", &b).unwrap();
        assert!(!is_move_sane(&b, m));
        assert_eq!(m.semi_validate(&b), Err(ValidateError::NotSane));

        let m = Move::from_str("a3a4", &b).unwrap();
        assert!(!is_move_sane(&b, m));
        assert_eq!(m.semi_validate(&b), Err(ValidateError::NotSane));

        let m = Move::from_str("e1d1", &b).unwrap();
        assert!(!is_move_sane(&b, m));
        assert_eq!(m.semi_validate(&b), Err(ValidateError::NotSane));

        assert_eq!(
            Move::from_str("c3c5", &b),
            Err(BasicParseError::Create(CreateError::NotWellFormed))
        );
    }
}
