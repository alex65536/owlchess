//! Utilities to work with moves in SAN format

use super::base::{self, CreateError, MoveKind, PromotePiece, ValidateError};
use super::uci;
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::movegen::{self, MovePush};
use crate::types::{CastlingSide, Coord, CoordParseError, File, Piece, Rank};
use crate::{bitboard_consts, geometry};

use std::fmt;
use std::marker::PhantomData;
use std::str::{self, FromStr};

use thiserror::Error;

/// Error parsing SAN representation from string
#[derive(Debug, Clone, Error, Eq, PartialEq)]
pub enum RawParseError {
    /// String is empty
    #[error("string is empty")]
    EmptyString,
    /// Destination cell is invalid
    #[error("invalid destination cell")]
    InvalidDst(#[from] CoordParseError),
    /// Extra bytes in non-pawn move
    #[error("non-pawn move too long")]
    NonPawnMoveTooLong,
    /// String for pawn move is too short
    #[error("pawn move too short")]
    PawnMoveTooShort,
    /// Extra bytes in pawn move
    #[error("pawn move too long")]
    PawnMoveTooLong,
    /// Parsing failed for unspecified reasons
    #[error("syntax error")]
    Syntax,
}

/// Error converting SAN move into [`moves::Move`](super::Move)
#[derive(Debug, Clone, Error, Eq, PartialEq)]
pub enum IntoMoveError {
    /// Resulting move is not well-formed
    ///
    /// Note that, when the move is not well-formed, you can also get [`IntoMoveError::NotFound`] error.
    #[error("cannot create move: {0}")]
    Create(#[from] CreateError),
    /// Resulting mvoe is not legal
    ///
    /// Note that, when the move is not legal, you can also get [`IntoMoveError::NotFound`] error.
    #[error("invalid move: {0}")]
    Validate(#[from] ValidateError),
    /// Capture sign is put when the move is non-capture
    #[error("got capture sign on a non-capture move")]
    CaptureExpected,
    /// Cannot find a corresponding legal move described by the given SAN string
    #[error("no such move")]
    NotFound,
    /// The description given by SAN string is ambiguous
    #[error("ambiguous move (candidates are at least `{0}` and `{1}`)")]
    Ambiguity(base::Move, base::Move),
}

/// Error parsing [`moves::Move`](super::Move) from SAN string
#[derive(Debug, Clone, Error, Eq, PartialEq)]
pub enum ParseError {
    /// Cannot parse SAN string
    #[error("cannot parse move: {0}")]
    Parse(#[from] RawParseError),
    /// Cannot convert a parsed string into a legal move
    #[error("cannot convert move: {0}")]
    Convert(#[from] IntoMoveError),
}

/// Style for formatiing SAN moves
///
/// Note that the style can be used only for _formatting_. Move parser only accepts
/// ASCII characters as piece names and doesn't accept Unicode pieces,
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Style {
    /// Use capical Latin letters for pieces
    Algebraic,
    /// Use Unicode chess symbols for pieces
    Utf8,
}

trait PieceTheme {
    fn marker() -> PhantomData<Self>;
    fn piece_to_char(piece: Piece) -> char;
    fn promote_sign() -> &'static str;

    fn promote_to_char(promote: PromotePiece) -> char {
        Self::piece_to_char(promote.into())
    }
}

struct Utf8Theme;

impl PieceTheme for Utf8Theme {
    fn marker() -> PhantomData<Self> {
        PhantomData
    }

    fn promote_sign() -> &'static str {
        ""
    }

    fn piece_to_char(piece: Piece) -> char {
        match piece {
            Piece::Pawn => '♙',
            Piece::Knight => '♘',
            Piece::Bishop => '♗',
            Piece::Rook => '♖',
            Piece::Queen => '♕',
            Piece::King => '♔',
        }
    }
}

struct AlgebraicTheme;

impl PieceTheme for AlgebraicTheme {
    fn marker() -> PhantomData<Self> {
        PhantomData
    }

    fn promote_sign() -> &'static str {
        "="
    }

    fn piece_to_char(piece: Piece) -> char {
        match piece {
            Piece::Pawn => 'P',
            Piece::Knight => 'N',
            Piece::Bishop => 'B',
            Piece::Rook => 'R',
            Piece::Queen => 'Q',
            Piece::King => 'K',
        }
    }
}

/// Parsed SAN string, without check indicator
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Data {
    /// Move in UCI format
    Uci(uci::Move),
    /// Castling
    Castling(CastlingSide),
    /// Simple pawn move
    PawnMove {
        /// Destination square
        dst: Coord,
        /// Piece to promote, if any
        promote: Option<PromotePiece>,
    },
    /// Pawn capture
    PawnCapture {
        /// Source file
        src: File,
        /// Destination square
        dst: Coord,
        /// Piece to promote, if any
        promote: Option<PromotePiece>,
    },
    /// Simplified pawn capture (like `cd`, `fe`, etc.)
    PawnCaptureShort {
        /// Source file
        src: File,
        /// Destination file
        dst: File,
        /// Piece to promote, if any
        promote: Option<PromotePiece>,
    },
    /// Non-pawn move
    Simple {
        /// Piece to move
        piece: Piece,
        /// Source file, if specified
        file: Option<File>,
        /// Source rank, if specified
        rank: Option<Rank>,
        /// Is the move capture?
        is_capture: bool,
        /// Destination square
        dst: Coord,
    },
}

struct PromoteFmt<T: PieceTheme>(Option<PromotePiece>, PhantomData<T>);

impl<T: PieceTheme> fmt::Display for PromoteFmt<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self.0 {
            None => Ok(()),
            Some(promote) => write!(f, "{}{}", T::promote_sign(), T::promote_to_char(promote)),
        }
    }
}

struct AmbigDetector {
    mv: base::Move,
    sim_any: bool,
    sim_file: bool,
    sim_rank: bool,
}

impl AmbigDetector {
    fn new(mv: base::Move) -> Self {
        Self {
            mv,
            sim_any: false,
            sim_file: false,
            sim_rank: false,
        }
    }

    fn file(&self) -> Option<File> {
        if self.sim_any && (self.sim_rank || !self.sim_file) {
            return Some(self.mv.src().file());
        }
        None
    }

    fn rank(&self) -> Option<Rank> {
        if self.sim_any && self.sim_file {
            return Some(self.mv.src().rank());
        }
        None
    }
}

impl MovePush for AmbigDetector {
    fn push(&mut self, mv: base::Move) {
        if mv == self.mv {
            return;
        }
        self.sim_any = true;
        if self.mv.src().file() == mv.src().file() {
            self.sim_file = true;
        }
        if self.mv.src().rank() == mv.src().rank() {
            self.sim_rank = true;
        }
    }
}

#[derive(Copy, Clone)]
enum AmbigSearcherState {
    Empty,
    Found(base::Move),
    Ambiguity(base::Move, base::Move),
}

struct AmbigSearcher {
    srcs: Bitboard,
    state: AmbigSearcherState,
}

impl AmbigSearcher {
    fn new(file: Option<File>, rank: Option<Rank>) -> AmbigSearcher {
        let mut srcs = Bitboard::FULL;
        if let Some(file) = file {
            srcs &= bitboard_consts::file(file);
        }
        if let Some(rank) = rank {
            srcs &= bitboard_consts::rank(rank);
        }
        AmbigSearcher {
            srcs,
            state: AmbigSearcherState::Empty,
        }
    }

    fn get_move(&self) -> Result<base::Move, IntoMoveError> {
        match &self.state {
            AmbigSearcherState::Empty => Err(IntoMoveError::NotFound),
            AmbigSearcherState::Found(mv) => Ok(*mv),
            AmbigSearcherState::Ambiguity(mv, mv2) => Err(IntoMoveError::Ambiguity(*mv, *mv2)),
        }
    }
}

impl MovePush for AmbigSearcher {
    fn push(&mut self, mv: base::Move) {
        if !self.srcs.has(mv.src()) {
            return;
        }
        self.state = match self.state {
            AmbigSearcherState::Empty => AmbigSearcherState::Found(mv),
            AmbigSearcherState::Found(mv2) => AmbigSearcherState::Ambiguity(mv, mv2),
            s @ AmbigSearcherState::Ambiguity(_, _) => s,
        };
    }
}

impl Data {
    /// Returns the wrapper which helps to format the move with the given style `style`
    ///
    /// See [`Move::styled()`] doc for details.
    #[inline]
    pub fn styled(&self, style: Style) -> StyledData<'_> {
        StyledData(self, style)
    }

    /// Creates the parsed SAN from move `mv` in position `b`
    pub fn from_move(mv: base::Move, b: &Board) -> Data {
        match mv.kind() {
            MoveKind::Null => Data::Uci(uci::Move::Null),
            MoveKind::PawnDouble => Data::PawnMove {
                dst: mv.dst(),
                promote: None,
            },
            MoveKind::Enpassant => Data::PawnCapture {
                src: mv.src().file(),
                dst: mv.dst(),
                promote: None,
            },
            MoveKind::PawnSimple
            | MoveKind::PromoteKnight
            | MoveKind::PromoteBishop
            | MoveKind::PromoteRook
            | MoveKind::PromoteQueen => {
                if mv.src().file() == mv.dst().file() {
                    Data::PawnMove {
                        dst: mv.dst(),
                        promote: mv.kind().try_into().ok(),
                    }
                } else {
                    Data::PawnCapture {
                        src: mv.src().file(),
                        dst: mv.dst(),
                        promote: mv.kind().try_into().ok(),
                    }
                }
            }
            MoveKind::CastlingKingside | MoveKind::CastlingQueenside => {
                Data::Castling(mv.kind().try_into().unwrap())
            }
            MoveKind::Simple => {
                let piece = b.get(mv.src()).piece().unwrap();
                let is_capture = b.get(mv.dst()).is_occupied();
                let mut detector = AmbigDetector::new(mv);
                movegen::san_candidates(b, piece, mv.dst(), &mut detector);
                Data::Simple {
                    piece,
                    file: detector.file(),
                    rank: detector.rank(),
                    is_capture,
                    dst: mv.dst(),
                }
            }
        }
    }

    /// Converts the parsed SAN into [`moves::Move`](super::Move) in the position `b`
    pub fn into_move(self, b: &Board) -> Result<base::Move, IntoMoveError> {
        match self {
            Self::Uci(uci) => {
                let mv = uci.into_move(b)?;
                mv.validate(b)?;
                Ok(mv)
            }
            Self::Castling(side) => {
                let mv = base::Move::from_castling(b.side(), side);
                mv.validate(b)?;
                Ok(mv)
            }
            Self::PawnMove { dst, promote } => {
                if dst.rank() == geometry::promote_src_rank(b.side().inv()) {
                    return Err(IntoMoveError::Create(CreateError::NotWellFormed));
                }
                let mut src = dst.add(-geometry::pawn_forward_delta(b.side()));
                let mut kind = MoveKind::PawnSimple;
                if !b.get(src).is_occupied() {
                    src = Coord::from_parts(dst.file(), geometry::double_move_src_rank(b.side()));
                    kind = MoveKind::PawnDouble;
                }
                let mv = base::Move::new(
                    promote.map(MoveKind::from).unwrap_or(kind),
                    src,
                    dst,
                    b.side(),
                )?;
                mv.validate(b)?;
                Ok(mv)
            }
            Self::PawnCapture { src, dst, promote } => {
                if dst.rank() == geometry::promote_src_rank(b.side().inv()) {
                    return Err(IntoMoveError::Create(CreateError::NotWellFormed));
                }
                let mut kind = MoveKind::PawnSimple;
                if Some(dst) == b.r.ep_dest() {
                    kind = MoveKind::Enpassant;
                }
                if kind != MoveKind::Enpassant && b.get(dst).is_free() {
                    return Err(IntoMoveError::CaptureExpected);
                }
                let src =
                    Coord::from_parts(src, dst.rank()).add(-geometry::pawn_forward_delta(b.side()));
                let mv = base::Move::new(
                    promote.map(MoveKind::from).unwrap_or(kind),
                    src,
                    dst,
                    b.side(),
                )?;
                mv.validate(b)?;
                Ok(mv)
            }
            Self::PawnCaptureShort { src, dst, promote } => {
                let mut detector = AmbigSearcher::new(None, None);
                movegen::san_pawn_capture_candidates(b, src, dst, promote, &mut detector);
                detector.get_move()
            }
            Self::Simple {
                piece,
                file,
                rank,
                is_capture,
                dst,
            } => {
                if is_capture && b.get(dst).is_free() {
                    return Err(IntoMoveError::CaptureExpected);
                }
                let mut detector = AmbigSearcher::new(file, rank);
                movegen::san_candidates(b, piece, dst, &mut detector);
                detector.get_move()
            }
        }
    }

    pub(self) fn do_fmt<P: PieceTheme>(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> Result<(), fmt::Error> {
        match *self {
            Self::Uci(uci) => fmt::Display::fmt(&uci, f),
            Self::Castling(CastlingSide::King) => write!(f, "O-O"),
            Self::Castling(CastlingSide::Queen) => write!(f, "O-O-O"),
            Self::PawnMove { dst, promote } => {
                write!(f, "{}{}", dst, PromoteFmt(promote, P::marker()))
            }
            Self::PawnCapture { src, dst, promote } => {
                write!(
                    f,
                    "{}x{}{}",
                    src.as_char(),
                    dst,
                    PromoteFmt(promote, P::marker())
                )
            }
            Self::PawnCaptureShort { src, dst, promote } => write!(
                f,
                "{}{}{}",
                src.as_char(),
                dst.as_char(),
                PromoteFmt(promote, P::marker())
            ),
            Self::Simple {
                piece,
                file,
                rank,
                is_capture,
                dst,
            } => {
                if piece == Piece::Pawn {
                    panic!("cannot store pawn move as Move::Simple");
                }
                write!(f, "{}", P::piece_to_char(piece))?;
                if let Some(file) = file {
                    write!(f, "{}", file.as_char())?;
                }
                if let Some(rank) = rank {
                    write!(f, "{}", rank.as_char())?;
                }
                if is_capture {
                    write!(f, "x")?;
                }
                write!(f, "{}", dst)
            }
        }
    }
}

impl fmt::Display for Data {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        self.do_fmt::<AlgebraicTheme>(f)
    }
}

impl<'a> fmt::Display for StyledData<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self.1 {
            Style::Algebraic => self.0.do_fmt::<AlgebraicTheme>(f),
            Style::Utf8 => self.0.do_fmt::<Utf8Theme>(f),
        }
    }
}

impl FromStr for Data {
    type Err = RawParseError;

    fn from_str(data: &str) -> Result<Data, Self::Err> {
        if data == "O-O" || data == "0-0" {
            return Ok(Self::Castling(CastlingSide::King));
        }
        if data == "O-O-O" || data == "0-0-0" {
            return Ok(Self::Castling(CastlingSide::Queen));
        }
        if data.is_empty() {
            return Err(RawParseError::EmptyString);
        }
        if let Ok(mv) = uci::Move::from_str(data) {
            return Ok(Self::Uci(mv));
        }

        let bytes = data.as_bytes();

        if let first @ (b'N' | b'B' | b'R' | b'Q' | b'K') = bytes[0] {
            let piece = match first {
                b'N' => Piece::Knight,
                b'B' => Piece::Bishop,
                b'R' => Piece::Rook,
                b'Q' => Piece::Queen,
                b'K' => Piece::King,
                _ => unreachable!(),
            };
            let bytes = &bytes[1..];
            let (bytes, dst_bytes) = bytes.split_at(bytes.len() - 2);
            let dst = Coord::from_str(str::from_utf8(dst_bytes).unwrap())?;
            let (file, bytes) = match bytes.first() {
                Some(b @ b'a'..=b'h') => (File::from_char(*b as char), &bytes[1..]),
                _ => (None, bytes),
            };
            let (rank, bytes) = match bytes.first() {
                Some(b @ b'1'..=b'8') => (Rank::from_char(*b as char), &bytes[1..]),
                _ => (None, bytes),
            };
            let (is_capture, bytes) = match bytes.first() {
                Some(b'x' | b':') => (true, &bytes[1..]),
                _ => (false, bytes),
            };
            if !bytes.is_empty() {
                return Err(RawParseError::NonPawnMoveTooLong);
            }
            return Ok(Data::Simple {
                piece,
                file,
                rank,
                is_capture,
                dst,
            });
        }

        let (promote, bytes) = match bytes.split_last() {
            Some((b @ (b'N' | b'B' | b'R' | b'Q'), rest)) => {
                let promote = match b {
                    b'N' => PromotePiece::Knight,
                    b'B' => PromotePiece::Bishop,
                    b'R' => PromotePiece::Rook,
                    b'Q' => PromotePiece::Queen,
                    _ => unreachable!(),
                };
                let rest = match rest.split_last() {
                    Some((b'=', data)) => data,
                    _ => rest,
                };
                (Some(promote), rest)
            }
            _ => (None, bytes),
        };

        if bytes.len() < 2 {
            return Err(RawParseError::PawnMoveTooShort);
        }
        if bytes.len() == 2 && matches!(bytes[0], b'a'..=b'h') && matches!(bytes[1], b'a'..=b'h') {
            return Ok(Data::PawnCaptureShort {
                src: File::from_char(bytes[0] as char).unwrap(),
                dst: File::from_char(bytes[1] as char).unwrap(),
                promote,
            });
        }

        let (bytes, dst_bytes) = bytes.split_at(bytes.len() - 2);
        let dst = Coord::from_str(str::from_utf8(dst_bytes).unwrap())?;

        match bytes.len() {
            0 => Ok(Data::PawnMove { dst, promote }),
            1 => Err(RawParseError::Syntax),
            2 => {
                if !matches!(bytes[0], b'a'..=b'h') || !matches!(bytes[1], b':' | b'x') {
                    return Err(RawParseError::Syntax);
                }
                Ok(Data::PawnCapture {
                    src: File::from_char(bytes[0] as char).unwrap(),
                    dst,
                    promote,
                })
            }
            _ => Err(RawParseError::PawnMoveTooLong),
        }
    }
}

/// Check indication
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum CheckMark {
    /// Check (a.k.a. "+")
    Single,
    /// Double check (a.k.a "++")
    ///
    /// This one is not set while converting the move into SAN and is used primarily for parsing.
    /// Note that in some notations "++" may denote checkmate, but it's still parsed as [`CheckMark::double`].
    Double,
    /// Checkmate (a.k.a "#")
    Checkmate,
}

/// Parsed SAN move with a [`CheckMark`]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Move {
    /// Data without check mark
    pub data: Data,
    /// Check mark, if any
    pub check: Option<CheckMark>,
}

/// Wrapper to format [`Data`] with the given style
///
/// See [`Move::styled()`] doc for details.
pub struct StyledData<'a>(&'a Data, Style);

/// Wrapper to format [`Move`] with the given style
///
/// See [`Move::styled()`] doc for details.
pub struct StyledMove<'a>(&'a Move, Style);

impl Move {
    /// Returns the wrapper which helps to format the move with the given style `style`
    ///
    /// The resulting wrapper implements [`fmt::Display`], so can be used with
    /// `write!()`, `println!()`, or `ToString::to_string`.
    ///
    /// The usage is similar to [`RawBoard::pretty()`](crate::RawBoard::pretty) or
    /// [`moves::Move::styled()`](super::Move::styled).
    #[inline]
    pub fn styled(&self, style: Style) -> StyledMove<'_> {
        StyledMove(self, style)
    }

    pub(self) fn do_fmt<P: PieceTheme>(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> Result<(), fmt::Error> {
        self.data.do_fmt::<P>(f)?;
        match self.check {
            Some(CheckMark::Single) => write!(f, "+")?,
            Some(CheckMark::Double) => write!(f, "++")?,
            Some(CheckMark::Checkmate) => write!(f, "#")?,
            None => {}
        };
        Ok(())
    }

    /// Creates the parsed SAN from move `mv` in position `b`
    pub fn from_move(mv: base::Move, b: &Board) -> Result<Move, ValidateError> {
        let data = Data::from_move(mv, b);
        let b_copy = base::make_move(b, mv)?;
        let check = if b_copy.is_check() {
            if movegen::has_legal_moves(&b_copy) {
                Some(CheckMark::Single)
            } else {
                Some(CheckMark::Checkmate)
            }
        } else {
            None
        };
        Ok(Move { data, check })
    }

    /// Converts the parsed SAN into [`moves::Move`](super::Move) in the position `b`
    pub fn into_move(self, b: &Board) -> Result<base::Move, IntoMoveError> {
        self.data.into_move(b)
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        self.do_fmt::<AlgebraicTheme>(f)
    }
}

impl<'a> fmt::Display for StyledMove<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self.1 {
            Style::Algebraic => self.0.do_fmt::<AlgebraicTheme>(f),
            Style::Utf8 => self.0.do_fmt::<Utf8Theme>(f),
        }
    }
}

impl FromStr for Move {
    type Err = RawParseError;

    fn from_str(s: &str) -> Result<Move, Self::Err> {
        let (check, s) = match s.as_bytes().split_last() {
            Some((b'#' | b'x', rest)) => {
                (Some(CheckMark::Checkmate), str::from_utf8(rest).unwrap())
            }
            Some((b'+', rest)) => match rest.split_last() {
                Some((b'+', rest2)) => (Some(CheckMark::Double), str::from_utf8(rest2).unwrap()),
                _ => (Some(CheckMark::Single), str::from_utf8(rest).unwrap()),
            },
            _ => (None, s),
        };
        Ok(Move {
            data: Data::from_str(s)?,
            check,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::moves::{base, ValidateError};

    #[test]
    fn test_simple() {
        let mut b = Board::initial();
        for (mv_str, fen_str) in [
            (
                "e4",
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            ),
            (
                "Nc6",
                "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
            ),
            (
                "Nf3",
                "r1bqkbnr/pppppppp/2n5/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 2",
            ),
            (
                "e5",
                "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq e6 0 3",
            ),
            (
                "Bb5",
                "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 3",
            ),
            (
                "Nf6",
                "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4",
            ),
            (
                "O-O",
                "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 3 4",
            ),
            (
                "Nxe4",
                "r1bqkb1r/pppp1ppp/2n5/1B2p3/4n3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 0 5",
            ),
            (
                "Re1",
                "r1bqkb1r/pppp1ppp/2n5/1B2p3/4n3/5N2/PPPP1PPP/RNBQR1K1 b kq - 1 5",
            ),
            (
                "Qh4",
                "r1b1kb1r/pppp1ppp/2n5/1B2p3/4n2q/5N2/PPPP1PPP/RNBQR1K1 w kq - 2 6",
            ),
            (
                "Kh1",
                "r1b1kb1r/pppp1ppp/2n5/1B2p3/4n2q/5N2/PPPP1PPP/RNBQR2K b kq - 3 6",
            ),
        ] {
            let m = base::Move::from_san(mv_str, &b).unwrap();
            assert_eq!(
                Move::from_str(mv_str).unwrap(),
                Move::from_move(m, &b).unwrap()
            );
            assert_eq!(m.san(&b).unwrap().to_string(), mv_str.to_string());
            b = b.make_move(m).unwrap();
            assert_eq!(b.as_fen(), fen_str);
            assert_eq!(b.raw().try_into(), Ok(b.clone()));
        }
    }

    #[test]
    fn test_pawn_conflict() {
        let b = Board::from_str("8/8/1p6/2P5/1p5k/2P5/7K/8 w - - 0 1").unwrap();
        assert!(matches!(
            base::Move::from_san("cb", &b),
            Err(ParseError::Convert(IntoMoveError::Ambiguity(_, _)))
        ));
        assert_eq!(
            base::Move::from_san("cxb4", &b).unwrap(),
            base::Move::from_uci("c3b4", &b).unwrap()
        );
        assert_eq!(
            base::Move::from_san("cxb6", &b).unwrap(),
            base::Move::from_uci("c5b6", &b).unwrap()
        );
        assert_eq!(
            base::Move::from_san("cd", &b),
            Err(ParseError::Convert(IntoMoveError::NotFound))
        );
    }

    #[test]
    fn test_conflict() {
        let b = Board::from_str("k5K1/8/5q2/6n1/8/2P5/5q2/8 b - - 0 1").unwrap();
        assert_eq!(
            base::Move::from_san("Qe5", &b).unwrap(),
            base::Move::from_uci("f6e5", &b).unwrap()
        );
        assert!(matches!(
            base::Move::from_san("Qd4", &b),
            Err(ParseError::Convert(IntoMoveError::Ambiguity(_, _)))
        ));
        assert_eq!(
            base::Move::from_san("Qxc3", &b).unwrap(),
            base::Move::from_uci("f6c3", &b).unwrap()
        );
        assert_eq!(
            base::Move::from_san("Qb2", &b).unwrap(),
            base::Move::from_uci("f2b2", &b).unwrap()
        );
        assert_eq!(
            base::Move::from_san("Qa1", &b),
            Err(ParseError::Convert(IntoMoveError::NotFound))
        );
        assert_eq!(
            base::Move::from_san("Qg5", &b),
            Err(ParseError::Convert(IntoMoveError::NotFound))
        );
        assert_eq!(
            base::Move::from_san("Qe3", &b).unwrap(),
            base::Move::from_uci("f2e3", &b).unwrap()
        );
        assert_eq!(
            base::Move::from_san("Q2d4", &b).unwrap(),
            base::Move::from_uci("f2d4", &b).unwrap()
        );
        assert_eq!(
            base::Move::from_san("Q6d4", &b).unwrap(),
            base::Move::from_uci("f6d4", &b).unwrap()
        );
        assert_eq!(
            base::Move::from_san("Qf6d4", &b).unwrap(),
            base::Move::from_uci("f6d4", &b).unwrap()
        );
        assert_eq!(
            base::Move::from_san("Qfe5", &b).unwrap(),
            base::Move::from_uci("f6e5", &b).unwrap()
        );
        assert_eq!(
            base::Move::from_san("Q6e5", &b).unwrap(),
            base::Move::from_uci("f6e5", &b).unwrap()
        );
        assert_eq!(
            base::Move::from_san("Qf6e5", &b).unwrap(),
            base::Move::from_uci("f6e5", &b).unwrap()
        );
        assert_eq!(
            base::Move::from_san("Qge5", &b),
            Err(ParseError::Convert(IntoMoveError::NotFound))
        );
        assert_eq!(
            base::Move::from_san("Q5e5", &b),
            Err(ParseError::Convert(IntoMoveError::NotFound))
        );
        assert_eq!(
            base::Move::from_san("Qg5e5", &b),
            Err(ParseError::Convert(IntoMoveError::NotFound))
        );
        assert!(matches!(
            base::Move::from_san("Qfd4", &b),
            Err(ParseError::Convert(IntoMoveError::Ambiguity(_, _)))
        ));
        assert_eq!(
            base::Move::from_san("Kaa7", &b).unwrap(),
            base::Move::from_uci("a8a7", &b).unwrap()
        );
    }

    #[test]
    fn test_capture() {
        let b = Board::from_str("k5K1/8/p4q2/1P4n1/8/2P5/5q2/8 b - - 0 1").unwrap();
        assert_eq!(
            base::Move::from_san("Qxe5", &b),
            Err(ParseError::Convert(IntoMoveError::CaptureExpected))
        );
        assert_eq!(
            base::Move::from_san("Qe5", &b).unwrap(),
            base::Move::from_uci("f6e5", &b).unwrap()
        );
        assert_eq!(
            base::Move::from_san("Qc3", &b).unwrap(),
            base::Move::from_uci("f6c3", &b).unwrap()
        );
        assert_eq!(
            base::Move::from_san("Qxc3", &b).unwrap(),
            base::Move::from_uci("f6c3", &b).unwrap()
        );
        assert_eq!(
            base::Move::from_san("axb5", &b).unwrap(),
            base::Move::from_uci("a6b5", &b).unwrap()
        );
        assert_eq!(
            base::Move::from_san("b5", &b),
            Err(ParseError::Convert(IntoMoveError::Validate(
                ValidateError::NotSemiLegal
            )))
        );
        assert_eq!(
            base::Move::from_san("axa5", &b),
            Err(ParseError::Convert(IntoMoveError::CaptureExpected))
        );
        assert_eq!(
            base::Move::from_san("a5", &b).unwrap(),
            base::Move::from_uci("a6a5", &b).unwrap()
        );
    }

    #[test]
    fn test_pawns() {
        for (fen_str, uci_str, mv_str, real_mv_str) in [
            ("8/8/8/4p3/3P4/8/3P4/5K1k w - - 0 1", "d4e5", "de", "dxe5"),
            ("8/8/8/2PpP3/8/8/5k1K/8 w - d6 0 1", "c5d6", "cd", "cxd6"),
            ("8/8/8/2PpP3/8/8/5k1K/8 w - d6 0 1", "e5d6", "ed", "exd6"),
            ("8/8/8/3pP3/2P5/8/5k1K/8 w - d6 0 1", "c4d5", "cd", "cxd5"),
            ("8/8/8/3pP3/2P5/8/5k1K/8 w - d6 0 1", "e5d6", "ed", "exd6"),
            (
                "2n2n1n/3P2P1/8/8/8/8/3K1k2/8 w - - 0 1",
                "d7d8n",
                "d8N",
                "d8=N",
            ),
            (
                "2n2n1n/3P2P1/8/8/8/8/3K1k2/8 w - - 0 1",
                "d7c8b",
                "dcB",
                "dxc8=B",
            ),
            (
                "2n2n1n/3P2P1/8/8/8/8/3K1k2/8 w - - 0 1",
                "g7f8r",
                "gf=R",
                "gxf8=R+",
            ),
            (
                "2n2n1n/3P2P1/8/8/8/8/3K1k2/8 w - - 0 1",
                "g7h8q",
                "gh=Q",
                "gxh8=Q",
            ),
            ("8/8/8/8/3p3k/2P5/1PP4K/8 w - - 0 1", "b2b3", "b3", "b3"),
            ("8/8/8/8/3p3k/2P5/1PP4K/8 w - - 0 1", "b2b4", "b4", "b4"),
            ("8/8/8/8/3p3k/2P5/1PP4K/8 w - - 0 1", "c3c4", "c4", "c4"),
            ("8/8/8/8/3p3k/2P5/1PP4K/8 w - - 0 1", "c3d4", "cd", "cxd4"),
        ] {
            let b = Board::from_str(fen_str).unwrap();
            let m = base::Move::from_san(mv_str, &b).unwrap();
            assert_eq!(m, base::Move::from_uci(uci_str, &b).unwrap());
            assert_eq!(m, base::Move::from_san(real_mv_str, &b).unwrap());
            assert_eq!(m.san(&b).unwrap().to_string(), real_mv_str.to_string());
            m.validate(&b).unwrap();
        }
    }

    #[test]
    fn test_tricky() {
        for (fen_str, uci_str, mv_str) in [
            ("4k3/6K1/8/2N5/8/8/8/N7 w - - 0 1", "a1b3", "Nab3"),
            ("4k3/6K1/8/N7/8/8/8/N7 w - - 0 1", "a1b3", "N1b3"),
            ("4k3/6K1/8/8/8/8/8/N1N5 w - - 0 1", "a1b3", "Nab3"),
            ("4k3/6K1/8/N1N5/8/8/8/N1N5 w - - 0 1", "a1b3", "Na1b3"),
            ("5k2/8/5K2/8/3R3R/8/8/b7 w - - 0 1", "h4f4", "Rf4"),
            ("4k3/6K1/8/2N5/8/1r6/8/N7 w - - 0 1", "a1b3", "Naxb3"),
            ("4k3/6K1/8/N7/8/1r6/8/N7 w - - 0 1", "a1b3", "N1xb3"),
            ("4k3/6K1/8/8/8/1r6/8/N1N5 w - - 0 1", "a1b3", "Naxb3"),
            ("4k3/6K1/8/N1N5/8/1r6/8/N1N5 w - - 0 1", "a1b3", "Na1xb3"),
        ] {
            let b = Board::from_str(fen_str).unwrap();
            let m = base::Move::from_san(mv_str, &b).unwrap();
            assert_eq!(m, base::Move::from_uci(uci_str, &b).unwrap());
            assert_eq!(
                Move::from_str(mv_str).unwrap(),
                Move::from_move(m, &b).unwrap()
            );
            assert_eq!(m.san(&b).unwrap().to_string(), mv_str.to_string());
            m.validate(&b).unwrap();
        }
    }

    #[test]
    fn test_styled() {
        let b = Board::from_str("8/2P5/8/8/8/8/4k1K1/8 w - - 0 1").unwrap();
        assert_eq!(
            base::Move::from_uci("g2h2", &b)
                .unwrap()
                .san(&b)
                .unwrap()
                .styled(Style::Utf8)
                .to_string(),
            "♔h2".to_string()
        );
        assert_eq!(
            base::Move::from_uci("g2h2", &b)
                .unwrap()
                .san(&b)
                .unwrap()
                .styled(Style::Algebraic)
                .to_string(),
            "Kh2".to_string()
        );
        assert_eq!(
            base::Move::from_uci("c7c8b", &b)
                .unwrap()
                .san(&b)
                .unwrap()
                .styled(Style::Utf8)
                .to_string(),
            "c8♗".to_string()
        );
    }

    #[test]
    fn test_check() {
        let b = Board::from_str("1r5k/8/8/8/8/6p1/r7/5K2 b - - 0 1").unwrap();
        assert_eq!(
            base::Move::from_uci("g3g2", &b)
                .unwrap()
                .san(&b)
                .unwrap()
                .to_string(),
            "g2+".to_string(),
        );
        assert_eq!(
            base::Move::from_uci("b8b1", &b)
                .unwrap()
                .san(&b)
                .unwrap()
                .to_string(),
            "Rb1#".to_string(),
        );
        assert_eq!(
            base::Move::from_san("g2", &b).unwrap(),
            base::Move::from_uci("g3g2", &b).unwrap()
        );
        assert_eq!(
            base::Move::from_san("g2+", &b).unwrap(),
            base::Move::from_uci("g3g2", &b).unwrap()
        );
        assert_eq!(
            base::Move::from_san("Rb1", &b).unwrap(),
            base::Move::from_uci("b8b1", &b).unwrap()
        );
        assert_eq!(
            base::Move::from_san("Rb1+", &b).unwrap(),
            base::Move::from_uci("b8b1", &b).unwrap()
        );
        assert_eq!(
            base::Move::from_san("Rb1#", &b).unwrap(),
            base::Move::from_uci("b8b1", &b).unwrap()
        );
    }
}
