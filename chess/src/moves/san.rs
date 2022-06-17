use super::base::{self, CreateError, MoveKind, PromoteKind, ValidateError};
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

#[derive(Debug, Clone, Error, Eq, PartialEq)]
pub enum RawParseError {
    #[error("string is empty")]
    EmptyString,
    #[error("invalid destination cell")]
    InvalidDst(#[from] CoordParseError),
    #[error("pawn move too short")]
    PawnMoveTooShort,
    #[error("pawn move too long")]
    PawnMoveTooLong,
    #[error("syntax error")]
    Syntax,
}

#[derive(Debug, Clone, Error, Eq, PartialEq)]
pub enum IntoMoveError {
    #[error("cannot create move: {0}")]
    Create(#[from] CreateError),
    #[error("invalid move: {0}")]
    Validate(#[from] ValidateError),
    #[error("expected capture")]
    CaptureExpected,
    #[error("move not found")]
    NotFound,
    #[error("ambiguous move (candidates are at least `{0}` and `{1}`)")]
    Ambiguity(base::Move, base::Move),
}

#[derive(Debug, Clone, Error, Eq, PartialEq)]
pub enum ParseError {
    #[error("cannot parse move: {0}")]
    Parse(#[from] RawParseError),
    #[error("cannot convert move: {0}")]
    Convert(#[from] IntoMoveError),
}

trait PieceTheme {
    fn marker() -> PhantomData<Self>;
    fn piece_to_char(piece: Piece) -> char;

    fn promote_to_char(promote: PromoteKind) -> char {
        Self::piece_to_char(promote.piece())
    }
}

struct AlgebraicTheme;

impl PieceTheme for AlgebraicTheme {
    fn marker() -> PhantomData<Self> {
        PhantomData
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

struct PrettyTheme;

impl PieceTheme for PrettyTheme {
    fn marker() -> PhantomData<Self> {
        PhantomData
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

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Data {
    Uci(uci::Move),
    Castling(CastlingSide),
    PawnMove {
        dst: Coord,
        promote: Option<PromoteKind>,
    },
    PawnCapture {
        src: File,
        dst: Coord,
        promote: Option<PromoteKind>,
    },
    PawnCaptureShort {
        src: File,
        dst: File,
        promote: Option<PromoteKind>,
    },
    Simple {
        piece: Piece,
        file: Option<File>,
        rank: Option<Rank>,
        is_capture: bool,
        dst: Coord,
    },
}

pub struct PrettyData<'a>(&'a Data);

struct PromoteFmt<T: PieceTheme>(Option<PromoteKind>, PhantomData<T>);

impl<T: PieceTheme> fmt::Display for PromoteFmt<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self.0 {
            None => Ok(()),
            Some(promote) => write!(f, "={}", T::promote_to_char(promote)),
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
    pub fn pretty(&self) -> PrettyData<'_> {
        PrettyData(self)
    }

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
                        promote: mv.kind().promote(),
                    }
                } else {
                    Data::PawnCapture {
                        src: mv.src().file(),
                        dst: mv.dst(),
                        promote: mv.kind().promote(),
                    }
                }
            }
            MoveKind::CastlingKingside | MoveKind::CastlingQueenside => {
                Data::Castling(mv.kind().castling().unwrap())
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

    pub fn into_move(self, b: &Board) -> Result<base::Move, IntoMoveError> {
        match self {
            Self::Uci(uci) => {
                let mv = uci.into_move(b)?;
                mv.validate(b)?;
                Ok(mv)
            }
            Self::Castling(side) => {
                let mv = base::Move::castling(b.side(), side);
                mv.validate(b)?;
                Ok(mv)
            }
            Self::PawnMove { dst, promote } => {
                if matches!(dst.rank(), Rank::R1 | Rank::R8) {
                    return Err(IntoMoveError::Create(CreateError::NotWellFormed));
                }
                let mut src = dst.add(-geometry::pawn_forward_delta(b.side()));
                let mut kind = MoveKind::PawnSimple;
                if !b.get(src).is_occupied() {
                    src = Coord::from_parts(dst.file(), geometry::double_move_src_rank(b.side()));
                    kind = MoveKind::PawnDouble;
                }
                let mv = base::Move::new(
                    promote.map(MoveKind::from_promote).unwrap_or(kind),
                    src,
                    dst,
                    b.side(),
                )?;
                mv.validate(b)?;
                Ok(mv)
            }
            Self::PawnCapture { src, dst, promote } => {
                if matches!(dst.rank(), Rank::R1 | Rank::R8) {
                    return Err(IntoMoveError::Create(CreateError::NotWellFormed));
                }
                if b.get(dst).is_empty() {
                    return Err(IntoMoveError::CaptureExpected);
                }
                let src =
                    Coord::from_parts(src, dst.rank()).add(-geometry::pawn_forward_delta(b.side()));
                let mv = base::Move::new(
                    promote
                        .map(MoveKind::from_promote)
                        .unwrap_or(MoveKind::PawnSimple),
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
                if is_capture && b.get(dst).is_empty() {
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
                let c = match piece {
                    Piece::Pawn => panic!("cannot store pawn move as Move::Simple"),
                    Piece::Knight => 'N',
                    Piece::Bishop => 'B',
                    Piece::Rook => 'R',
                    Piece::Queen => 'Q',
                    Piece::King => 'K',
                };
                write!(f, "{}", c)?;
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

impl<'a> fmt::Display for PrettyData<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        self.0.do_fmt::<PrettyTheme>(f)
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
            let (file, bytes) = match bytes.first() {
                Some(b @ b'a'..=b'f') => (File::from_char(*b as char), &bytes[1..]),
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
            let dst = Coord::from_str(str::from_utf8(bytes).unwrap())?;
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
                    b'N' => PromoteKind::Knight,
                    b'B' => PromoteKind::Bishop,
                    b'R' => PromoteKind::Rook,
                    b'Q' => PromoteKind::Queen,
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
                if !matches!(bytes[0], b':' | b'x') || !matches!(bytes[1], b'a'..=b'h') {
                    return Err(RawParseError::Syntax);
                }
                Ok(Data::PawnCapture {
                    src: File::from_char(bytes[1] as char).unwrap(),
                    dst,
                    promote,
                })
            }
            _ => Err(RawParseError::PawnMoveTooLong),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum CheckMark {
    Single,
    Double,
    Checkmate,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Move {
    pub data: Data,
    pub check: Option<CheckMark>,
}

pub struct PrettyMove<'a>(&'a Move);

impl Move {
    pub fn pretty(&self) -> PrettyMove<'_> {
        PrettyMove(self)
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

    pub fn into_move(self, b: &Board) -> Result<base::Move, IntoMoveError> {
        self.data.into_move(b)
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        self.do_fmt::<AlgebraicTheme>(f)
    }
}

impl<'a> fmt::Display for PrettyMove<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        self.0.do_fmt::<PrettyTheme>(f)
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

// TODO implement san_candidates in movegen
// TODO tests
