use super::base::PromoteKind;
use super::uci;
use crate::types::{CastlingSide, Coord, CoordParseError, File, Piece, Rank};

use std::fmt;
use std::marker::PhantomData;
use std::str::{self, FromStr};

use thiserror::Error;

#[derive(Debug, Copy, Clone, Error, Eq, PartialEq)]
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

impl Data {
    pub fn pretty(&self) -> PrettyData<'_> {
        PrettyData(self)
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

// TODO convert moves::Move to san::Move
// TODO convert san::Move to moves::Move
// TODO tests
