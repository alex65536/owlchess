use crate::types::{Coord, Piece, Cell, File, Color, CoordParseError};
use crate::board::Board;
use super::moves::{self, CreateError, MoveKind, PromoteKind, ValidateError};
use crate::{generic, geometry};

use std::fmt;
use std::str::FromStr;

use thiserror::Error;

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

pub enum Move {
    Null,
    Move {
        src: Coord,
        dst: Coord,
        promote: Option<PromoteKind>,
    },
}

impl Move {
    fn do_into_move<C: generic::Color>(&self, b: &Board) -> Result<moves::Move, CreateError> {
        match *self {
            Move::Null => Ok(moves::Move::NULL),
            Move::Move { src, dst, promote } => {
                let kind = promote.map(MoveKind::promote).unwrap_or_else(|| {
                    // Pawn moves
                    if b.get(src) == Cell::from_parts(C::COLOR, Piece::Pawn) {
                        if src.rank() == geometry::double_move_src_rank(C::COLOR)
                            && dst.rank() == geometry::double_move_dst_rank(C::COLOR)
                        {
                            return MoveKind::PawnDouble;
                        }
                        if src.file() != dst.file() && b.get(dst).is_empty() {
                            return MoveKind::Enpassant;
                        }
                        return MoveKind::PawnSimple;
                    }

                    // Castling
                    if b.get(src) == Cell::from_parts(C::COLOR, Piece::King) {
                        let rank = geometry::castling_rank(C::COLOR);
                        if src == Coord::from_parts(File::E, rank) {
                            if dst == Coord::from_parts(File::G, rank) {
                                return MoveKind::CastlingKingside;
                            }
                            if dst == Coord::from_parts(File::C, rank) {
                                return MoveKind::CastlingQueenside;
                            }
                        }
                    }

                    MoveKind::Simple
                });

                moves::Move::new(kind, src, dst, C::COLOR)
            }
        }
    }

    pub fn into_move(self, b: &Board) -> Result<moves::Move, CreateError> {
        match b.r.side {
            Color::White => self.do_into_move::<generic::White>(b),
            Color::Black => self.do_into_move::<generic::Black>(b),
        }
    }
}

impl From<moves::Move> for Move {
    fn from(mv: moves::Move) -> Move {
        if mv.kind() == MoveKind::Null {
            return Move::Null;
        }
        Move::Move {
            src: mv.src(),
            dst: mv.dst(),
            promote: mv.kind().promote_to(),
        }
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match *self {
            Move::Null => write!(f, "0000"),
            Move::Move { src, dst, promote } => {
                write!(f, "{}{}", src, dst)?;
                match promote {
                    Some(PromoteKind::Knight) => write!(f, "n")?,
                    Some(PromoteKind::Bishop) => write!(f, "b")?,
                    Some(PromoteKind::Rook) => write!(f, "r")?,
                    Some(PromoteKind::Queen) => write!(f, "q")?,
                    None => {}
                };
                Ok(())
            }
        }
    }
}

impl FromStr for Move {
    type Err = RawParseError;

    fn from_str(s: &str) -> Result<Move, Self::Err> {
        if s == "0000" {
            return Ok(Move::Null);
        }
        if !matches!(s.len(), 4 | 5) {
            return Err(RawParseError::BadLength);
        }
        let src = Coord::from_str(&s[0..2]).map_err(RawParseError::BadSrc)?;
        let dst = Coord::from_str(&s[2..4]).map_err(RawParseError::BadDst)?;
        let promote = if s.len() == 5 {
            Some(match s.as_bytes()[4] {
                b'n' => PromoteKind::Knight,
                b'b' => PromoteKind::Bishop,
                b'r' => PromoteKind::Rook,
                b'q' => PromoteKind::Queen,
                b => return Err(RawParseError::BadPromote(b as char)),
            })
        } else {
            None
        };
        Ok(Move::Move { src, dst, promote })
    }
}
