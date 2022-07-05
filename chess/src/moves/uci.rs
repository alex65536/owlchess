//! Utilities to work with moves in UCI format

use super::base::{self, CreateError, MoveKind, PromotePiece, ValidateError};
use crate::board::Board;
use crate::types::{Cell, Color, Coord, CoordParseError, File, Piece};
use crate::{generic, geometry};

use std::fmt;
use std::str::FromStr;

use thiserror::Error;

/// Error creating a parsed UCI representation from string
#[derive(Debug, Clone, Error, Eq, PartialEq)]
pub enum RawParseError {
    /// Bad string length
    #[error("bad string length")]
    BadLength,
    /// Bad source square
    #[error("bad source: {0}")]
    BadSrc(CoordParseError),
    /// Bad destination square
    #[error("bad destination: {0}")]
    BadDst(CoordParseError),
    /// Bad promote character
    #[error("bad promote char {0:?}")]
    BadPromote(char),
}

/// Error parsing UCI into a well-formed [`moves::Move`](super::Move)
#[derive(Debug, Clone, Error, Eq, PartialEq)]
pub enum BasicParseError {
    /// Error parsing move
    #[error("cannot parse move: {0}")]
    Parse(#[from] RawParseError),
    /// Error converting the parsed move into a well-formed move
    #[error("cannot create move: {0}")]
    Create(#[from] CreateError),
}

/// Error parsing UCI into a semilegal or legal [`moves::Move`](super::Move)
#[derive(Debug, Clone, Error, Eq, PartialEq)]
pub enum ParseError {
    /// Error parsing move
    #[error("cannot parse move: {0}")]
    Parse(#[from] RawParseError),
    /// Error converting the parsed move into a well-formed move
    #[error("cannot create move: {0}")]
    Create(#[from] CreateError),
    /// Move is not semilegal or legal
    #[error("invalid move: {0}")]
    Validate(#[from] ValidateError),
}

/// Parsed move in UCI format
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Move {
    /// Null UCI move
    Null,
    /// Non-null UCI move
    Move {
        /// Source square
        src: Coord,
        /// Destination square
        dst: Coord,
        /// Piece to promote, if any
        promote: Option<PromotePiece>,
    },
}

impl Move {
    fn do_into_move<C: generic::Color>(&self, b: &Board) -> Result<base::Move, CreateError> {
        match *self {
            Move::Null => Ok(base::Move::NULL),
            Move::Move { src, dst, promote } => {
                let kind = promote.map(MoveKind::from).unwrap_or_else(|| {
                    // Pawn moves
                    if b.get(src) == Cell::from_parts(C::COLOR, Piece::Pawn) {
                        if src.rank() == geometry::double_move_src_rank(C::COLOR)
                            && dst.rank() == geometry::double_move_dst_rank(C::COLOR)
                        {
                            return MoveKind::PawnDouble;
                        }
                        if src.file() != dst.file() && b.get(dst).is_free() {
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

                base::Move::new(kind, src, dst, C::COLOR)
            }
        }
    }

    /// Converts the UCI move into [`moves::Move`](super::Move) in position `b`
    pub fn into_move(self, b: &Board) -> Result<base::Move, CreateError> {
        match b.r.side {
            Color::White => self.do_into_move::<generic::White>(b),
            Color::Black => self.do_into_move::<generic::Black>(b),
        }
    }
}

impl From<base::Move> for Move {
    #[inline]
    fn from(mv: base::Move) -> Move {
        if mv.kind() == MoveKind::Null {
            return Move::Null;
        }
        Move::Move {
            src: mv.src(),
            dst: mv.dst(),
            promote: mv.kind().try_into().ok(),
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
                    Some(PromotePiece::Knight) => write!(f, "n")?,
                    Some(PromotePiece::Bishop) => write!(f, "b")?,
                    Some(PromotePiece::Rook) => write!(f, "r")?,
                    Some(PromotePiece::Queen) => write!(f, "q")?,
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
                b'n' => PromotePiece::Knight,
                b'b' => PromotePiece::Bishop,
                b'r' => PromotePiece::Rook,
                b'q' => PromotePiece::Queen,
                b => return Err(RawParseError::BadPromote(b as char)),
            })
        } else {
            None
        };
        Ok(Move::Move { src, dst, promote })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::types::{Color, Coord, File, Rank};

    #[test]
    fn test_simple() {
        assert_eq!(Move::from_str("0000").unwrap(), Move::Null);
        assert_eq!(
            Move::from_str("0000")
                .unwrap()
                .into_move(&Board::initial())
                .unwrap(),
            base::Move::NULL
        );

        let e2 = Coord::from_parts(File::E, Rank::R2);
        let e4 = Coord::from_parts(File::E, Rank::R4);
        assert_eq!(
            Move::from_str("e2e4").unwrap(),
            Move::Move {
                src: e2,
                dst: e4,
                promote: None
            }
        );
        assert_eq!(
            Move::from_str("e2e4")
                .unwrap()
                .into_move(&Board::initial())
                .unwrap(),
            base::Move::new(MoveKind::PawnDouble, e2, e4, Color::White).unwrap(),
        );
    }
}
