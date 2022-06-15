use crate::bitboard::Bitboard;
use crate::moves::{self, Move};
use crate::types::{
    self, CastlingRights, CastlingSide, Cell, Color, Coord, DrawKind, File, Piece, Rank,
};
use crate::{bitboard_consts, geometry, movegen, zobrist};

use std::fmt::{self, Display};
use std::num::ParseIntError;
use std::str::FromStr;

use thiserror::Error;

#[derive(Debug, Copy, Clone, Error, Eq, PartialEq)]
pub enum ValidateError {
    #[error("invalid enpassant position {0}")]
    InvalidEnpassant(Coord),
    #[error("too many pieces of color {0:?}")]
    TooManyPieces(Color),
    #[error("no king of color {0:?}")]
    NoKing(Color),
    #[error("more than one king of color {0:?}")]
    TooManyKings(Color),
    #[error("invalid pawn position {0}")]
    InvalidPawn(Coord),
    #[error("opponent's king is attacked")]
    OpponentKingAttacked,
}

#[derive(Debug, Copy, Clone, Error, Eq, PartialEq)]
pub enum CellsParseError {
    #[error("too many items in rank {0}")]
    RankOverflow(Rank),
    #[error("not enough items in rank {0}")]
    RankUnderflow(Rank),
    #[error("too many ranks")]
    Overflow,
    #[error("not enough ranks")]
    Underflow,
    #[error("unexpected char {0:?}")]
    UnexpectedChar(char),
}

#[derive(Debug, Clone, Error, Eq, PartialEq)]
pub enum RawFenParseError {
    #[error("non-ASCII data in FEN")]
    NonAscii,
    #[error("board not specified")]
    NoBoard,
    #[error("bad board: {0}")]
    Board(#[from] CellsParseError),
    #[error("no move side")]
    NoMoveSide,
    #[error("bad move side: {0}")]
    MoveSide(#[from] types::ColorParseError),
    #[error("no castling rights")]
    NoCastling,
    #[error("bad castling rights: {0}")]
    Castling(#[from] types::CastlingRightsParseError),
    #[error("no enpassant")]
    NoEnpassant,
    #[error("bad enpassant: {0}")]
    Enpassant(#[from] types::CoordParseError),
    #[error("invalid enpassant rank {0}")]
    InvalidEnpassantRank(Rank),
    #[error("bad move counter: {0}")]
    MoveCounter(ParseIntError),
    #[error("bad move number: {0}")]
    MoveNumber(ParseIntError),
    #[error("extra data in FEN")]
    ExtraData,
}

#[derive(Debug, Clone, Error, Eq, PartialEq)]
pub enum FenParseError {
    #[error("cannot parse fen: {0}")]
    Fen(#[from] RawFenParseError),
    #[error("invalid position: {0}")]
    Valid(#[from] ValidateError),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct RawBoard {
    pub cells: [Cell; 64],
    pub side: Color,
    pub castling: CastlingRights,
    pub enpassant: Option<Coord>,
    pub move_counter: u16,
    pub move_number: u16,
}

impl RawBoard {
    pub fn empty() -> RawBoard {
        RawBoard {
            cells: [Cell::EMPTY; 64],
            side: Color::White,
            castling: CastlingRights::EMPTY,
            enpassant: None,
            move_counter: 0,
            move_number: 1,
        }
    }

    pub fn initial() -> RawBoard {
        let mut res = RawBoard {
            cells: [Cell::EMPTY; 64],
            side: Color::White,
            castling: CastlingRights::FULL,
            enpassant: None,
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

    pub fn try_from_fen(fen: &str) -> Result<RawBoard, RawFenParseError> {
        RawBoard::from_str(fen)
    }

    pub fn get(&self, c: Coord) -> Cell {
        unsafe { *self.cells.get_unchecked(c.index()) }
    }

    pub fn get2(&self, file: File, rank: Rank) -> Cell {
        self.get(Coord::from_parts(file, rank))
    }

    pub fn put(&mut self, c: Coord, cell: Cell) {
        unsafe {
            *self.cells.get_unchecked_mut(c.index()) = cell;
        }
    }

    pub fn put2(&mut self, file: File, rank: Rank, cell: Cell) {
        self.put(Coord::from_parts(file, rank), cell);
    }

    pub fn zobrist_hash(&self) -> u64 {
        let mut hash = if self.side == Color::White {
            zobrist::MOVE_SIDE
        } else {
            0
        };
        if let Some(p) = self.enpassant {
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

    pub fn pretty(&self, style: PrettyStyle) -> Pretty<'_> {
        Pretty { raw: self, style }
    }

    pub fn as_fen(&self) -> String {
        self.to_string()
    }
}

impl Default for RawBoard {
    fn default() -> RawBoard {
        RawBoard::empty()
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Board {
    pub(crate) r: RawBoard,
    pub(crate) hash: u64,
    pub(crate) white: Bitboard,
    pub(crate) black: Bitboard,
    pub(crate) all: Bitboard,
    pub(crate) pieces: [Bitboard; Cell::MAX_INDEX],
}

impl Board {
    pub fn initial() -> Board {
        RawBoard::initial().try_into().unwrap()
    }

    pub fn try_from_fen(fen: &str) -> Result<Board, FenParseError> {
        Board::from_str(fen)
    }

    pub fn raw(&self) -> &RawBoard {
        &self.r
    }

    pub fn get(&self, c: Coord) -> Cell {
        self.r.get(c)
    }

    pub fn get2(&self, file: File, rank: Rank) -> Cell {
        self.r.get2(file, rank)
    }

    pub fn color(&self, c: Color) -> &Bitboard {
        if c == Color::White {
            &self.white
        } else {
            &self.black
        }
    }

    pub(crate) fn color_mut(&mut self, c: Color) -> &mut Bitboard {
        if c == Color::White {
            &mut self.white
        } else {
            &mut self.black
        }
    }

    pub fn piece(&self, c: Cell) -> Bitboard {
        unsafe { *self.pieces.get_unchecked(c.index()) }
    }

    pub fn piece2(&self, c: Color, p: Piece) -> Bitboard {
        self.piece(Cell::from_parts(c, p))
    }

    pub(crate) fn piece_mut(&mut self, c: Cell) -> &mut Bitboard {
        unsafe { self.pieces.get_unchecked_mut(c.index()) }
    }

    pub fn king_pos(&self, c: Color) -> Coord {
        self.piece(Cell::from_parts(c, Piece::King))
            .into_iter()
            .next()
            .unwrap()
    }

    pub fn zobrist_hash(&self) -> u64 {
        self.hash
    }

    pub fn make_move_weak(&self, mv: Move) -> Result<Self, moves::ValidateError> {
        moves::make_move_weak(self, mv)
    }

    pub fn make_move(&self, mv: Move) -> Result<Self, moves::ValidateError> {
        moves::make_move(self, mv)
    }

    pub fn is_opponent_king_attacked(&self) -> bool {
        let c = self.r.side;
        movegen::is_cell_attacked(self, self.king_pos(c.inv()), c)
    }

    pub fn is_check(&self) -> bool {
        let c = self.r.side;
        movegen::is_cell_attacked(self, self.king_pos(c), c.inv())
    }

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
        if all_without_kings == knights && knights.popcount() == 1 {
            return true;
        }

        // Kings and bishops of the same cell color. Note that we checked above that all the pieces
        // have the same cell color, so we just need to ensure that all the pieces are bishops
        let bishops =
            self.piece2(Color::White, Piece::Bishop) | self.piece2(Color::Black, Piece::Bishop);
        if all_without_kings == bishops {
            return true;
        }

        false
    }

    pub fn is_draw_simple(&self, with_proposed: bool) -> Option<DrawKind> {
        // Check for insufficient material
        if self.is_insufficient_material() {
            return Some(DrawKind::InsufficientMaterial);
        }

        // Check for 50/75 move rule
        if self.r.move_counter >= 150 {
            return Some(DrawKind::Moves75);
        }
        if with_proposed && self.r.move_counter >= 100 {
            return Some(DrawKind::Moves50);
        }

        None
    }

    pub fn pretty(&self, style: PrettyStyle) -> Pretty<'_> {
        self.r.pretty(style)
    }

    pub fn as_fen(&self) -> String {
        self.to_string()
    }
}

impl TryFrom<RawBoard> for Board {
    type Error = ValidateError;

    fn try_from(mut raw: RawBoard) -> Result<Board, ValidateError> {
        // Check InvalidEnpassant
        if let Some(p) = raw.enpassant {
            if p.rank() != geometry::enpassant_src_rank(raw.side) {
                return Err(ValidateError::InvalidEnpassant(p));
            }
        }

        // Reset enpassant if either there is no pawn or the cell on the pawn's path is occupied
        if let Some(p) = raw.enpassant {
            let pp = p.add(geometry::pawn_forward_delta(raw.side));
            if raw.get(p) != Cell::from_parts(raw.side.inv(), Piece::Pawn)
                || raw.get(pp) != Cell::EMPTY
            {
                raw.enpassant = None;
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
        let mut pieces = [Bitboard::EMPTY; Cell::MAX_INDEX];
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
        if white.popcount() > 16 {
            return Err(ValidateError::TooManyPieces(Color::White));
        }
        if black.popcount() > 16 {
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
        if white_king.popcount() > 1 {
            return Err(ValidateError::TooManyKings(Color::White));
        }
        if black_king.popcount() > 1 {
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

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum PrettyStyle {
    Ascii,
    Utf8,
}

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

fn parse_enpassant(s: &str, side: Color) -> Result<Option<Coord>, RawFenParseError> {
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
        let enpassant = parse_enpassant(iter.next().ok_or(Error::NoEnpassant)?, side)?;
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
            enpassant,
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
            if cell.is_empty() {
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
        match self.enpassant {
            Some(p) => write!(
                f,
                " {}",
                Coord::from_parts(p.file(), geometry::enpassant_dst_rank(self.side))
            )?,
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
    use std::mem;

    #[test]
    fn test_size() {
        assert_eq!(mem::size_of::<RawBoard>(), 72);
        assert_eq!(mem::size_of::<Board>(), 208);
    }

    #[test]
    fn test_initial() {
        const INI_FEN: &'static str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

        assert_eq!(RawBoard::initial().to_string(), INI_FEN);
        assert_eq!(Board::initial().to_string(), INI_FEN);
        assert_eq!(RawBoard::from_str(INI_FEN), Ok(RawBoard::initial()));
        assert_eq!(Board::from_str(INI_FEN), Ok(Board::initial()));
    }

    #[test]
    fn test_midgame() {
        const FEN: &'static str =
            "1rq1r1k1/1p3ppp/pB3n2/3ppP2/Pbb1P3/1PN2B2/2P2QPP/R1R4K w - - 1 21";

        let board = Board::try_from_fen(FEN).unwrap();
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
        assert_eq!(board.raw().enpassant, None);
        assert_eq!(board.raw().move_counter, 1);
        assert_eq!(board.raw().move_number, 21);
    }

    #[test]
    fn test_fixes() {
        const FEN: &'static str =
            "r1bq1b1r/ppppkppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK1R1 w KQkq c6 6 5";

        let raw = RawBoard::try_from_fen(FEN).unwrap();
        assert_eq!(raw.castling, CastlingRights::FULL);
        assert_eq!(raw.enpassant, Some(Coord::from_parts(File::C, Rank::R5)));
        assert_eq!(raw.as_fen(), FEN);

        let board: Board = raw.try_into().unwrap();
        assert_eq!(
            board.raw().castling,
            CastlingRights::EMPTY.with(Color::White, CastlingSide::Queen)
        );
        assert_eq!(board.raw().enpassant, None);
        assert_eq!(
            board.as_fen(),
            "r1bq1b1r/ppppkppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK1R1 w Q - 6 5"
        );
    }

    #[test]
    fn test_incomplete() {
        assert_eq!(
            RawBoard::try_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"),
            Err(RawFenParseError::NoMoveSide)
        );

        assert_eq!(
            RawBoard::try_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w"),
            Err(RawFenParseError::NoCastling)
        );

        assert_eq!(
            RawBoard::try_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq"),
            Err(RawFenParseError::NoEnpassant)
        );

        let raw =
            RawBoard::try_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -").unwrap();
        assert_eq!(raw.move_counter, 0);
        assert_eq!(raw.move_number, 1);

        let raw = RawBoard::try_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 10")
            .unwrap();
        assert_eq!(raw.move_counter, 10);
        assert_eq!(raw.move_number, 1);
    }
}
