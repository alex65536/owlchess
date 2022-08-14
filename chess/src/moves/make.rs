// TODO doc
// TODO tests

use super::base::{self, Move, RawUndo, ValidateError};
use super::{san, uci};
use crate::board::Board;

use core::convert::Infallible;

pub trait Make {
    type Err;

    fn make_raw(&self, board: &mut Board) -> Result<(Move, RawUndo), Self::Err>;

    fn make(&self, board: &Board) -> Result<Board, Self::Err> {
        let mut cloned = board.clone();
        let _ = self.make_raw(&mut cloned)?;
        Ok(cloned)
    }
}

pub struct Unchecked(Move);

impl Unchecked {
    #[inline]
    pub unsafe fn new(mv: Move) -> Self {
        Self(mv)
    }
}

impl Make for Unchecked {
    type Err = Infallible;

    #[inline]
    fn make_raw(&self, board: &mut Board) -> Result<(Move, RawUndo), Self::Err> {
        let undo = unsafe { base::make_move_unchecked(board, self.0) };
        Ok((self.0, undo))
    }
}

pub struct TryUnchecked(Move);

impl TryUnchecked {
    #[inline]
    pub unsafe fn new(mv: Move) -> Self {
        Self(mv)
    }
}

impl Make for TryUnchecked {
    type Err = ValidateError;

    #[inline]
    fn make_raw(&self, board: &mut Board) -> Result<(Move, RawUndo), Self::Err> {
        let undo = unsafe { base::make_move_unchecked(board, self.0) };
        if board.is_opponent_king_attacked() {
            unsafe { base::unmake_move_unchecked(board, self.0, undo) };
            return Err(ValidateError::NotLegal);
        }
        Ok((self.0, undo))
    }

    #[inline]
    fn make(&self, board: &Board) -> Result<Board, Self::Err> {
        let mut cloned = board.clone();
        let _ = unsafe { base::make_move_unchecked(&mut cloned, self.0) };
        if cloned.is_opponent_king_attacked() {
            return Err(ValidateError::NotLegal);
        }
        Ok(cloned)
    }
}

impl Make for Move {
    type Err = ValidateError;

    #[inline]
    fn make_raw(&self, board: &mut Board) -> Result<(Move, RawUndo), Self::Err> {
        base::semi_validate(board, *self)?;
        unsafe { TryUnchecked::new(*self) }.make_raw(board)
    }

    #[inline]
    fn make(&self, board: &Board) -> Result<Board, Self::Err> {
        base::semi_validate(board, *self)?;
        unsafe { TryUnchecked::new(*self) }.make(board)
    }
}

pub struct Uci<S: AsRef<str>>(pub S);

impl<S: AsRef<str>> Make for Uci<S> {
    type Err = uci::ParseError;

    #[inline]
    fn make_raw(&self, board: &mut Board) -> Result<(Move, RawUndo), Self::Err> {
        let mv = Move::from_uci_semilegal(self.0.as_ref(), board)?;
        unsafe { TryUnchecked::new(mv) }
            .make_raw(board)
            .map_err(uci::ParseError::Validate)
    }

    #[inline]
    fn make(&self, board: &Board) -> Result<Board, Self::Err> {
        let mv = Move::from_uci_semilegal(self.0.as_ref(), board)?;
        unsafe { TryUnchecked::new(mv) }
            .make(board)
            .map_err(uci::ParseError::Validate)
    }
}

pub struct San<S: AsRef<str>>(pub S);

impl<S: AsRef<str>> Make for San<S> {
    type Err = san::ParseError;

    #[inline]
    fn make_raw(&self, board: &mut Board) -> Result<(Move, RawUndo), Self::Err> {
        let mv = Move::from_san(self.0.as_ref(), board)?;
        let undo = unsafe { base::make_move_unchecked(board, mv) };
        Ok((mv, undo))
    }
}
