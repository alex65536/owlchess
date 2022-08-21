//! Types to apply the moves and move-like objects
//!
//! This module contains a trait [`Make`] and its implementations. Each implementation of this
//! trait can be applied as a move.

use super::base::{self, Move, RawUndo, ValidateError};
use super::{san, uci};
use crate::board::Board;

use core::convert::Infallible;

/// Trait for move-like objects
///
/// Each implementation of this trait can be applied as a chess move.
///
/// # Safety
///
/// The implementation of [`Make::make_raw()`] must return a valid pair of [`Move`] and [`RawUndo`],
/// which can be safety passed to [`moves::unmake_move_unchecked`](base::unmake_move_unchecked).
/// Also, the implementations of [`Make::make_raw()`] and [`Make::make()`] must be consistent with
/// respect to each other.
pub unsafe trait Make {
    type Err;

    /// Applies the move to the board `board`
    ///
    /// In case of success, a tuple is returned. It contains an applied move and a [`RawUndo`],
    /// which can be passed to [`moves::unmake_move_unchecked`](base::unmake_move_unchecked) to
    /// revert this move. In case of error, the board is unchanged.
    ///
    /// If the move is not legal, then an error is returned. Some implementations (like [`Unchecked`]
    /// or [`TryUnchecked`]) also allow making null moves.
    fn make_raw(&self, board: &mut Board) -> Result<(Move, RawUndo), Self::Err>;

    /// Applies the move to the board `board`
    ///
    /// Returns the new board after applying the given move.
    ///
    /// If the move is not legal, then an error is returned. Some implementations (like [`Unchecked`]
    /// or [`TryUnchecked`]) also allow making null moves.
    fn make(&self, board: &Board) -> Result<Board, Self::Err> {
        let mut cloned = board.clone();
        let _ = self.make_raw(&mut cloned)?;
        Ok(cloned)
    }
}

/// Wrapper to make a move without validity checks
///
/// # Safety
///
/// See the doc for [`Unchecked::new()`].
pub struct Unchecked(Move);

impl Unchecked {
    /// Wraps a move `mv` into [`Unchecked`]
    ///
    /// # Safety
    ///
    /// You must guarantee that this move will be applied only to positions where it is legal or null,
    /// and null moves are allowed only if the king is not is check. Applying this move to other
    /// positions is considered undefined behaviour.
    #[inline]
    pub unsafe fn new(mv: Move) -> Self {
        Self(mv)
    }
}

unsafe impl Make for Unchecked {
    type Err = Infallible;

    #[inline]
    fn make_raw(&self, board: &mut Board) -> Result<(Move, RawUndo), Self::Err> {
        let undo = unsafe { base::make_move_unchecked(board, self.0) };
        Ok((self.0, undo))
    }
}

/// Wrapper a semilegal or null move that bypasses semi-legality checks
///
/// This wrapper makes a semilegal move and checks whether this move is legal after making it. No
/// other validity checks are performed. So, it will return an error is the move is semilegal but not
/// legal. If the move is null, an error is returned if and only if the king is in check.
///
/// # Safety
///
/// See the doc for [`TryUnchecked::new()`].
pub struct TryUnchecked(Move);

impl TryUnchecked {
    /// Wraps a move `mv` into [`Unchecked`]
    ///
    /// # Safety
    ///
    /// You must guarantee that this move will be applied only to positions where it is legal or null,
    /// otherwise the behavior is undefined.
    #[inline]
    pub unsafe fn new(mv: Move) -> Self {
        Self(mv)
    }
}

unsafe impl Make for TryUnchecked {
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

unsafe impl Make for Move {
    type Err = ValidateError;

    #[inline]
    fn make_raw(&self, board: &mut Board) -> Result<(Move, RawUndo), Self::Err> {
        self.semi_validate(board)?;
        unsafe { TryUnchecked::new(*self) }.make_raw(board)
    }

    #[inline]
    fn make(&self, board: &Board) -> Result<Board, Self::Err> {
        self.semi_validate(board)?;
        unsafe { TryUnchecked::new(*self) }.make(board)
    }
}

unsafe impl Make for uci::Move {
    type Err = uci::ParseError;

    #[inline]
    fn make_raw(&self, board: &mut Board) -> Result<(Move, RawUndo), Self::Err> {
        Ok(self.into_move(board)?.make_raw(board)?)
    }

    #[inline]
    fn make(&self, board: &Board) -> Result<Board, Self::Err> {
        Ok(self.into_move(board)?.make(board)?)
    }
}

unsafe impl Make for san::Move {
    type Err = san::IntoMoveError;

    #[inline]
    fn make_raw(&self, board: &mut Board) -> Result<(Move, RawUndo), Self::Err> {
        let mv = self.into_move(board)?;
        let undo = unsafe { base::make_move_unchecked(board, mv) };
        Ok((mv, undo))
    }
}

/// Wrapper to apply a move directly from its UCI notation
pub struct Uci<S: AsRef<str>>(pub S);

unsafe impl<S: AsRef<str>> Make for Uci<S> {
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

/// Wrapper to apply a move directly from its SAN notation
pub struct San<S: AsRef<str>>(pub S);

unsafe impl<S: AsRef<str>> Make for San<S> {
    type Err = san::ParseError;

    #[inline]
    fn make_raw(&self, board: &mut Board) -> Result<(Move, RawUndo), Self::Err> {
        let mv = Move::from_san(self.0.as_ref(), board)?;
        let undo = unsafe { base::make_move_unchecked(board, mv) };
        Ok((mv, undo))
    }
}
