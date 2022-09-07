//! Board that remembers previous moves
//!
//! Sometimes, using [`Board`](crate::board::Board) is not very convenient. The possible
//! reasons are as follows:
//!
//! - you cannot undo moves in a fast manner. You need to either create a new board after
//!   each move, or deal with [`RawUndo`](crate::moves::RawUndo) and `unsafe` manually
//! - you cannot detect draw by repetitions
//! - you cannot inspect all the previous moves or get the notation of the entire game
//!
//! To solve all these problems, there is a [`MoveChain`].
//!
//! # Example
//!
//! ```
//! # use owlchess::{Board, MoveChain};
//! # use owlchess::moves::make;
//! #
//! // Create an empty chain from the empty position
//! let mut chain = MoveChain::new_initial();
//!
//! // Push d2d4 move
//! chain.push(make::Uci("d2d4")).unwrap();
//! assert_eq!(
//!     chain.last().to_string(),
//!     "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1".to_string(),
//! );
//!
//! // Undo the last move
//! chain.pop().unwrap();
//! assert_eq!(chain.last(), &Board::initial());
//!
//! // Push other moves
//! chain.push(make::Uci("e2e4")).unwrap();
//! chain.push(make::Uci("e7e5")).unwrap();
//!
//! // Get notation as a UCI move sequence
//! assert_eq!(chain.uci().to_string(), "e2e4 e7e5".to_string());
//! ```

use crate::board::{self, Board, RawBoard};
use crate::moves::{self, make, Make, Move, RawUndo};
use crate::types::{Color, DrawReason, GameStatus, Outcome, OutcomeFilter};

use std::collections::HashMap;
use std::fmt;

use thiserror::Error;

/// Error while parsing and applying UCI move sequence
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[error("cannot parse UCI move #{}: {}", .pos + 1, .source)]
pub struct UciParseError {
    /// The number of moves successfully applied before the error occurred
    pub pos: usize,
    /// The error happened when trying to apply the move
    pub source: moves::uci::ParseError,
}

/// Repetition table trait
///
/// It allows to customize [`MoveChain`] and to implement something better than
/// the built-in [`HashRepeat`] to detect draws by repetitions.
///
/// Technically, a repetition table is just a multiset that stored all the positions
/// occurred during the game.
pub trait Repeat: Default {
    /// Adds a board `b` to the set
    ///
    /// Note that this is a multiset, so there can be multiple instances of the same
    /// board in it.
    fn push(&mut self, b: &Board);

    /// Removes a board `b` from the set
    ///
    /// # Panics
    ///
    /// The function must panic if there is no such board `b`.
    fn pop(&mut self, b: &Board);

    /// Returns the number of times board `b` is present in the set.
    fn count(&self, b: &Board) -> usize;
}

/// Simple repetition table based on [`HashMap`]
#[derive(Default, Debug, Clone, Eq, PartialEq)]
pub struct HashRepeat(HashMap<u64, usize>);

impl Repeat for HashRepeat {
    fn push(&mut self, b: &Board) {
        self.0
            .entry(b.zobrist_hash())
            .and_modify(|x| *x += 1)
            .or_insert(1);
    }

    fn pop(&mut self, b: &Board) {
        let hash = b.zobrist_hash();
        let r = self.0.get_mut(&hash).unwrap();
        *r -= 1;
        if *r == 0 {
            self.0.remove(&hash);
        }
    }

    fn count(&self, b: &Board) -> usize {
        *self.0.get(&b.zobrist_hash()).unwrap_or(&0)
    }
}

/// Convenience instantiation of [`BaseMoveChain`] with default repetition table
pub type MoveChain = BaseMoveChain<HashRepeat>;

/// Board that remembers previous moves
///
/// See the [module docs](crate::chain) for more details.
///
/// This version allows customizing its repetition table. Use [`MoveChain`] if you don't
/// need this.
#[derive(Debug, Clone)]
pub struct BaseMoveChain<R: Repeat> {
    start: RawBoard,
    board: Board,
    repeat: R,
    stack: Vec<(Move, RawUndo)>,
    outcome: Option<Outcome>,
}

impl<R: Repeat> BaseMoveChain<R> {
    /// Creates an empty move chain, starting with position `b`
    pub fn new(b: Board) -> Self {
        let mut res = BaseMoveChain {
            start: b.r,
            board: b,
            repeat: R::default(),
            stack: Vec::new(),
            outcome: None,
        };
        res.repeat.push(&res.board);
        res
    }

    /// Creates an empty move chain, starting with initial position
    #[inline]
    pub fn new_initial() -> Self {
        Self::new(Board::initial())
    }

    /// Creates a move chain starting from position `b` with UCI moves from the
    /// space-separated list `uci_list` applied
    #[inline]
    pub fn from_uci_list(b: Board, uci_list: &str) -> Result<Self, UciParseError> {
        let mut res = BaseMoveChain::new(b);
        res.push_uci_list(uci_list)?;
        Ok(res)
    }

    /// Creates an empty move chain, starting from position with FEN string `fen`
    #[inline]
    pub fn from_fen(fen: &str) -> Result<Self, board::FenParseError> {
        Ok(Self::new(Board::from_fen(fen)?))
    }

    /// Returns the position from which this chain starts
    #[inline]
    pub fn startpos(&self) -> &RawBoard {
        &self.start
    }

    /// Returns the current position
    #[inline]
    pub fn last(&self) -> &Board {
        &self.board
    }

    /// Returns the number of applied moves
    #[inline]
    pub fn len(&self) -> usize {
        self.stack.len()
    }

    /// Returns `true` if there is no applied moves
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    /// Iterates over all the applied moves, from first to last
    ///
    /// If you need previous positions alongside with the moves, you need to use
    /// [`BaseMoveChain::walk()`] instead.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = Move> + '_ {
        self.stack.iter().map(|(m, _)| *m)
    }

    /// Returns the `idx`th applied move
    ///
    /// # Panics
    ///
    /// The function panics if `idx` is out of range.
    #[inline]
    pub fn get(&self, idx: usize) -> Move {
        self.stack[idx].0
    }

    /// Returns the `idx`th applied move
    ///
    /// # Safety
    ///
    /// The behavior is undefined if `idx` is out of range.
    #[inline]
    pub unsafe fn get_unchecked(&self, idx: usize) -> Move {
        self.stack.get_unchecked(idx).0
    }

    /// Returns the game outcome stored in the chain
    #[inline]
    pub fn outcome(&self) -> &Option<Outcome> {
        &self.outcome
    }

    /// Returns `true` if there is a game outcome stored
    #[inline]
    pub fn is_finished(&self) -> bool {
        self.outcome.is_some()
    }

    /// Removes the stored game outcome
    #[inline]
    pub fn clear_outcome(&mut self) {
        self.outcome = None;
    }

    /// Sets the game outcome to `outcome`
    ///
    /// # Panics
    ///
    /// The function panics is the outcome was already set.
    #[inline]
    pub fn set_outcome(&mut self, outcome: Outcome) {
        assert!(!self.is_finished());
        self.outcome = Some(outcome);
    }

    /// Resets the outcome to `outcome`
    ///
    /// If the outcome was already set, then it is just replaced by `outcome`
    #[inline]
    pub fn reset_outcome(&mut self, outcome: Option<Outcome>) {
        self.outcome = outcome;
    }

    /// Calculates the current outcome of the game
    ///
    /// This function ignores the value stored in [`BaseMoveChain::outcome()`] and uses
    /// just current position and repetition table for calculations.
    ///
    /// This function can be computationally expensive, as it calls
    /// [`movegen::has_legal_moves`](crate::movegen::has_legal_moves).
    ///
    /// Unlike [`Board::calc_outcome()`], it is able to detect draw by repetitions.
    ///
    /// For the priority of different outcomes, see the docs for [`Board::calc_outcome()`]
    pub fn calc_outcome(&self) -> Option<Outcome> {
        // We need to handle the priority of different outcomes carefully.
        //
        // Checkmate and stalemate are definitely preferred over all the other
        // outcomes, though they cannot happen at the same time with draw by
        // repetitions.
        //
        // Next, all the strict outcomes must be checked before the non-strict ones.
        // For example, if there is both `DrawReason::Moves50` and `DrawReason::Repeat5`,
        // the latter must be preferred, as it's strict. Still, we have no priority
        // between `DrawReason::Moves75` and `DrawReason::Repeat5` or between
        // `DrawReason::Moves50` and `DrawReason::Repeat3`.
        //
        // So, we can do the following:
        // - first, call `Board::calc_outcome()` and check for strict outcomes
        // - then, check for draw by repetitions
        // - finally, return a non-strict outcome from `Board::calc_outcome()`, if any

        let outcome = self.board.calc_outcome();
        if let Some(out) = outcome {
            if out.passes(OutcomeFilter::Strict) {
                return outcome;
            }
        }

        let rep = self.repeat.count(&self.board);
        if rep >= 5 {
            return Some(Outcome::Draw(DrawReason::Repeat5));
        }
        if rep >= 3 {
            return Some(Outcome::Draw(DrawReason::Repeat3));
        }

        outcome
    }

    /// Calculates the game outcome using [`BaseMoveChain::calc_outcome()`] and sets the calculated
    /// outcome if it passes the filter `filter`
    ///
    /// This function can be computationally expensive, as it calls
    /// [`movegen::has_legal_moves`](crate::movegen::has_legal_moves).
    ///
    /// # Panics
    ///
    /// The function panics is the outcome was already set.
    pub fn set_auto_outcome(&mut self, filter: OutcomeFilter) -> Option<Outcome> {
        assert!(!self.is_finished());
        if let Some(outcome) = self.calc_outcome() {
            if outcome.passes(filter) {
                self.set_outcome(outcome);
            }
        }
        self.outcome
    }

    #[inline]
    fn do_finish_push(&mut self, mv: Move, u: RawUndo) {
        self.repeat.push(&self.board);
        self.stack.push((mv, u));
    }

    /// Pushes a move `mv` to the chain
    ///
    /// # Safety
    ///
    /// The move must be either legal or null, and the move cannot be null if the king is in
    /// check. Also, game outcome must be unset. Otherwise, the behavior is undefined.
    #[inline]
    pub unsafe fn push_unchecked(&mut self, mv: Move) {
        let u = moves::make_move_unchecked(&mut self.board, mv);
        self.do_finish_push(mv, u);
    }

    /// Pushes a move-like object `m` to the chain
    ///
    /// If `m` doesn't represent a legal move, then nothing is pushed, and the error
    /// is returned.
    #[inline]
    pub fn push<M: Make>(&mut self, m: M) -> Result<(), M::Err> {
        assert!(!self.is_finished());
        let (mv, undo) = m.make_raw(&mut self.board)?;
        self.do_finish_push(mv, undo);
        Ok(())
    }

    /// Pushes a space-separated list of moves `uci_list` to the chain
    ///
    /// The string is first split into tokens. Then, the function tries to apply each token as
    /// a UCI move. It stops when either all the moves were applied or an error occurred. So,
    /// in case of errors only a prefix of `uci_list` is pushed.
    ///
    /// To determine how much moves were pushed in case of error, you need to inspect
    /// [`UciParseError`].
    pub fn push_uci_list(&mut self, uci_list: &str) -> Result<(), UciParseError> {
        for (pos, token) in uci_list.split_ascii_whitespace().enumerate() {
            self.push(make::Uci(token))
                .map_err(|source| UciParseError { pos, source })?;
        }
        Ok(())
    }

    /// Removes the last pushed move from the chain and returns the removed move
    ///
    /// If the chain doesn't contain any moves, it remains unchanged and `None` is
    /// returned.
    pub fn pop(&mut self) -> Option<Move> {
        let (m, u) = self.stack.pop()?;
        self.repeat.pop(&self.board);
        self.clear_outcome();
        unsafe { moves::unmake_move_unchecked(&mut self.board, m, u) };
        Some(m)
    }

    /// Returns the wrapper which helps to format the move chain as a space-separated
    /// UCI move list
    ///
    /// The resulting wrapper implements [`fmt::Display`], so can be used with
    /// `write!()`, `println!()`, or `ToString::to_string`.
    ///
    /// # Example
    ///
    /// ```
    /// # use owlchess::{board::Board, chain::MoveChain};
    /// #
    /// let chain = MoveChain::from_uci_list(Board::initial(), "e2e4 e7e5 g1f3").unwrap();
    /// assert_eq!(format!("Your UCI chain is {}", chain.uci()), "Your UCI chain is e2e4 e7e5 g1f3");
    /// ```
    #[inline]
    pub fn uci(&self) -> UciList<'_, R> {
        UciList(self)
    }

    /// Creates a [`Walker`] over the current move chain
    ///
    /// [`Walker`] looks similar to iterator, but is bidirectional and can iterate over moves
    /// alongside with positions preceding these moves.
    ///
    /// The returned walker is initially at the beginning of the move chain.
    #[inline]
    pub fn walk(&self) -> Walker<'_> {
        Walker {
            board: self.board.clone(),
            stack: &self.stack,
            pos: 0,
            board_pos: self.stack.len(),
        }
    }

    /// Returns the wrapper which helps to format the move chain as a styled move list
    ///
    /// The resulting wrapper implements [`fmt::Display`], so can be used with
    /// `write!()`, `println!()`, or `ToString::to_string`.
    ///
    /// The formatting is customizable. `nums` indicate whether and how move numbers must
    /// be formatted, `style` indicates the move formatting style, and `status` indicates
    /// whether game status must be placed in the end of the list.
    ///
    /// # Example
    ///
    /// ```
    /// # use owlchess::{
    /// #     board::Board,
    /// #     chain::{MoveChain, NumberPolicy, GameStatusPolicy},
    /// #     moves::Style,
    /// #     types::OutcomeFilter,
    /// # };
    /// #
    /// let mut chain = MoveChain::from_uci_list(
    ///     Board::initial(),
    ///     "e2e4 e7e5 f1c4 b8c6 d1h5 g8f6 h5f7",
    /// ).unwrap();
    /// chain.set_auto_outcome(OutcomeFilter::Strict).unwrap();
    ///
    /// let formatted = chain.styled(
    ///     NumberPolicy::FromBoard,
    ///     Style::San,
    ///     GameStatusPolicy::Show
    /// ).to_string();
    ///
    /// assert_eq!(formatted, "1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0".to_string());
    /// ```
    #[inline]
    pub fn styled(
        &self,
        nums: NumberPolicy,
        style: moves::Style,
        status: GameStatusPolicy,
    ) -> StyledList<'_, R> {
        StyledList {
            inner: self,
            nums,
            style,
            status,
        }
    }
}

impl<R: Repeat> Default for BaseMoveChain<R> {
    #[inline]
    fn default() -> Self {
        Self::new_initial()
    }
}

impl<R: Repeat + Eq> PartialEq<Self> for BaseMoveChain<R> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.board != other.board
            || self.repeat != other.repeat
            || self.stack.len() != other.stack.len()
        {
            return false;
        }
        self.stack
            .iter()
            .zip(other.stack.iter())
            .all(|((m1, _), (m2, _))| m1 == m2)
    }
}

impl<R: Repeat + Eq> Eq for BaseMoveChain<R> {}

/// Wrapper that helps to format [`BaseMoveChain`] as a UCI move list
///
/// See [`BaseMoveChain::uci()`] for more details.
pub struct UciList<'a, R: Repeat>(&'a BaseMoveChain<R>);

impl<'a, R: Repeat> fmt::Display for UciList<'a, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        for (i, m) in self.0.iter().enumerate() {
            if i != 0 {
                write!(f, " ")?;
            }
            write!(f, "{}", m)?;
        }
        Ok(())
    }
}

/// Walks over the move chain, iterating over moves with their preceding positions
///
/// Walker works similar to bidirectional iterator, but doesn't implement [`Iterator`]
/// in fact. The reason is that [`Walker::next()`] returns returns reference to [`Board`]
/// which can live only before the iterator is changed. But the [`Iterator`] trait requires
/// that the references to items returned by `next()` must live at least as long as the
/// iterator itself.
///
/// Walker can be considered a stack, where [`next()`][Walker::next] pushes the next move
/// from the move chain, and [`prev()`][Walker::prev()] pops the last pushed move.
///
/// To create a walker, one needs to use [`BaseMoveChain::walk()`].
///
/// # Example
///
/// ```
/// # use owlchess::{Board, MoveChain, chain::Walker};
/// #
/// let chain = MoveChain::from_uci_list(Board::initial(), "e2e4 e7e5 g1f3").unwrap();
/// let mut walker = chain.walk();
///
/// // Push two moves into Walker
/// assert_eq!(walker.next().unwrap().1.to_string(), "e2e4".to_string());
/// assert_eq!(walker.next().unwrap().1.to_string(), "e7e5".to_string());
///
/// // Then, pop one of them (i.e. "e7e5")
/// assert_eq!(walker.prev().unwrap().1.to_string(), "e7e5".to_string());
///
/// // Push the rest of the moves
/// assert_eq!(walker.next().unwrap().1.to_string(), "e7e5".to_string());
/// assert_eq!(walker.next().unwrap().1.to_string(), "g1f3".to_string());
///
/// // We are at the end
/// assert_eq!(walker.next(), None);
/// ```
pub struct Walker<'a> {
    board: Board,
    stack: &'a [(Move, RawUndo)],
    pos: usize,
    board_pos: usize,
}

impl<'a> Walker<'a> {
    /// Returns the number of moves in the underlying move chain
    #[inline]
    pub fn len(&self) -> usize {
        self.stack.len()
    }

    /// Returns `true` if the underlying move chain has no moves
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    /// Returns the number of moves pushed into the walker
    ///
    /// This is equal to the move index that will be fetched by the next call of
    /// [`Walker::next()`].
    #[inline]
    pub fn pos(&self) -> usize {
        self.pos
    }

    fn set_board_pos(&mut self, target: usize) {
        while self.board_pos > target {
            self.board_pos -= 1;
            let (mv, u) = self.stack[self.board_pos];
            unsafe {
                moves::unmake_move_unchecked(&mut self.board, mv, u);
            }
        }
        while self.board_pos < target {
            let (mv, _) = self.stack[self.board_pos];
            unsafe {
                moves::make_move_unchecked(&mut self.board, mv);
            }
            self.board_pos += 1;
        }
    }

    /// Tries to push the next move from chain into the walker
    ///
    /// If there are no moves to push (i. e. the walker is at the end of
    /// the chain), then `None` is returned.
    ///
    /// Otherwise, the pushed move is returned, alongside with the position
    /// immediately preceding this move.
    ///
    /// Note that the reference to [`Board`] is only active until the walker
    /// is mutated.
    #[allow(clippy::should_implement_trait)] // cannot implement Iterator
    pub fn next(&mut self) -> Option<(&Board, Move)> {
        if self.pos == self.stack.len() {
            return None;
        }
        self.pos += 1;
        self.set_board_pos(self.pos - 1);
        Some((&self.board, self.stack[self.pos - 1].0))
    }

    /// Tries to pop the last move from the walker
    ///
    /// If there are no moves to pop (i. e. the walker is at the beginning
    /// of the chain), then `None` is returned.
    ///
    /// Otherwise, the popped move is returned, alongside with the position
    /// immediately preceding this move.
    ///
    /// Note that the reference to [`Board`] is only active until the walker
    /// is mutated.
    pub fn prev(&mut self) -> Option<(&Board, Move)> {
        if self.pos == 0 {
            return None;
        }
        self.pos -= 1;
        self.set_board_pos(self.pos);
        Some((&self.board, self.stack[self.pos].0))
    }

    /// Pops all the moves from the walker
    ///
    /// In other words, the walker moves to the start of the chain.
    #[inline]
    pub fn start(&mut self) {
        self.pos = 0;
    }

    /// Pushes all the moves to the walker
    ///
    /// In other words, the walker moves to the end of the chain.
    #[inline]
    pub fn end(&mut self) {
        self.pos = self.stack.len();
    }
}

/// Indicates whether to show move numbers in [`BaseMoveChain::styled()`]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum NumberPolicy {
    /// Do not show move numbers
    Omit,
    /// Show move numbers according to the number specified in the board
    FromBoard,
    /// Show move numbers, but start from the given number instead of the one
    /// specified in the board
    Custom(usize),
}

/// Indicates whether to show game status in [`BaseMoveChain::styled()`]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum GameStatusPolicy {
    /// Show game status
    Show,
    /// Hide game status
    Hide,
}

/// Wrapper which helps to format the move chain as a styled move list
///
/// See [`BaseMoveChain::styled()`] doc for more details.
pub struct StyledList<'a, R: Repeat> {
    inner: &'a BaseMoveChain<R>,
    nums: NumberPolicy,
    style: moves::Style,
    status: GameStatusPolicy,
}

impl<'a, R: Repeat> fmt::Display for StyledList<'a, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        if self.inner.is_empty() {
            match self.status {
                GameStatusPolicy::Show => write!(f, "{}", GameStatus::from(*self.inner.outcome()))?,
                GameStatusPolicy::Hide => {}
            }
            return Ok(());
        }

        let mut walker = self.inner.walk();
        let (b, mv) = walker.next().unwrap();
        let real_start_num = b.raw().move_number as usize;
        let start_num = match self.nums {
            NumberPolicy::Omit => None,
            NumberPolicy::FromBoard => Some(real_start_num),
            NumberPolicy::Custom(u) => Some(u),
        };

        if let Some(num) = start_num {
            match b.side() {
                Color::White => write!(f, "{}. ", num)?,
                Color::Black => write!(f, "{}... ", num)?,
            }
        }
        write!(f, "{}", mv.styled(b, self.style).unwrap())?;

        while let Some((b, mv)) = walker.next() {
            if let Some(num) = start_num {
                if b.side() == Color::White {
                    write!(
                        f,
                        " {}.",
                        b.raw().move_number as usize - real_start_num + num
                    )?;
                }
            }
            write!(f, " {}", mv.styled(b, self.style).unwrap())?;
        }

        match self.status {
            GameStatusPolicy::Show => write!(f, " {}", GameStatus::from(*self.inner.outcome()))?,
            GameStatusPolicy::Hide => {}
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::moves::make;
    use crate::types::{DrawReason, Outcome, OutcomeFilter, WinReason};

    #[test]
    fn test_simple() {
        let mut chain =
            MoveChain::from_fen("rnbqk1nr/ppp1bppp/3p4/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 3 4")
                .unwrap();
        assert_eq!(
            chain.last().as_fen(),
            "rnbqk1nr/ppp1bppp/3p4/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 3 4"
        );
        assert_eq!(
            chain.startpos().as_fen(),
            "rnbqk1nr/ppp1bppp/3p4/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 3 4"
        );
        assert_eq!(chain.len(), 0);

        chain.push(make::Uci("g8f6")).unwrap();
        assert_eq!(
            chain.last().as_fen(),
            "rnbqk2r/ppp1bppp/3p1n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 4 5"
        );
        assert_eq!(
            chain.startpos().as_fen(),
            "rnbqk1nr/ppp1bppp/3p4/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 3 4"
        );
        assert_eq!(chain.len(), 1);

        chain.push(make::Uci("d2d4")).unwrap();
        assert_eq!(
            chain.last().as_fen(),
            "rnbqk2r/ppp1bppp/3p1n2/4p3/2BPP3/5N2/PPP2PPP/RNBQ1RK1 b kq d3 0 5"
        );
        assert_eq!(
            chain.startpos().as_fen(),
            "rnbqk1nr/ppp1bppp/3p4/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 3 4"
        );
        assert_eq!(chain.len(), 2);

        chain.pop().unwrap();
        assert_eq!(
            chain.last().as_fen(),
            "rnbqk2r/ppp1bppp/3p1n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 4 5"
        );
        assert_eq!(
            chain.startpos().as_fen(),
            "rnbqk1nr/ppp1bppp/3p4/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 3 4"
        );
        assert_eq!(chain.len(), 1);

        chain.push(make::San("d4")).unwrap();
        assert_eq!(
            chain.last().as_fen(),
            "rnbqk2r/ppp1bppp/3p1n2/4p3/2BPP3/5N2/PPP2PPP/RNBQ1RK1 b kq d3 0 5"
        );
        assert_eq!(
            chain.startpos().as_fen(),
            "rnbqk1nr/ppp1bppp/3p4/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 3 4"
        );
        assert_eq!(chain.len(), 2);

        assert_eq!(
            chain.iter().map(|m| m.to_string()).collect::<Vec<_>>(),
            vec!["g8f6", "d2d4"]
        );
        assert_eq!(chain.get(0).to_string(), "g8f6");
        assert_eq!(chain.get(1).to_string(), "d2d4");
        assert_eq!(chain.uci().to_string(), "g8f6 d2d4");
    }

    #[test]
    fn test_repeat() {
        let mut chain = MoveChain::new_initial();
        assert!(!chain.is_finished());
        assert_eq!(chain.outcome(), &None);

        chain
            .push_uci_list("g1f3 b8c6 f3g1 c6b8 g1f3 b8c6 f3g1 c6b8")
            .unwrap();
        assert_eq!(chain.outcome(), &None);
        assert_eq!(
            chain.calc_outcome(),
            Some(Outcome::Draw(DrawReason::Repeat3))
        );

        let _ = chain.set_auto_outcome(OutcomeFilter::Strict);
        assert_eq!(chain.outcome(), &None);

        chain
            .push_uci_list("g1f3 b8c6 f3g1 c6b8 g1f3 b8c6 f3g1 c6b8")
            .unwrap();
        assert_eq!(chain.outcome(), &None);
        assert_eq!(
            chain.calc_outcome(),
            Some(Outcome::Draw(DrawReason::Repeat5))
        );

        let _ = chain.set_auto_outcome(OutcomeFilter::Strict);
        assert!(chain.is_finished());
        assert_eq!(chain.outcome(), &Some(Outcome::Draw(DrawReason::Repeat5)));

        chain.pop().unwrap();
        assert!(!chain.is_finished());
        assert_eq!(chain.outcome(), &None);

        chain.set_auto_outcome(OutcomeFilter::Relaxed);
        assert!(chain.is_finished());
        assert_eq!(chain.outcome(), &Some(Outcome::Draw(DrawReason::Repeat3)));

        assert_eq!(
            chain.uci().to_string(),
            "g1f3 b8c6 f3g1 c6b8 g1f3 b8c6 f3g1 c6b8 g1f3 b8c6 f3g1 c6b8 g1f3 b8c6 f3g1"
        );
    }

    #[test]
    fn test_checkmate() {
        let mut chain = MoveChain::new_initial();
        chain.push_uci_list("g2g4 e7e5 f2f4 d8h4").unwrap();
        assert_eq!(
            chain.set_auto_outcome(OutcomeFilter::Force),
            Some(Outcome::Win {
                side: Color::Black,
                reason: WinReason::Checkmate
            }),
        );
        assert!(chain.is_finished());
        assert_eq!(
            chain.outcome(),
            &Some(Outcome::Win {
                side: Color::Black,
                reason: WinReason::Checkmate
            })
        );

        assert_eq!(
            chain
                .styled(
                    NumberPolicy::FromBoard,
                    moves::Style::San,
                    GameStatusPolicy::Show,
                )
                .to_string(),
            "1. g4 e5 2. f4 Qh4# 0-1".to_string()
        );
    }

    #[test]
    fn test_walker() {
        let mut chain = MoveChain::new_initial();
        for mv in [
            "e4", "e5", "Nf3", "d6", "Bc4", "Bg4", "Nc3", "g6", "Nxe5", "Bxd1", "Bxf7", "Ke7",
            "Nd5#",
        ] {
            chain.push(make::San(mv)).unwrap();
        }

        let mut w = chain.walk();
        assert_eq!(w.len(), 13);

        assert_eq!(w.pos(), 0);
        assert_eq!(w.prev(), None);
        assert_eq!(w.pos(), 0);

        w.start();
        assert_eq!(w.pos(), 0);
        assert_eq!(w.prev(), None);
        assert_eq!(w.pos(), 0);

        w.end();
        assert_eq!(w.pos(), 13);
        assert_eq!(w.next(), None);
        assert_eq!(w.pos(), 13);

        w.prev().unwrap();
        w.prev().unwrap();
        let (b, mv) = w.prev().unwrap();
        assert_eq!(
            b.as_fen(),
            "rn1qkbnr/ppp2p1p/3p2p1/4N3/2B1P3/2N5/PPPP1PPP/R1BbK2R w KQkq - 0 6"
        );
        assert_eq!(mv.san(b).unwrap().to_string(), "Bxf7+");
        assert_eq!(w.pos(), 10);

        let (b, mv) = w.next().unwrap();
        assert_eq!(
            b.as_fen(),
            "rn1qkbnr/ppp2p1p/3p2p1/4N3/2B1P3/2N5/PPPP1PPP/R1BbK2R w KQkq - 0 6"
        );
        assert_eq!(mv.san(b).unwrap().to_string(), "Bxf7+");
        assert_eq!(w.pos(), 11);

        let (b, mv) = w.next().unwrap();
        assert_eq!(
            b.as_fen(),
            "rn1qkbnr/ppp2B1p/3p2p1/4N3/4P3/2N5/PPPP1PPP/R1BbK2R b KQkq - 0 6"
        );
        assert_eq!(mv.san(b).unwrap().to_string(), "Ke7");
        assert_eq!(w.pos(), 12);
    }

    #[test]
    fn test_styled() {
        let chain = MoveChain::from_uci_list(Board::initial(), "e2e4 e7e5 g1f3 d7d6 f1b5").unwrap();
        assert_eq!(chain.len(), 5);
        assert_eq!(
            chain.last().as_fen(),
            "rnbqkbnr/ppp2ppp/3p4/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 3"
        );
        assert_eq!(
            chain
                .styled(
                    NumberPolicy::Omit,
                    moves::Style::San,
                    GameStatusPolicy::Hide
                )
                .to_string(),
            "e4 e5 Nf3 d6 Bb5+".to_string()
        );
        assert_eq!(
            chain
                .styled(
                    NumberPolicy::FromBoard,
                    moves::Style::San,
                    GameStatusPolicy::Hide
                )
                .to_string(),
            "1. e4 e5 2. Nf3 d6 3. Bb5+".to_string()
        );
        assert_eq!(
            chain
                .styled(
                    NumberPolicy::Custom(42),
                    moves::Style::San,
                    GameStatusPolicy::Hide
                )
                .to_string(),
            "42. e4 e5 43. Nf3 d6 44. Bb5+".to_string()
        );

        assert_eq!(
            chain
                .styled(
                    NumberPolicy::FromBoard,
                    moves::Style::SanUtf8,
                    GameStatusPolicy::Hide
                )
                .to_string(),
            "1. e4 e5 2. ♘f3 d6 3. ♗b5+".to_string()
        );
        assert_eq!(
            chain
                .styled(
                    NumberPolicy::FromBoard,
                    moves::Style::Uci,
                    GameStatusPolicy::Hide
                )
                .to_string(),
            "1. e2e4 e7e5 2. g1f3 d7d6 3. f1b5".to_string()
        );

        let chain =
            MoveChain::from_uci_list(Board::initial(), "e2e4 e7e5 g1f3 d7d6 f1b5 c7c6").unwrap();
        assert_eq!(
            chain.last().as_fen(),
            "rnbqkbnr/pp3ppp/2pp4/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"
        );
        assert_eq!(chain.len(), 6);
        assert_eq!(
            chain
                .styled(
                    NumberPolicy::Omit,
                    moves::Style::San,
                    GameStatusPolicy::Hide
                )
                .to_string(),
            "e4 e5 Nf3 d6 Bb5+ c6".to_string()
        );
        assert_eq!(
            chain
                .styled(
                    NumberPolicy::FromBoard,
                    moves::Style::San,
                    GameStatusPolicy::Hide
                )
                .to_string(),
            "1. e4 e5 2. Nf3 d6 3. Bb5+ c6".to_string()
        );
        assert_eq!(
            chain
                .styled(
                    NumberPolicy::Custom(42),
                    moves::Style::San,
                    GameStatusPolicy::Hide
                )
                .to_string(),
            "42. e4 e5 43. Nf3 d6 44. Bb5+ c6".to_string()
        );

        let board = Board::from_fen("5K2/4P3/8/8/8/8/6p1/7k b - - 0 12").unwrap();
        let chain = MoveChain::from_uci_list(board.clone(), "g2g1q e7e8q").unwrap();
        assert_eq!(chain.last().as_fen(), "4QK2/8/8/8/8/8/8/6qk b - - 0 13");
        assert_eq!(chain.len(), 2);
        assert_eq!(
            chain
                .styled(
                    NumberPolicy::Omit,
                    moves::Style::San,
                    GameStatusPolicy::Hide
                )
                .to_string(),
            "g1=Q e8=Q".to_string()
        );
        assert_eq!(
            chain
                .styled(
                    NumberPolicy::FromBoard,
                    moves::Style::San,
                    GameStatusPolicy::Hide
                )
                .to_string(),
            "12... g1=Q 13. e8=Q".to_string()
        );
        assert_eq!(
            chain
                .styled(
                    NumberPolicy::Custom(42),
                    moves::Style::San,
                    GameStatusPolicy::Hide
                )
                .to_string(),
            "42... g1=Q 43. e8=Q".to_string()
        );

        let board = Board::from_fen("5K2/4P3/8/8/8/8/6p1/7k b - - 0 12").unwrap();
        let chain = MoveChain::from_uci_list(board.clone(), "g2g1q e7e8q g1c5").unwrap();
        assert_eq!(chain.last().as_fen(), "4QK2/8/8/2q5/8/8/8/7k w - - 1 14");
        assert_eq!(chain.len(), 3);
        assert_eq!(
            chain
                .styled(
                    NumberPolicy::Omit,
                    moves::Style::San,
                    GameStatusPolicy::Hide
                )
                .to_string(),
            "g1=Q e8=Q Qc5+".to_string()
        );
        assert_eq!(
            chain
                .styled(
                    NumberPolicy::FromBoard,
                    moves::Style::San,
                    GameStatusPolicy::Hide
                )
                .to_string(),
            "12... g1=Q 13. e8=Q Qc5+".to_string()
        );
        assert_eq!(
            chain
                .styled(
                    NumberPolicy::Custom(42),
                    moves::Style::San,
                    GameStatusPolicy::Hide
                )
                .to_string(),
            "42... g1=Q 43. e8=Q Qc5+".to_string()
        );

        let mut chain =
            MoveChain::from_uci_list(Board::initial(), "e2e4 e7e5 g1f3 d7d6 f1b5").unwrap();
        chain.set_outcome(Outcome::Draw(DrawReason::Agreement));
        assert_eq!(
            chain
                .styled(
                    NumberPolicy::FromBoard,
                    moves::Style::San,
                    GameStatusPolicy::Show
                )
                .to_string(),
            "1. e4 e5 2. Nf3 d6 3. Bb5+ 1/2-1/2".to_string()
        );
    }
}
