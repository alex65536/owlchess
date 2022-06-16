use crate::board::{self, Board};
use crate::moves::{self, Move, RawUndo, ValidateError};
use crate::types::{DrawKind, Outcome, OutcomeFilter};

use std::collections::HashMap;
use std::fmt;

use thiserror::Error;

#[derive(Debug, Error, Copy, Clone, PartialEq, Eq)]
#[error("cannot parse UCI move #{}: {}", .pos + 1, .source)]
pub struct UciParseError {
    pub pos: usize,
    pub source: moves::uci::ParseError,
}

pub trait Repeat: Default {
    fn push(&mut self, b: &Board);
    fn pop(&mut self, b: &Board);
    fn repeat_count(&self, b: &Board) -> usize;
}

#[derive(Default, Debug, Eq, PartialEq, Clone)]
pub struct HashRepeat(HashMap<u64, usize>);

impl HashRepeat {
    pub fn new() -> Self {
        Self::default()
    }
}

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

    fn repeat_count(&self, b: &Board) -> usize {
        *self.0.get(&b.zobrist_hash()).unwrap_or(&0)
    }
}

pub type MoveChain = BaseMoveChain<HashRepeat>;

#[derive(Debug, Clone)]
pub struct BaseMoveChain<R: Repeat> {
    board: Board,
    repeat: R,
    stack: Vec<(Move, RawUndo)>,
    outcome: Option<Outcome>,
}

impl<R: Repeat> BaseMoveChain<R> {
    pub fn new(b: Board) -> Self {
        let mut res = BaseMoveChain {
            board: b,
            repeat: R::default(),
            stack: Vec::new(),
            outcome: None,
        };
        res.repeat.push(&res.board);
        res
    }

    pub fn new_initial() -> Self {
        Self::new(Board::initial())
    }

    pub fn from_uci_list(b: Board, uci_list: &str) -> Result<Self, UciParseError> {
        let mut res = BaseMoveChain::new(b);
        res.push_uci_list(uci_list)?;
        Ok(res)
    }

    pub fn from_fen(s: &str) -> Result<Self, board::FenParseError> {
        Ok(Self::new(Board::from_fen(s)?))
    }

    pub fn last(&self) -> &Board {
        &self.board
    }

    pub fn iter(&self) -> impl Iterator<Item = Move> + '_ {
        self.stack.iter().map(|(m, _)| *m)
    }

    pub fn get(&self, idx: usize) -> Move {
        self.stack[idx].0
    }

    pub unsafe fn get_unchecked(&self, idx: usize) -> Move {
        self.stack.get_unchecked(idx).0
    }

    pub fn outcome(&self) -> &Option<Outcome> {
        &self.outcome
    }

    pub fn is_finished(&self) -> bool {
        self.outcome.is_none()
    }

    pub fn clear_outcome(&mut self) {
        self.outcome = None;
    }

    pub fn set_outcome(&mut self, outcome: Outcome) {
        assert!(!self.is_finished());
        self.outcome = Some(outcome);
    }

    pub fn reset_outcome(&mut self, outcome: Option<Outcome>) {
        self.outcome = outcome;
    }

    pub fn calc_outcome(&self) -> Option<Outcome> {
        let rep = self.repeat.repeat_count(&self.board);
        if rep >= 5 {
            return Some(Outcome::Draw(DrawKind::Repeat5));
        }
        if rep >= 3 {
            return Some(Outcome::Draw(DrawKind::Repeat3));
        }
        self.board.calc_outcome()
    }

    pub fn set_auto_outcome(&mut self, filter: OutcomeFilter) -> Option<Outcome> {
        assert!(!self.is_finished());
        if let Some(outcome) = self.calc_outcome() {
            if outcome.is_auto(filter) {
                self.set_outcome(outcome);
            }
        }
        self.outcome
    }

    fn do_finish_push(&mut self, mv: Move, u: RawUndo) {
        self.repeat.push(&self.board);
        self.stack.push((mv, u));
    }

    pub unsafe fn push_unchecked(&mut self, mv: Move) {
        let u = moves::make_move_unchecked(&mut self.board, mv);
        self.do_finish_push(mv, u);
    }

    pub unsafe fn try_push_unchecked(&mut self, mv: Move) -> Result<(), ValidateError> {
        let u = moves::try_make_move_unchecked(&mut self.board, mv)?;
        self.do_finish_push(mv, u);
        Ok(())
    }

    pub fn push_weak(&mut self, mv: Move) -> Result<(), ValidateError> {
        assert!(!self.is_finished());
        if !moves::is_move_sane(&self.board, mv) {
            return Err(ValidateError::NotSane);
        }
        unsafe { self.try_push_unchecked(mv) }
    }

    pub fn push(&mut self, mv: Move) -> Result<(), ValidateError> {
        assert!(!self.is_finished());
        mv.semi_validate(&self.board)?;
        unsafe { self.try_push_unchecked(mv) }
    }

    pub fn push_uci_list(&mut self, uci_list: &str) -> Result<(), UciParseError> {
        for (pos, token) in uci_list.split_ascii_whitespace().enumerate() {
            Move::from_uci_semilegal(token, &self.board)
                .and_then(|mv| unsafe { Ok(self.try_push_unchecked(mv)?) })
                .map_err(|source| UciParseError { pos, source })?;
        }
        Ok(())
    }

    pub fn pop(&mut self) -> Option<Move> {
        let (m, u) = self.stack.pop()?;
        self.repeat.pop(&self.board);
        unsafe { moves::unmake_move_unchecked(&mut self.board, m, u) };
        Some(m)
    }

    pub fn uci_list(&self) -> UciList<'_, R> {
        UciList(self)
    }
}

impl<R: Repeat + Eq> PartialEq<Self> for BaseMoveChain<R> {
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

// TODO : tests
