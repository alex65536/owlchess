use crate::board::{self, Board};
use crate::moves::{self, uci, san, Move, RawUndo, ValidateError};
use crate::types::{Color, DrawKind, Outcome, OutcomeFilter};

use std::collections::HashMap;
use std::fmt;

use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
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

#[derive(Default, Debug, Clone, Eq, PartialEq)]
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

    pub fn len(&self) -> usize {
        self.stack.len()
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

    pub fn push_uci(&mut self, s: &str) -> Result<(), uci::ParseError> {
        let mv = Move::from_uci_semilegal(s, &self.board)?;
        unsafe { self.try_push_unchecked(mv).map_err(uci::ParseError::Validate)?; }
        Ok(())
    }

    pub fn push_san(&mut self, s: &str) -> Result<(), san::ParseError> {
        let mv = Move::from_san(s, &self.board)?;
        unsafe { self.push_unchecked(mv); }
        Ok(())
    }

    pub fn push_uci_list(&mut self, uci_list: &str) -> Result<(), UciParseError> {
        for (pos, token) in uci_list.split_ascii_whitespace().enumerate() {
            self.push_uci(token).map_err(|source| UciParseError { pos, source })?;
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

    pub fn walk(&self) -> Walker<'_> {
        Walker {
            board: self.board.clone(),
            stack: &self.stack,
            pos: 0,
            board_pos: self.stack.len(),
        }
    }

    pub fn san_list(&self, policy: NumberPolicy) -> SanList<'_, R> {
        SanList {inner: self, policy}
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

pub struct Walker<'a> {
    board: Board,
    stack: &'a [(Move, RawUndo)],
    pos: usize,
    board_pos: usize,
}

impl<'a> Walker<'a> {
    pub fn len(&self) -> usize {
        self.stack.len()
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    fn set_board_pos(&mut self, target: usize) {
        while self.board_pos > target {
            self.board_pos -= 1;
            let (mv, u) = self.stack[self.board_pos];
            unsafe { moves::unmake_move_unchecked(&mut self.board, mv, u); }
        }
        while self.board_pos < target {
            let (mv, _) = self.stack[self.board_pos];
            unsafe { moves::make_move_unchecked(&mut self.board, mv); }
            self.board_pos += 1;
        }
    }

    pub fn next(&mut self) -> Option<(&Board, Move)> {
        if self.pos == self.stack.len() {
            return None;
        }
        self.pos += 1;
        self.set_board_pos(self.pos - 1);
        Some((&self.board, self.stack[self.pos - 1].0))
    }

    pub fn prev(&mut self) -> Option<(&Board, Move)> {
        if self.pos == 0 {
            return None;
        }
        self.pos -= 1;
        self.set_board_pos(self.pos);
        Some((&self.board, self.stack[self.pos].0))
    }

    pub fn start(&mut self) {
        self.pos = 0;
    }

    pub fn end(&mut self) {
        self.pos = self.stack.len();
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum NumberPolicy {
    Omit,
    FromBoard,
    Custom(usize),
}

pub struct SanList<'a, R: Repeat> {
    inner: &'a BaseMoveChain<R>,
    policy: NumberPolicy,
}

impl<'a, R: Repeat> fmt::Display for SanList<'a, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        if self.inner.len() == 0 {
            return Ok(());
        }

        let mut walker = self.inner.walk();
        let (b, mv) = walker.next().unwrap();
        let real_start_num = b.raw().move_number as usize;
        let start_num = match self.policy {
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
        write!(f, "{}", mv.san(b).unwrap())?;

        while let Some((b, mv)) = walker.next() {
            if let Some(num) = start_num {
                if b.side() == Color::White {
                    write!(f, "{}. ", b.raw().move_number as usize - real_start_num + num)?;
                }
            }
            write!(f, " {}", mv.san(b).unwrap())?;
        }

        Ok(())
    }
}

// TODO : tests
// TODO : tests for SanMoveList
