use crate::board::{self, Board, RawBoard};
use crate::moves::{self, san, uci, Move, RawUndo, ValidateError};
use crate::types::{Color, DrawKind, GameStatus, Outcome, OutcomeFilter};

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
    start: RawBoard,
    board: Board,
    repeat: R,
    stack: Vec<(Move, RawUndo)>,
    outcome: Option<Outcome>,
}

impl<R: Repeat> BaseMoveChain<R> {
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

    #[inline]
    pub fn startpos(&self) -> &RawBoard {
        &self.start
    }

    #[inline]
    pub fn last(&self) -> &Board {
        &self.board
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.stack.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = Move> + '_ {
        self.stack.iter().map(|(m, _)| *m)
    }

    #[inline]
    pub fn get(&self, idx: usize) -> Move {
        self.stack[idx].0
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, idx: usize) -> Move {
        self.stack.get_unchecked(idx).0
    }

    #[inline]
    pub fn outcome(&self) -> &Option<Outcome> {
        &self.outcome
    }

    #[inline]
    pub fn is_finished(&self) -> bool {
        self.outcome.is_some()
    }

    #[inline]
    pub fn clear_outcome(&mut self) {
        self.outcome = None;
    }

    #[inline]
    pub fn set_outcome(&mut self, outcome: Outcome) {
        assert!(!self.is_finished());
        self.outcome = Some(outcome);
    }

    #[inline]
    pub fn reset_outcome(&mut self, outcome: Option<Outcome>) {
        self.outcome = outcome;
    }

    pub fn calc_outcome(&self) -> Option<Outcome> {
        // We need to handle the priority of different outcomes carefully.
        //
        // Checkmate and stalemate are definitely preferred over all the other
        // outcomes, though they cannot happen at the same time with draw by
        // repetitions.
        //
        // Next, all the strict outcomes must be checked before the non-strict ones.
        // For example, if there is both `DrawKind::Moves50` and `DrawKind::Repeat5`,
        // the latter must be preferred, as it's strict. Still, we have no priority
        // between `DrawKind::Moves75` and `DrawKind::Repeat5` or between
        // `DrawKind::Moves50` and `DrawKind::Repeat3`.
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

        let rep = self.repeat.repeat_count(&self.board);
        if rep >= 5 {
            return Some(Outcome::Draw(DrawKind::Repeat5));
        }
        if rep >= 3 {
            return Some(Outcome::Draw(DrawKind::Repeat3));
        }

        outcome
    }

    pub fn set_auto_outcome(&mut self, filter: OutcomeFilter) -> Option<Outcome> {
        assert!(!self.is_finished());
        if let Some(outcome) = self.calc_outcome() {
            if outcome.passes(filter) {
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

    pub fn push(&mut self, mv: Move) -> Result<(), ValidateError> {
        assert!(!self.is_finished());
        mv.semi_validate(&self.board)?;
        unsafe { self.try_push_unchecked(mv) }
    }

    pub fn push_uci(&mut self, s: &str) -> Result<(), uci::ParseError> {
        let mv = Move::from_uci_semilegal(s, &self.board)?;
        unsafe {
            self.try_push_unchecked(mv)
                .map_err(uci::ParseError::Validate)?;
        }
        Ok(())
    }

    pub fn push_san(&mut self, s: &str) -> Result<(), san::ParseError> {
        let mv = Move::from_san(s, &self.board)?;
        unsafe {
            self.push_unchecked(mv);
        }
        Ok(())
    }

    pub fn push_uci_list(&mut self, uci_list: &str) -> Result<(), UciParseError> {
        for (pos, token) in uci_list.split_ascii_whitespace().enumerate() {
            self.push_uci(token)
                .map_err(|source| UciParseError { pos, source })?;
        }
        Ok(())
    }

    pub fn pop(&mut self) -> Option<Move> {
        let (m, u) = self.stack.pop()?;
        self.repeat.pop(&self.board);
        self.clear_outcome();
        unsafe { moves::unmake_move_unchecked(&mut self.board, m, u) };
        Some(m)
    }

    #[inline]
    pub fn uci(&self) -> UciList<'_, R> {
        UciList(self)
    }

    #[inline]
    pub fn walk(&self) -> Walker<'_> {
        Walker {
            board: self.board.clone(),
            stack: &self.stack,
            pos: 0,
            board_pos: self.stack.len(),
        }
    }

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
    #[inline]
    pub fn len(&self) -> usize {
        self.stack.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

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

    pub fn walk_next(&mut self) -> Option<(&Board, Move)> {
        if self.pos == self.stack.len() {
            return None;
        }
        self.pos += 1;
        self.set_board_pos(self.pos - 1);
        Some((&self.board, self.stack[self.pos - 1].0))
    }

    pub fn walk_prev(&mut self) -> Option<(&Board, Move)> {
        if self.pos == 0 {
            return None;
        }
        self.pos -= 1;
        self.set_board_pos(self.pos);
        Some((&self.board, self.stack[self.pos].0))
    }

    #[inline]
    pub fn walk_start(&mut self) {
        self.pos = 0;
    }

    #[inline]
    pub fn walk_end(&mut self) {
        self.pos = self.stack.len();
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum NumberPolicy {
    Omit,
    FromBoard,
    Custom(usize),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum GameStatusPolicy {
    Show,
    Hide,
}

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
        let (b, mv) = walker.walk_next().unwrap();
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

        while let Some((b, mv)) = walker.walk_next() {
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
    use crate::types::{DrawKind, Outcome, OutcomeFilter, WinKind};

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

        chain.push_uci("g8f6").unwrap();
        assert_eq!(
            chain.last().as_fen(),
            "rnbqk2r/ppp1bppp/3p1n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 4 5"
        );
        assert_eq!(
            chain.startpos().as_fen(),
            "rnbqk1nr/ppp1bppp/3p4/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 3 4"
        );
        assert_eq!(chain.len(), 1);

        chain.push_uci("d2d4").unwrap();
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

        chain.push_san("d4").unwrap();
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
        assert_eq!(chain.calc_outcome(), Some(Outcome::Draw(DrawKind::Repeat3)));

        let _ = chain.set_auto_outcome(OutcomeFilter::Strict);
        assert_eq!(chain.outcome(), &None);

        chain
            .push_uci_list("g1f3 b8c6 f3g1 c6b8 g1f3 b8c6 f3g1 c6b8")
            .unwrap();
        assert_eq!(chain.outcome(), &None);
        assert_eq!(chain.calc_outcome(), Some(Outcome::Draw(DrawKind::Repeat5)));

        let _ = chain.set_auto_outcome(OutcomeFilter::Strict);
        assert!(chain.is_finished());
        assert_eq!(chain.outcome(), &Some(Outcome::Draw(DrawKind::Repeat5)));

        chain.pop().unwrap();
        assert!(!chain.is_finished());
        assert_eq!(chain.outcome(), &None);

        chain.set_auto_outcome(OutcomeFilter::Relaxed);
        assert!(chain.is_finished());
        assert_eq!(chain.outcome(), &Some(Outcome::Draw(DrawKind::Repeat3)));

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
            Some(Outcome::Black(WinKind::Checkmate)),
        );
        assert!(chain.is_finished());
        assert_eq!(chain.outcome(), &Some(Outcome::Black(WinKind::Checkmate)));

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
            chain.push_san(mv).unwrap();
        }

        let mut w = chain.walk();
        assert_eq!(w.len(), 13);

        assert_eq!(w.pos(), 0);
        assert_eq!(w.walk_prev(), None);
        assert_eq!(w.pos(), 0);

        w.walk_start();
        assert_eq!(w.pos(), 0);
        assert_eq!(w.walk_prev(), None);
        assert_eq!(w.pos(), 0);

        w.walk_end();
        assert_eq!(w.pos(), 13);
        assert_eq!(w.walk_next(), None);
        assert_eq!(w.pos(), 13);

        w.walk_prev().unwrap();
        w.walk_prev().unwrap();
        let (b, mv) = w.walk_prev().unwrap();
        assert_eq!(
            b.as_fen(),
            "rn1qkbnr/ppp2p1p/3p2p1/4N3/2B1P3/2N5/PPPP1PPP/R1BbK2R w KQkq - 0 6"
        );
        assert_eq!(mv.san(b).unwrap().to_string(), "Bxf7+");
        assert_eq!(w.pos(), 10);

        let (b, mv) = w.walk_next().unwrap();
        assert_eq!(
            b.as_fen(),
            "rn1qkbnr/ppp2p1p/3p2p1/4N3/2B1P3/2N5/PPPP1PPP/R1BbK2R w KQkq - 0 6"
        );
        assert_eq!(mv.san(b).unwrap().to_string(), "Bxf7+");
        assert_eq!(w.pos(), 11);

        let (b, mv) = w.walk_next().unwrap();
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
        chain.set_outcome(Outcome::Draw(DrawKind::Agreement));
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
