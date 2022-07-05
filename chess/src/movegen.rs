//! Move generation and related things

use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::moves::{self, Move, MoveKind, PromotePiece};
use crate::types::{CastlingSide, Cell, Color, Coord, File, Piece};
use crate::{attack, bitboard_consts, castling, generic, geometry, pawns};

use std::convert::Infallible;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::slice;

use arrayvec::ArrayVec;

fn diag_pieces(b: &Board, c: Color) -> Bitboard {
    b.piece2(c, Piece::Bishop) | b.piece2(c, Piece::Queen)
}

fn line_pieces(b: &Board, c: Color) -> Bitboard {
    b.piece2(c, Piece::Rook) | b.piece2(c, Piece::Queen)
}

pub(crate) fn do_is_cell_attacked<C: generic::Color>(b: &Board, coord: Coord) -> bool {
    // Here, we use black attack map for white, as we need to trace the attack from destination piece,
    // not from the source one
    let pawn_attacks = attack::pawn(C::COLOR.inv(), coord);

    // Near attacks
    if (b.piece2(C::COLOR, Piece::Pawn) & pawn_attacks).is_nonempty()
        || (b.piece2(C::COLOR, Piece::King) & attack::king(coord)).is_nonempty()
        || (b.piece2(C::COLOR, Piece::Knight) & attack::knight(coord)).is_nonempty()
    {
        return true;
    }

    // Far attacks
    (attack::bishop(coord, b.all) & diag_pieces(b, C::COLOR)).is_nonempty()
        || (attack::rook(coord, b.all) & line_pieces(b, C::COLOR)).is_nonempty()
}

fn do_cell_attackers<C: generic::Color>(b: &Board, coord: Coord) -> Bitboard {
    let pawn_attacks = attack::pawn(C::COLOR.inv(), coord);
    (b.piece2(C::COLOR, Piece::Pawn) & pawn_attacks)
        | (b.piece2(C::COLOR, Piece::King) & attack::king(coord))
        | (b.piece2(C::COLOR, Piece::Knight) & attack::knight(coord))
        | (attack::bishop(coord, b.all) & diag_pieces(b, C::COLOR))
        | (attack::rook(coord, b.all) & line_pieces(b, C::COLOR))
}

/// Returns true if the square `coord` is attacked by pieces of color `color`
///
/// Suppose that the piece on the position `coord` is replaced by a piece of
/// color opposite to `color`, and `color` is to move. Then, the function checks
/// if the side `color` is able to capture the piece on the position `coord` via
/// a pseudo-legal move.
///
/// This function does the same as `cell_attackers(b, coord, color).is_nonempty()`.
///
/// **Note**: this function doesn't consider enpassant a capture.
pub fn is_cell_attacked(b: &Board, coord: Coord, color: Color) -> bool {
    match color {
        Color::White => do_is_cell_attacked::<generic::White>(b, coord),
        Color::Black => do_is_cell_attacked::<generic::Black>(b, coord),
    }
}

/// Returns the bitboard over all the pieces of color `color` that attack the square
/// `coord`
///
/// Suppose that the piece on the position `coord` is replaced by a piece of
/// color opposite to `color`, and `color` is to move. Then, the function enumerates
/// all the pieces of the side `color` which are able to capture the piece on the
/// position `coord` via a pseudo-legal move.
///
/// **Note**: this function doesn't consider enpassant a capture.
pub fn cell_attackers(b: &Board, coord: Coord, color: Color) -> Bitboard {
    match color {
        Color::White => do_cell_attackers::<generic::White>(b, coord),
        Color::Black => do_cell_attackers::<generic::Black>(b, coord),
    }
}

trait MaybeMovePush {
    type Err;

    fn push(&mut self, m: Move) -> Result<(), Self::Err>;
}

/// Compact, allocation-free vector which is able to hold all the moves in every given
/// position
///
/// This is a wrapper over [`ArrayVec`] and behaves basically the same way.
#[derive(Default, Debug, Clone, Eq, PartialEq)]
pub struct MoveList(ArrayVec<Move, 256>);

impl Deref for MoveList {
    type Target = ArrayVec<Move, 256>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for MoveList {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a> IntoIterator for &'a MoveList {
    type Item = &'a Move;
    type IntoIter = slice::Iter<'a, Move>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a> IntoIterator for &'a mut MoveList {
    type Item = &'a mut Move;
    type IntoIter = slice::IterMut<'a, Move>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

impl MoveList {
    /// Creates a new, empty `MoveList`
    #[inline]
    pub fn new() -> MoveList {
        MoveList(ArrayVec::new())
    }
}

/// Push a move into a collection
pub trait MovePush {
    /// Pushes the move `m` into a collection
    fn push(&mut self, m: Move);
}

impl<const N: usize> MovePush for ArrayVec<Move, N> {
    fn push(&mut self, m: Move) {
        self.push(m);
    }
}

impl MovePush for MoveList {
    fn push(&mut self, m: Move) {
        self.0.push(m);
    }
}

impl MovePush for Vec<Move> {
    fn push(&mut self, m: Move) {
        self.push(m);
    }
}

impl<T: MovePush> MaybeMovePush for T {
    type Err = Infallible;

    fn push(&mut self, m: Move) -> Result<(), Self::Err> {
        <Self as MovePush>::push(self, m);
        Ok(())
    }
}

struct UnsafeMoveList(MoveList);

impl UnsafeMoveList {
    unsafe fn new() -> UnsafeMoveList {
        UnsafeMoveList(MoveList::new())
    }
}

impl MovePush for UnsafeMoveList {
    fn push(&mut self, m: Move) {
        unsafe {
            self.0.push_unchecked(m);
        }
    }
}

struct LegalFilter<'a, P> {
    board: Board,
    inner: &'a mut P,
}

impl<'a, P: MaybeMovePush> LegalFilter<'a, P> {
    unsafe fn new(board: Board, inner: &'a mut P) -> Self {
        Self { board, inner }
    }
}

impl<'a, P: MaybeMovePush> MaybeMovePush for LegalFilter<'a, P> {
    type Err = P::Err;

    fn push(&mut self, mv: Move) -> Result<(), Self::Err> {
        let u = unsafe { moves::make_move_unchecked(&mut self.board, mv) };
        let is_legal = !self.board.is_opponent_king_attacked();
        unsafe { moves::unmake_move_unchecked(&mut self.board, mv, u) };
        match is_legal {
            true => self.inner.push(mv),
            false => Ok(()),
        }
    }
}

struct MoveGenImpl<'a, P, C> {
    board: &'a Board,
    dst: &'a mut P,
    _c: PhantomData<C>,
}

impl<'a, P: MaybeMovePush, C: generic::Color> MoveGenImpl<'a, P, C> {
    fn new(board: &'a Board, dst: &'a mut P, _c: C) -> Self {
        MoveGenImpl {
            board,
            dst,
            _c: PhantomData,
        }
    }

    unsafe fn add_move(&mut self, kind: MoveKind, src: Coord, dst: Coord) -> Result<(), P::Err> {
        self.dst.push(Move::new_unchecked(kind, src, dst, C::COLOR))
    }

    unsafe fn add_pawn_with_promote<const IS_PROMOTE: bool>(
        &mut self,
        src: Coord,
        dst: Coord,
    ) -> Result<(), P::Err> {
        if IS_PROMOTE {
            self.add_move(MoveKind::PromoteKnight, src, dst)?;
            self.add_move(MoveKind::PromoteBishop, src, dst)?;
            self.add_move(MoveKind::PromoteRook, src, dst)?;
            self.add_move(MoveKind::PromoteQueen, src, dst)?;
        } else {
            self.add_move(MoveKind::PawnSimple, src, dst)?;
        }
        Ok(())
    }

    unsafe fn do_gen_pawn_single<const IS_PROMOTE: bool>(
        &mut self,
        pawns: Bitboard,
    ) -> Result<(), P::Err> {
        for dst in pawns::advance_forward(C::COLOR, pawns) & !self.board.all {
            self.add_pawn_with_promote::<IS_PROMOTE>(
                dst.add_unchecked(-geometry::pawn_forward_delta(C::COLOR)),
                dst,
            )?;
        }
        Ok(())
    }

    unsafe fn do_gen_pawn_double(&mut self, pawns: Bitboard) -> Result<(), P::Err> {
        let tmp = pawns::advance_forward(C::COLOR, pawns) & !self.board.all;
        for dst in pawns::advance_forward(C::COLOR, tmp) & !self.board.all {
            let src = dst.add_unchecked(-2 * geometry::pawn_forward_delta(C::COLOR));
            self.add_move(MoveKind::PawnDouble, src, dst)?;
        }
        Ok(())
    }

    unsafe fn do_gen_pawn_capture<const IS_PROMOTE: bool>(
        &mut self,
        pawns: Bitboard,
    ) -> Result<(), P::Err> {
        let allowed = self.board.color(C::COLOR.inv());
        let left_delta = geometry::pawn_left_delta(C::COLOR);
        for dst in pawns::advance_left(C::COLOR, pawns) & allowed {
            self.add_pawn_with_promote::<IS_PROMOTE>(dst.add_unchecked(-left_delta), dst)?;
        }
        let right_delta = geometry::pawn_right_delta(C::COLOR);
        for dst in pawns::advance_right(C::COLOR, pawns) & allowed {
            self.add_pawn_with_promote::<IS_PROMOTE>(dst.add_unchecked(-right_delta), dst)?;
        }
        Ok(())
    }

    fn gen_pawn_simple<const NON_PROMOTE: bool, const PROMOTE: bool>(
        &mut self,
    ) -> Result<(), P::Err> {
        let promote_mask = bitboard_consts::rank(geometry::promote_src_rank(C::COLOR));
        let double_mask = bitboard_consts::rank(geometry::double_move_src_rank(C::COLOR));
        let pawns = self.board.piece2(C::COLOR, Piece::Pawn);
        if NON_PROMOTE {
            unsafe {
                self.do_gen_pawn_single::<false>(pawns & !promote_mask)?;
                self.do_gen_pawn_double(pawns & double_mask)?;
            }
        }
        if PROMOTE {
            unsafe {
                self.do_gen_pawn_single::<true>(pawns & promote_mask)?;
            }
        }
        Ok(())
    }

    fn gen_pawn_capture(&mut self) -> Result<(), P::Err> {
        let promote_mask = bitboard_consts::rank(geometry::promote_src_rank(C::COLOR));
        let pawns = self.board.piece2(C::COLOR, Piece::Pawn);
        unsafe {
            self.do_gen_pawn_capture::<false>(pawns & !promote_mask)?;
            self.do_gen_pawn_capture::<true>(pawns & promote_mask)?;
        }
        Ok(())
    }

    fn gen_pawn_enpassant(&mut self) -> Result<(), P::Err> {
        if let Some(enpassant) = self.board.r.enpassant {
            let file = enpassant.file();
            let dst = unsafe { enpassant.add_unchecked(geometry::pawn_forward_delta(C::COLOR)) };
            // We assume that the cell behind the pawn that made double move is empty, so don't check it
            let pawn = Cell::from_parts(C::COLOR, Piece::Pawn);
            let (left_pawn, right_pawn) =
                unsafe { (enpassant.add_unchecked(-1), enpassant.add_unchecked(1)) };
            if file != File::A && self.board.get(left_pawn) == pawn {
                unsafe {
                    self.add_move(MoveKind::Enpassant, left_pawn, dst)?;
                }
            }
            if file != File::H && self.board.get(right_pawn) == pawn {
                unsafe {
                    self.add_move(MoveKind::Enpassant, right_pawn, dst)?;
                }
            }
        }
        Ok(())
    }

    fn allowed_mask<const SIMPLE: bool, const CAPTURE: bool>(&self) -> Bitboard {
        match (SIMPLE, CAPTURE) {
            (true, true) => !self.board.color(C::COLOR),
            (true, false) => !self.board.all,
            (false, true) => self.board.color(C::COLOR.inv()),
            (false, false) => Bitboard::EMPTY,
        }
    }

    #[inline]
    fn do_gen_kn<const SIMPLE: bool, const CAPTURE: bool>(
        &mut self,
        p: Piece,
    ) -> Result<(), P::Err> {
        let allowed = self.allowed_mask::<SIMPLE, CAPTURE>();
        for src in self.board.piece2(C::COLOR, p) {
            let attack = match p {
                Piece::Knight => attack::knight(src),
                Piece::King => attack::king(src),
                _ => unreachable!(),
            };
            for dst in attack & allowed {
                unsafe {
                    self.add_move(MoveKind::Simple, src, dst)?;
                }
            }
        }
        Ok(())
    }

    fn gen_knight<const SIMPLE: bool, const CAPTURE: bool>(&mut self) -> Result<(), P::Err> {
        self.do_gen_kn::<SIMPLE, CAPTURE>(Piece::Knight)
    }

    fn gen_king<const SIMPLE: bool, const CAPTURE: bool>(&mut self) -> Result<(), P::Err> {
        self.do_gen_kn::<SIMPLE, CAPTURE>(Piece::King)
    }

    #[inline]
    fn do_gen_brq<const SIMPLE: bool, const CAPTURE: bool, const IS_DIAG: bool>(
        &mut self,
        b: Bitboard,
    ) -> Result<(), P::Err> {
        let allowed = self.allowed_mask::<SIMPLE, CAPTURE>();
        for src in b {
            let attack = match IS_DIAG {
                true => attack::bishop(src, self.board.all),
                false => attack::rook(src, self.board.all),
            };
            for dst in attack & allowed {
                unsafe {
                    self.add_move(MoveKind::Simple, src, dst)?;
                }
            }
        }
        Ok(())
    }

    fn gen_brq<const SIMPLE: bool, const CAPTURE: bool>(&mut self) -> Result<(), P::Err> {
        self.do_gen_brq::<SIMPLE, CAPTURE, true>(diag_pieces(self.board, C::COLOR))?;
        self.do_gen_brq::<SIMPLE, CAPTURE, false>(line_pieces(self.board, C::COLOR))?;
        Ok(())
    }

    fn gen_castling(&mut self) -> Result<(), P::Err> {
        let rank = geometry::castling_rank(C::COLOR);
        if self.board.r.castling.has(C::COLOR, CastlingSide::King) {
            let pass = castling::pass(C::COLOR, CastlingSide::King).shl(C::CASTLING_OFFSET);
            let src = Coord::from_parts(File::E, rank);
            let tmp = Coord::from_parts(File::F, rank);
            let dst = Coord::from_parts(File::G, rank);
            if (pass & self.board.all).is_empty()
                && !do_is_cell_attacked::<C>(self.board, src)
                && !do_is_cell_attacked::<C>(self.board, tmp)
            {
                unsafe {
                    self.add_move(MoveKind::CastlingKingside, src, dst)?;
                }
            }
        }
        if self.board.r.castling.has(C::COLOR, CastlingSide::Queen) {
            let pass = castling::pass(C::COLOR, CastlingSide::Queen).shl(C::CASTLING_OFFSET);
            let src = Coord::from_parts(File::E, rank);
            let tmp = Coord::from_parts(File::D, rank);
            let dst = Coord::from_parts(File::C, rank);
            if (pass & self.board.all).is_empty()
                && !do_is_cell_attacked::<C>(self.board, src)
                && !do_is_cell_attacked::<C>(self.board, tmp)
            {
                unsafe {
                    self.add_move(MoveKind::CastlingQueenside, src, dst)?;
                }
            }
        }
        Ok(())
    }

    fn gen<
        const SIMPLE: bool,
        const CAPTURE: bool,
        const SIMPLE_PROMOTE: bool,
        const CASTLING: bool,
    >(
        &mut self,
    ) -> Result<(), P::Err> {
        if SIMPLE || SIMPLE_PROMOTE {
            self.gen_pawn_simple::<SIMPLE, SIMPLE_PROMOTE>()?;
        }
        if CAPTURE {
            self.gen_pawn_capture()?;
            self.gen_pawn_enpassant()?;
        }
        self.gen_knight::<SIMPLE, CAPTURE>()?;
        self.gen_king::<SIMPLE, CAPTURE>()?;
        self.gen_brq::<SIMPLE, CAPTURE>()?;
        if CASTLING {
            self.gen_castling()?;
        }
        Ok(())
    }

    pub fn gen_all_for_detect(&mut self) -> Result<(), P::Err> {
        self.gen_king::<true, true>()?;
        self.gen_brq::<true, true>()?;
        self.gen_knight::<true, true>()?;
        self.gen_pawn_simple::<true, true>()?;
        self.gen_pawn_capture()?;
        self.gen_pawn_enpassant()?;
        // Castlings are intentionally skipped here, as there is no such situation where castling
        // is the only move
        Ok(())
    }

    pub fn san_candidates(&mut self, piece: Piece, dst: Coord) -> Result<(), P::Err> {
        if self.board.get(dst).color() == Some(C::COLOR) {
            return Ok(());
        }
        let mask = match piece {
            Piece::Pawn => panic!("pawns are not supported here"),
            Piece::King => attack::king(dst),
            Piece::Knight => attack::knight(dst),
            Piece::Bishop => attack::bishop(dst, self.board.all),
            Piece::Rook => attack::rook(dst, self.board.all),
            Piece::Queen => attack::bishop(dst, self.board.all) | attack::rook(dst, self.board.all),
        };
        for src in mask & self.board.piece2(C::COLOR, piece) {
            unsafe {
                self.add_move(MoveKind::Simple, src, dst)?;
            }
        }
        Ok(())
    }

    pub fn san_pawn_capture_candidates(
        &mut self,
        src: File,
        dst: File,
        promote: Option<PromotePiece>,
    ) -> Result<(), P::Err> {
        let promote_mask = bitboard_consts::rank(geometry::promote_src_rank(C::COLOR));
        let pawn_mask = match promote {
            Some(_) => promote_mask,
            None => !promote_mask,
        };
        let pawns =
            self.board.piece2(C::COLOR, Piece::Pawn) & pawn_mask & bitboard_consts::file(src);
        let allowed = self.board.color(C::COLOR.inv());
        let kind = promote
            .map(MoveKind::from)
            .unwrap_or(MoveKind::PawnSimple);
        if src.index() == dst.index() + 1 {
            let left_delta = geometry::pawn_left_delta(C::COLOR);
            for dst in pawns::advance_left(C::COLOR, pawns) & allowed {
                unsafe {
                    self.add_move(kind, dst.add_unchecked(-left_delta), dst)?;
                }
            }
        }
        if src.index() + 1 == dst.index() {
            let right_delta = geometry::pawn_right_delta(C::COLOR);
            for dst in pawns::advance_right(C::COLOR, pawns) & allowed {
                unsafe {
                    self.add_move(kind, dst.add_unchecked(-right_delta), dst)?;
                }
            }
        }

        if let Some(enpassant) = self.board.r.enpassant {
            if enpassant.file() == dst && promote.is_none() {
                let dst_coord =
                    unsafe { enpassant.add_unchecked(geometry::pawn_forward_delta(C::COLOR)) };
                // We assume that the cell behind the pawn that made double move is empty, so don't check it
                let pawn = Cell::from_parts(C::COLOR, Piece::Pawn);
                let (left_pawn, right_pawn) =
                    unsafe { (enpassant.add_unchecked(-1), enpassant.add_unchecked(1)) };
                if src.index() + 1 == dst.index() && self.board.get(left_pawn) == pawn {
                    unsafe {
                        self.add_move(MoveKind::Enpassant, left_pawn, dst_coord)?;
                    }
                }
                if src.index() == dst.index() + 1 && self.board.get(right_pawn) == pawn {
                    unsafe {
                        self.add_move(MoveKind::Enpassant, right_pawn, dst_coord)?;
                    }
                }
            }
        }

        Ok(())
    }

    pub fn gen_all(&mut self) -> Result<(), P::Err> {
        self.gen::<true, true, true, true>()
    }

    pub fn gen_capture(&mut self) -> Result<(), P::Err> {
        self.gen::<false, true, false, false>()
    }

    pub fn gen_simple(&mut self) -> Result<(), P::Err> {
        self.gen::<true, false, true, true>()
    }

    pub fn gen_simple_no_promote(&mut self) -> Result<(), P::Err> {
        self.gen::<true, false, false, true>()
    }

    pub fn gen_simple_promote(&mut self) -> Result<(), P::Err> {
        self.gen::<false, false, true, false>()
    }
}

/// Semilegal move generator
///
/// The move is considered semilegal if it is allowed by chess rules if the king would be
/// allowed to be under attack after making such move.
///
/// Semilegal generator works faster than the legal one, so it can be better in the cases
/// when moves can be validated for legality while trying to apply them, not beforehands.
pub mod semilegal {
    use super::{MoveGenImpl, MoveList, MovePush, UnsafeMoveList};
    use crate::{board::Board, generic, types::Color};

    macro_rules! do_impl {
        ($($(#[$attr:meta])* $name:ident; $(#[$attr_into:meta])* $name_into:ident;)*) => {
            $(
                $(#[$attr])*
                pub fn $name_into<P: MovePush>(b: &Board, dst: &mut P) {
                    let _ = match b.r.side {
                        Color::White => MoveGenImpl::new(b, dst, generic::White).$name(),
                        Color::Black => MoveGenImpl::new(b, dst, generic::Black).$name(),
                    };
                }

                $(#[$attr_into])*
                pub fn $name(b: &Board) -> MoveList {
                    let mut res = unsafe { UnsafeMoveList::new() };
                    $name_into(b, &mut res);
                    res.0
                }
            )*
        }
    }

    do_impl! {
        /// Generates all the semilegal moves in the given position
        gen_all;
        /// Generates all the semilegal moves in the given position and puts generated
        /// moves into `dst`
        gen_all_into;

        /// Generates all the semilegal captures in the given position
        gen_capture;
        /// Generates all the semilegal captures in the given position and puts generated
        /// moves into `dst`
        gen_capture_into;

        /// Generates all the semilegal non-capture moves in the given position
        gen_simple;
        /// Generates all the semilegal non-capture moves in the given position and puts
        /// generated moves into `dst`
        gen_simple_into;

        /// Generates all the semilegal non-capture moves (except promotes) in the given
        /// position
        gen_simple_no_promote;
        /// Generates all the semilegal non-capture moves (except promotes) in the given
        /// position and puts generated moves into `dst`
        gen_simple_no_promote_into;

        /// Generates all the semilegal non-capture promotes in the given position
        gen_simple_promote;
        /// Generates all the semilegal non-capture promotes in the given position and
        /// puts generated moves into `dst`
        gen_simple_promote_into;
    }
}

/// Legal move generator
///
/// Legal moves are the moves fully allowed by chess rules. Use this generator instead of the
/// [semilegal one](semilegal) if you want all the moves to be validated for legality beforehands.
pub mod legal {
    use super::MoveList;
    use crate::board::Board;

    macro_rules! do_impl {
        ($($(#[$attr:meta])* $name:ident;)*) => {
            $(
                $(#[$attr])*
                pub fn $name(b: &Board) -> MoveList {
                    let mut res = super::semilegal::$name(b);
                    let mut b_copy = b.clone();
                    res.retain(|&mut mv| unsafe {
                        let u = crate::moves::make_move_unchecked(&mut b_copy, mv);
                        let ok = !b_copy.is_opponent_king_attacked();
                        crate::moves::unmake_move_unchecked(&mut b_copy, mv, u);
                        ok
                    });
                    res
                }
            )*
        }
    }

    do_impl! {
        /// Generates all the legal moves in the given position
        gen_all;
        /// Generates all the legal captures in the given position
        gen_capture;
        /// Generates all the legal non-capture moves in the given position
        gen_simple;
        /// Generates all the legal non-capture moves (except promotes) in the given position
        gen_simple_no_promote;
        /// Generates all the legal non-capture promotes in the given position
        gen_simple_promote;
    }
}

struct ErrOnFirst;

impl MaybeMovePush for ErrOnFirst {
    type Err = ();

    fn push(&mut self, _mv: Move) -> Result<(), ()> {
        Err(())
    }
}

/// Returns `true` if the current side has at least one legal move
///
/// This function is faster than just generating all the legal moves via [`legal::gen_all`],
/// because it is able to stop on the first generated legal move.
pub fn has_legal_moves(b: &Board) -> bool {
    let mut err_on_first = ErrOnFirst;
    let mut p = unsafe { LegalFilter::new(b.clone(), &mut err_on_first) };
    (match b.r.side {
        Color::White => MoveGenImpl::new(b, &mut p, generic::White).gen_all_for_detect(),
        Color::Black => MoveGenImpl::new(b, &mut p, generic::Black).gen_all_for_detect(),
    })
    .is_err()
}

pub(crate) fn san_candidates<P: MovePush>(b: &Board, piece: Piece, dst: Coord, res: &mut P) {
    let mut p = unsafe { LegalFilter::new(b.clone(), res) };
    let _ = match b.r.side {
        Color::White => MoveGenImpl::new(b, &mut p, generic::White).san_candidates(piece, dst),
        Color::Black => MoveGenImpl::new(b, &mut p, generic::Black).san_candidates(piece, dst),
    };
}

pub(crate) fn san_pawn_capture_candidates<P: MovePush>(
    b: &Board,
    src: File,
    dst: File,
    promote: Option<PromotePiece>,
    res: &mut P,
) {
    let mut p = unsafe { LegalFilter::new(b.clone(), res) };
    let _ = match b.r.side {
        Color::White => MoveGenImpl::new(b, &mut p, generic::White)
            .san_pawn_capture_candidates(src, dst, promote),
        Color::Black => MoveGenImpl::new(b, &mut p, generic::Black)
            .san_pawn_capture_candidates(src, dst, promote),
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitboard::Bitboard;
    use crate::types::{Color, File, Rank};
    use crate::Board;
    use std::collections::BTreeSet;

    #[test]
    fn test_cell_attackers() {
        let b = Board::from_fen("3R3B/8/3R4/1NP1Q3/3p4/1NP5/5B2/3R1K1k w - - 0 1").unwrap();
        assert!(is_cell_attacked(
            &b,
            Coord::from_parts(File::D, Rank::R4),
            Color::White
        ));
        let attackers = Bitboard::EMPTY
            .with(Coord::from_parts(File::D, Rank::R6))
            .with(Coord::from_parts(File::B, Rank::R5))
            .with(Coord::from_parts(File::E, Rank::R5))
            .with(Coord::from_parts(File::B, Rank::R3))
            .with(Coord::from_parts(File::C, Rank::R3))
            .with(Coord::from_parts(File::F, Rank::R2))
            .with(Coord::from_parts(File::D, Rank::R1));
        assert_eq!(
            cell_attackers(&b, Coord::from_parts(File::D, Rank::R4), Color::White),
            attackers
        );
        assert!(!is_cell_attacked(
            &b,
            Coord::from_parts(File::D, Rank::R4),
            Color::Black
        ));
        assert_eq!(
            cell_attackers(&b, Coord::from_parts(File::D, Rank::R4), Color::Black),
            Bitboard::EMPTY
        );

        let b = Board::from_fen("8/8/8/2KPk3/8/8/8/8 w - - 0 1").unwrap();
        assert!(is_cell_attacked(
            &b,
            Coord::from_parts(File::D, Rank::R5),
            Color::White
        ));
        assert_eq!(
            cell_attackers(&b, Coord::from_parts(File::D, Rank::R5), Color::White),
            Bitboard::from_coord(Coord::from_parts(File::C, Rank::R5)),
        );
        assert!(is_cell_attacked(
            &b,
            Coord::from_parts(File::D, Rank::R5),
            Color::Black
        ));
        assert_eq!(
            cell_attackers(&b, Coord::from_parts(File::D, Rank::R5), Color::Black),
            Bitboard::from_coord(Coord::from_parts(File::E, Rank::R5)),
        );
    }

    #[test]
    fn test_san_candidates() {
        let b = Board::from_fen("3R3B/B7/1B1R4/1N2Q3/RQ1p4/1N6/5B2/3R1K1k w - - 0 1").unwrap();
        let d4 = Coord::from_parts(File::D, Rank::R4);

        let mut ml = MoveList::new();
        san_candidates(&b, Piece::Knight, d4, &mut ml);
        assert_eq!(
            ml.iter().map(ToString::to_string).collect::<BTreeSet<_>>(),
            BTreeSet::from(["b3d4".to_string(), "b5d4".to_string()]),
        );

        let mut ml = MoveList::new();
        san_candidates(&b, Piece::Bishop, d4, &mut ml);
        assert_eq!(
            ml.iter().map(ToString::to_string).collect::<BTreeSet<_>>(),
            BTreeSet::from(["b6d4".to_string(), "f2d4".to_string()]),
        );

        let mut ml = MoveList::new();
        san_candidates(&b, Piece::Rook, d4, &mut ml);
        assert_eq!(
            ml.iter().map(ToString::to_string).collect::<BTreeSet<_>>(),
            BTreeSet::from(["d1d4".to_string(), "d6d4".to_string()]),
        );

        let mut ml = MoveList::new();
        san_candidates(&b, Piece::Queen, d4, &mut ml);
        assert_eq!(
            ml.iter().map(ToString::to_string).collect::<BTreeSet<_>>(),
            BTreeSet::from(["b4d4".to_string(), "e5d4".to_string()]),
        );

        let mut ml = MoveList::new();
        san_candidates(&b, Piece::King, d4, &mut ml);
        assert_eq!(
            ml.iter().map(ToString::to_string).collect::<BTreeSet<_>>(),
            BTreeSet::new(),
        );

        let mut ml = MoveList::new();
        san_candidates(
            &b,
            Piece::King,
            Coord::from_parts(File::E, Rank::R2),
            &mut ml,
        );
        assert_eq!(
            ml.iter().map(ToString::to_string).collect::<BTreeSet<_>>(),
            BTreeSet::from(["f1e2".to_string()]),
        );
    }

    #[test]
    fn test_san_pawn_capture_candidates() {
        let b = Board::from_fen("7n/6P1/8/2PpP2r/2P1P1P1/7q/6P1/3k1K2 w - d6 0 1").unwrap();

        let mut ml = MoveList::new();
        san_pawn_capture_candidates(&b, File::C, File::D, None, &mut ml);
        assert_eq!(
            ml.iter().map(ToString::to_string).collect::<BTreeSet<_>>(),
            BTreeSet::from(["c4d5".to_string(), "c5d6".to_string()]),
        );

        let mut ml = MoveList::new();
        san_pawn_capture_candidates(&b, File::E, File::D, None, &mut ml);
        assert_eq!(
            ml.iter().map(ToString::to_string).collect::<BTreeSet<_>>(),
            BTreeSet::from(["e4d5".to_string(), "e5d6".to_string()]),
        );

        let mut ml = MoveList::new();
        san_pawn_capture_candidates(&b, File::E, File::D, Some(PromotePiece::Knight), &mut ml);
        assert_eq!(
            ml.iter().map(ToString::to_string).collect::<BTreeSet<_>>(),
            BTreeSet::new(),
        );

        let mut ml = MoveList::new();
        san_pawn_capture_candidates(&b, File::G, File::H, None, &mut ml);
        assert_eq!(
            ml.iter().map(ToString::to_string).collect::<BTreeSet<_>>(),
            BTreeSet::from(["g2h3".to_string(), "g4h5".to_string()]),
        );

        let mut ml = MoveList::new();
        san_pawn_capture_candidates(&b, File::G, File::H, Some(PromotePiece::Knight), &mut ml);
        assert_eq!(
            ml.iter().map(ToString::to_string).collect::<BTreeSet<_>>(),
            BTreeSet::from(["g7h8n".to_string()]),
        );
    }

    #[test]
    fn test_has_legal_moves() {
        let b = Board::initial();
        assert!(has_legal_moves(&b));

        let b =
            Board::from_fen("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
                .unwrap();
        assert!(!has_legal_moves(&b));

        let b = Board::from_fen("K7/8/2n5/2n2p1p/5P1P/8/8/5k2 w - - 0 1").unwrap();
        assert!(!has_legal_moves(&b));
    }

    #[test]
    fn test_simple() {
        let b = Board::initial();
        assert_eq!(semilegal::gen_all(&b).len(), 20);
        assert_eq!(semilegal::gen_simple(&b).len(), 20);
        assert_eq!(semilegal::gen_capture(&b).len(), 0);
        assert_eq!(legal::gen_all(&b).len(), 20);
        assert_eq!(legal::gen_simple(&b).len(), 20);
        assert_eq!(legal::gen_capture(&b).len(), 0);

        let b = Board::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
            .unwrap();
        assert_eq!(semilegal::gen_all(&b).len(), 27);
        assert_eq!(semilegal::gen_simple(&b).len(), 26);
        assert_eq!(semilegal::gen_capture(&b).len(), 1);
        assert_eq!(legal::gen_all(&b).len(), 27);
        assert_eq!(legal::gen_simple(&b).len(), 26);
        assert_eq!(legal::gen_capture(&b).len(), 1);

        let b = Board::from_fen("6kb/R7/8/4P3/8/1p6/1K6/2r4r w - - 0 1").unwrap();
        assert_eq!(semilegal::gen_all(&b).len(), 23);
        assert_eq!(semilegal::gen_simple(&b).len(), 21);
        assert_eq!(semilegal::gen_capture(&b).len(), 2);
        assert_eq!(legal::gen_all(&b).len(), 16);
        assert_eq!(legal::gen_simple(&b).len(), 15);
        assert_eq!(legal::gen_capture(&b).len(), 1);
    }
}
