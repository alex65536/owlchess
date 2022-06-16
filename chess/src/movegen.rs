use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::moves::{self, Move, MoveKind};
use crate::types::{CastlingSide, Cell, Color, Coord, File, Piece};
use crate::{attack, bitboard_consts, castling, generic, geometry, pawns};

use std::convert::Infallible;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use arrayvec::ArrayVec;

fn diag_pieces(b: &Board, c: Color) -> Bitboard {
    b.piece2(c, Piece::Bishop) | b.piece2(c, Piece::Queen)
}

fn line_pieces(b: &Board, c: Color) -> Bitboard {
    b.piece2(c, Piece::Rook) | b.piece2(c, Piece::Queen)
}

fn do_is_cell_attacked<C: generic::Color>(b: &Board, coord: Coord) -> bool {
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

pub fn is_cell_attacked(b: &Board, coord: Coord, color: Color) -> bool {
    match color {
        Color::White => do_is_cell_attacked::<generic::White>(b, coord),
        Color::Black => do_is_cell_attacked::<generic::Black>(b, coord),
    }
}

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

impl MoveList {
    pub fn new() -> MoveList {
        MoveList(ArrayVec::new())
    }

    pub fn filter_legal(&mut self, b: &Board) {
        let mut b_copy = b.clone();
        self.retain(|&mut mv| unsafe {
            let u = moves::make_move_unchecked(&mut b_copy, mv);
            let ok = !b_copy.is_opponent_king_attacked();
            moves::unmake_move_unchecked(&mut b_copy, mv, u);
            ok
        });
    }
}

pub trait MovePush {
    fn push(&mut self, m: Move);
}

impl<const N: usize> MovePush for ArrayVec<Move, N> {
    fn push(&mut self, m: Move) {
        self.push(m);
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

struct MoveGenImpl<'a, C, P> {
    board: &'a Board,
    dst: &'a mut P,
    _c: PhantomData<C>,
}

impl<'a, C: generic::Color, P: MaybeMovePush> MoveGenImpl<'a, C, P> {
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
                unsafe { self.add_move(MoveKind::Enpassant, left_pawn, dst)? };
            }
            if file != File::H && self.board.get(right_pawn) == pawn {
                unsafe { self.add_move(MoveKind::Enpassant, right_pawn, dst)? };
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
                unsafe { self.add_move(MoveKind::Simple, src, dst)? };
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
                unsafe { self.add_move(MoveKind::Simple, src, dst)? };
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
                unsafe { self.add_move(MoveKind::CastlingKingside, src, dst) }?;
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
                unsafe { self.add_move(MoveKind::CastlingQueenside, src, dst) }?;
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

pub mod semilegal {
    use super::{MoveGenImpl, MoveList, MovePush};
    use crate::{board::Board, moves::Move, generic, types::Color};

    struct UnsafeMoveList(MoveList);

    impl MovePush for UnsafeMoveList {
        fn push(&mut self, m: Move) {
            unsafe { self.0.push_unchecked(m); }
        }
    }

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
                    let mut res = UnsafeMoveList(MoveList::new());
                    $name_into(b, &mut res);
                    res.0
                }
            )*
        }
    }

    do_impl! {
        gen_all;
        gen_all_into;

        gen_capture;
        gen_capture_into;

        gen_simple;
        gen_simple_into;

        gen_simple_no_promote;
        gen_simple_no_promote_into;

        gen_simple_promote;
        gen_simple_promote_into;
    }
}

pub mod legal {
    use super::MoveList;
    use crate::board::Board;

    macro_rules! do_impl {
        ($($(#[$attr:meta])* $name:ident;)*) => {
            $(
                $(#[$attr])*
                pub fn $name(b: &Board) -> MoveList {
                    let mut res = super::semilegal::$name(b);
                    res.filter_legal(b);
                    res
                }
            )*
        }
    }

    do_impl!{
        gen_all;
        gen_capture;
        gen_simple;
        gen_simple_no_promote;
        gen_simple_promote;
    }
}

struct LegalChecker {
    b: Board,
}

impl MaybeMovePush for LegalChecker {
    type Err = ();

    fn push(&mut self, mv: Move) -> Result<(), ()> {
        let u = unsafe { moves::make_move_unchecked(&mut self.b, mv) };
        let is_legal = !self.b.is_opponent_king_attacked();
        unsafe { moves::unmake_move_unchecked(&mut self.b, mv, u) };
        match is_legal {
            true => Err(()),
            false => Ok(()),
        }
    }
}

pub fn has_legal_moves(b: &Board) -> bool {
    let mut c = LegalChecker{b: b.clone()};
    (match b.r.side {
        Color::White => MoveGenImpl::new(b, &mut c, generic::White).gen_all_for_detect(),
        Color::Black => MoveGenImpl::new(b, &mut c, generic::Black).gen_all_for_detect(),
    }).is_err()
}

// TODO tests
