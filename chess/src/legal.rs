use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::moves::{Move, MoveKind};
use crate::types::{Color, Coord, Piece};
use crate::{attack, between, pawns};

pub trait Prechecker {
    fn is_legal_pre(&self, mv: Move) -> Option<bool>;
}

#[derive(Clone, Debug)]
pub struct NilPrechecker;

impl Prechecker for NilPrechecker {
    #[inline]
    fn is_legal_pre(&self, _mv: Move) -> Option<bool> {
        None
    }
}

#[derive(Clone, Debug)]
enum PrecheckData {
    Check,
    NotCheck { pinned_or_king: Bitboard },
}

#[derive(Clone, Debug)]
pub struct DefaultPrechecker(PrecheckData);

impl DefaultPrechecker {
    fn bishop_xray(b: &Board, ours: Bitboard, king: Coord) -> Bitboard {
        let near = attack::bishop(king, b.all) & ours;
        attack::bishop(king, b.all ^ near)
    }

    fn rook_xray(b: &Board, ours: Bitboard, king: Coord) -> Bitboard {
        let near = attack::rook(king, b.all) & ours;
        attack::rook(king, b.all ^ near)
    }

    fn pinned(b: &Board, side: Color, king: Coord) -> Bitboard {
        let mut pinned = Bitboard::EMPTY;
        let ours = b.color(side);

        let pinners = Self::bishop_xray(b, ours, king) & b.piece_diag(side.inv());
        for p in pinners {
            pinned |= between::bishop_strict(p, king) & ours;
        }

        let pinners = Self::rook_xray(b, ours, king) & b.piece_line(side.inv());
        for p in pinners {
            pinned |= between::rook_strict(p, king) & ours;
        }

        pinned
    }

    pub fn new(b: &Board) -> DefaultPrechecker {
        if b.is_check() {
            return DefaultPrechecker(PrecheckData::Check);
        }

        let side = b.r.side;
        let king = b.king_pos(side);
        DefaultPrechecker(PrecheckData::NotCheck {
            pinned_or_king: Self::pinned(b, side, king) | Bitboard::from_coord(king),
        })
    }
}

impl Prechecker for DefaultPrechecker {
    #[inline]
    fn is_legal_pre(&self, mv: Move) -> Option<bool> {
        match self.0 {
            PrecheckData::Check => {
                // Check evasions need to be handled separately. We could consider the different
                // cases and detect legality right here (bypassing `Checker`), but those things
                // turn out to slow down the implementation on perft. So, just return `None` and
                // remember that checks are usually rare.
                None
            }
            PrecheckData::NotCheck { pinned_or_king } => {
                if !pinned_or_king.has(mv.src()) {
                    // The piece is not pinned and is not a king, so the move is definitely legal.
                    Some(true)
                } else {
                    None
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Checker<'a, P> {
    src: &'a Board,
    pre: P,
    inv: Color,
    king: Coord,
}

impl<'a, P: Prechecker> Checker<'a, P> {
    pub fn new(src: &'a Board, pre: P) -> Self {
        let side = src.r.side;
        Self {
            src,
            pre,
            inv: side.inv(),
            king: src.king_pos(side),
        }
    }

    fn is_attacked(&self, pos: Coord, all: Bitboard, mask: Bitboard) -> bool {
        let inv = self.inv;
        let b = self.src;

        // Here, we use black attack map for white, as we need to trace the attack from destination piece,
        // not from the source one. For example, if we are white (i.e. `self.inv == Color::Black`), then
        // we are attacked by black and thus need to use white attack map.
        let pawn_attacks = attack::pawn(inv.inv(), pos);

        // Near attacks
        let near_attackers = (b.piece2(inv, Piece::Pawn) & pawn_attacks)
            | (b.piece2(inv, Piece::King) & attack::king(pos))
            | (b.piece2(inv, Piece::Knight) & attack::knight(pos));
        if (near_attackers & mask).is_nonempty() {
            return true;
        }

        // Far attacks
        (attack::bishop(pos, all) & b.piece_diag(inv) & mask).is_nonempty()
            || (attack::rook(pos, all) & b.piece_line(inv) & mask).is_nonempty()
    }

    #[inline]
    pub fn is_legal(&self, mv: Move) -> bool {
        if let Some(ok) = self.pre.is_legal_pre(mv) {
            return ok;
        }

        let src = Bitboard::from_coord(mv.src());
        let dst = Bitboard::from_coord(mv.dst());

        if mv.src() == self.king {
            return !self.is_attacked(mv.dst(), self.src.all ^ src, Bitboard::FULL);
        }

        let all = (self.src.all ^ src) | dst;
        let mask = !dst;
        if mv.kind() == MoveKind::Enpassant {
            let tmp = pawns::advance_forward(self.inv, dst);
            return !self.is_attacked(self.king, all ^ tmp, mask ^ tmp);
        }
        !self.is_attacked(self.king, all, mask)
    }
}
