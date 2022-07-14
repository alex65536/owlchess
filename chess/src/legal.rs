use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::moves::{Move, MoveKind};
use crate::types::{Color, Coord, Piece};
use crate::{attack, between, geometry};

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
    NotCheck {
        pinned_or_king: Bitboard,
    },
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
    side: Color,
    king: Coord,
    all: Bitboard,
    opponent_pieces: [Bitboard; Piece::COUNT],
}

struct Undo(Option<Piece>);

impl<'a, P: Prechecker> Checker<'a, P> {
    pub fn new(src: &'a Board, pre: P) -> Self {
        let side = src.r.side;
        let inv = side.inv();
        Self {
            src,
            pre,
            side,
            king: src.king_pos(side),
            all: src.all,
            opponent_pieces: [
                src.piece2(inv, Piece::Pawn),
                src.piece2(inv, Piece::King),
                src.piece2(inv, Piece::Knight),
                src.piece2(inv, Piece::Bishop),
                src.piece2(inv, Piece::Rook),
                src.piece2(inv, Piece::Queen),
            ],
        }
    }

    fn opponent(&self, p: Piece) -> Bitboard {
        unsafe { *self.opponent_pieces.get_unchecked(p.index()) }
    }

    fn opponent_mut(&mut self, p: Piece) -> &mut Bitboard {
        unsafe { self.opponent_pieces.get_unchecked_mut(p.index()) }
    }

    fn diag_pieces(&self) -> Bitboard {
        self.opponent(Piece::Bishop) | self.opponent(Piece::Queen)
    }

    fn line_pieces(&self) -> Bitboard {
        self.opponent(Piece::Rook) | self.opponent(Piece::Queen)
    }

    fn is_attacked(&self, pos: Coord) -> bool {
        // Here, we use black attack map for white, as we need to trace the attack from destination piece,
        // not from the source one. For example, if we are white (i.e. `self.side == Color::White`), then
        // we are attacked by black and thus need to use white attack map.
        let pawn_attacks = attack::pawn(self.side, pos);

        // Near attacks
        if (self.opponent(Piece::Pawn) & pawn_attacks).is_nonempty()
            || (self.opponent(Piece::King) & attack::king(pos)).is_nonempty()
            || (self.opponent(Piece::Knight) & attack::knight(pos)).is_nonempty()
        {
            return true;
        }

        // Far attacks
        (attack::bishop(pos, self.all) & self.diag_pieces()).is_nonempty()
            || (attack::rook(pos, self.all) & self.line_pieces()).is_nonempty()
    }

    fn is_king_attacked(&self) -> bool {
        self.is_attacked(self.king)
    }

    fn do_make_enpassant(&mut self, dst: Coord) {
        let taken_pos = unsafe { dst.add_unchecked(-geometry::pawn_forward_delta(self.side)) };
        let taken = Bitboard::from_coord(taken_pos);
        self.all ^= taken;
        *self.opponent_mut(Piece::Pawn) ^= taken;
    }

    fn do_make_move(&mut self, mv: Move) {
        match mv.kind() {
            MoveKind::Enpassant => self.do_make_enpassant(mv.dst()),
            MoveKind::Simple
            | MoveKind::PawnSimple
            | MoveKind::PawnDouble
            | MoveKind::PromoteKnight
            | MoveKind::PromoteBishop
            | MoveKind::PromoteRook
            | MoveKind::PromoteQueen => {}
            MoveKind::CastlingKingside | MoveKind::CastlingQueenside | MoveKind::Null => {
                unreachable!()
            }
        }
    }

    fn make_move(&mut self, mv: Move) -> Undo {
        let p = self.src.get(mv.dst()).piece();
        let src = Bitboard::from_coord(mv.src());
        let dst = Bitboard::from_coord(mv.dst());
        self.all ^= src;
        self.all |= dst;
        if let Some(p) = p {
            *self.opponent_mut(p) ^= dst;
        }
        self.do_make_move(mv);
        Undo(p)
    }

    fn unmake_move(&mut self, mv: Move, u: Undo) {
        let src = Bitboard::from_coord(mv.src());
        let dst = Bitboard::from_coord(mv.dst());
        self.all ^= src;
        if let Some(p) = u.0 {
            *self.opponent_mut(p) ^= dst;
        } else {
            self.all ^= dst;
        }
        self.do_make_move(mv);
    }

    #[inline]
    pub fn is_legal(&mut self, mv: Move) -> bool {
        if let Some(ok) = self.pre.is_legal_pre(mv) {
            return ok;
        }

        if mv.src() == self.king {
            let src = Bitboard::from_coord(mv.src());
            self.all ^= src;
            let res = !self.is_attacked(mv.dst());
            self.all ^= src;
            return res;
        }

        let u = self.make_move(mv);
        let res = !self.is_king_attacked();
        self.unmake_move(mv, u);
        res
    }
}
