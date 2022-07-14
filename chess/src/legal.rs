use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::moves::{Move, MoveKind};
use crate::types::{Color, Coord, Piece};
use crate::{attack, geometry};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checker<'a> {
    src: &'a Board,
    side: Color,
    king: Coord,
    all: Bitboard,
    opponent_pieces: [Bitboard; Piece::COUNT],
}

impl<'a> From<&'a Board> for Checker<'a> {
    fn from(src: &'a Board) -> Self {
        let side = src.r.side;
        let inv = side.inv();
        Self {
            src,
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
}

struct Undo(Option<Piece>);

impl<'a> Checker<'a> {
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

    pub fn is_legal(&mut self, mv: Move) -> bool {
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
