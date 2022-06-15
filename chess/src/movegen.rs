use crate::{attack, generic};
use crate::board::Board;
use crate::bitboard::Bitboard;
use crate::types::{Color, Coord, Piece};

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
    if (b.piece2(C::COLOR, Piece::Pawn) & pawn_attacks).is_nonempty() ||
        (b.piece2(C::COLOR, Piece::King) & attack::king(coord)).is_nonempty() ||
        (b.piece2(C::COLOR, Piece::Knight) & attack::knight(coord)).is_nonempty() {
        return true;
    }

    // Far attacks
    (attack::bishop(coord, b.all) & diag_pieces(b, C::COLOR)).is_nonempty() ||
        (attack::rook(coord, b.all) & line_pieces(b, C::COLOR)).is_nonempty()
}

pub fn is_cell_attacked(b: &Board, coord: Coord, color: Color) -> bool {
    match color {
        Color::White => do_is_cell_attacked::<generic::White>(b, coord),
        Color::Black => do_is_cell_attacked::<generic::Black>(b, coord),
    }
}

// TODO
// TODO tests
