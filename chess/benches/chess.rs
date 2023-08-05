use criterion::{black_box, criterion_group, criterion_main, Criterion};
use owlchess::{
    movegen::{self, semilegal},
    moves, Board, Color, Coord, Make,
};

const BOARDS: [(&'static str, &'static str); 10] = [
    (
        "initial",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ),
    (
        "sicilian",
        "r1b1k2r/2qnbppp/p2ppn2/1p4B1/3NPPP1/2N2Q2/PPP4P/2KR1B1R w kq - 0 11",
    ),
    (
        "middle",
        "1rq1r1k1/1p3ppp/pB3n2/3ppP2/Pbb1P3/1PN2B2/2P2QPP/R1R4K w - - 1 21",
    ),
    (
        "open_position",
        "4r1k1/3R1ppp/8/5P2/p7/6PP/4pK2/1rN1B3 w - - 4 43",
    ),
    ("queen", "6K1/8/8/1k3q2/3Q4/8/8/8 w - - 0 1"),
    ("pawn_move", "4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1"),
    ("pawn_attack", "4k3/8/8/pppppppp/PPPPPPPP/8/8/4K3 w - - 0 1"),
    (
        "pawn_promote",
        "8/PPPPPPPP/8/2k1K3/8/8/pppppppp/8 w - - 0 1",
    ),
    (
        "cydonia",
        "5K2/1N1N1N2/8/1N1N1N2/1n1n1n2/8/1n1n1n2/5k2 w - - 0 1",
    ),
    (
        "max",
        "3Q4/1Q4Q1/4Q3/2Q4R/Q4Q2/3Q4/NR4Q1/kN1BB1K1 w - - 0 1",
    ),
];

fn boards() -> impl Iterator<Item = (&'static str, Board)> {
    BOARDS
        .iter()
        .map(|&(name, fen)| (name, Board::from_fen(fen).unwrap()))
}

fn bench_gen_moves(c: &mut Criterion) {
    let mut group = c.benchmark_group("gen_moves");
    for (name, board) in boards() {
        group.bench_function(name, |b| {
            b.iter(|| black_box(semilegal::gen_all(&board).len()))
        });
    }
}

fn bench_make_move(c: &mut Criterion) {
    let mut group = c.benchmark_group("make_move");
    for (name, mut board) in boards() {
        let moves = semilegal::gen_all(&board);
        group.bench_function(name, |b| {
            b.iter(|| {
                for mv in &moves {
                    unsafe {
                        let u = moves::make_move_unchecked(&mut board, *mv);
                        moves::unmake_move_unchecked(&mut board, *mv, u)
                    }
                }
            })
        });
    }
}

fn bench_make_move_checked(c: &mut Criterion) {
    let mut group = c.benchmark_group("make_move_checked");
    for (name, board) in boards() {
        let moves = semilegal::gen_all(&board);
        group.bench_function(name, |b| {
            b.iter(|| {
                for mv in &moves {
                    black_box(
                        mv.make(&board)
                            .as_ref()
                            .map(|b| b.zobrist_hash())
                            .unwrap_or(0),
                    );
                }
            })
        });
    }
}

fn bench_is_move_semilegal(c: &mut Criterion) {
    let mut group = c.benchmark_group("is_move_semilegal");
    for (name, board) in boards() {
        let moves = semilegal::gen_all(&board);
        group.bench_function(name, |b| {
            b.iter(|| {
                for mv in &moves {
                    black_box(mv.is_semilegal(&board));
                }
            })
        });
    }
}

fn bench_is_attacked(c: &mut Criterion) {
    let mut group = c.benchmark_group("is_attacked");
    for (name, board) in boards() {
        group.bench_function(name, |b| {
            b.iter(|| {
                for color in [Color::White, Color::Black] {
                    for coord in Coord::iter() {
                        black_box(movegen::is_cell_attacked(&board, coord, color));
                    }
                }
            })
        });
    }
}

fn bench_king_attack(c: &mut Criterion) {
    let mut group = c.benchmark_group("king_attack");
    for (name, board) in boards() {
        group.bench_function(name, |b| {
            b.iter(|| black_box(board.is_opponent_king_attacked()))
        });
    }
}

fn bench_has_legal_moves(c: &mut Criterion) {
    let mut group = c.benchmark_group("has_legal_moves");
    for (name, board) in boards() {
        group.bench_function(name, |b| b.iter(|| black_box(board.has_legal_moves())));
    }
}

criterion_group!(
    chess,
    bench_gen_moves,
    bench_make_move,
    bench_make_move_checked,
    bench_is_move_semilegal,
    bench_is_attacked,
    bench_king_attack,
    bench_has_legal_moves,
);

criterion_main!(chess);
