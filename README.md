# Owlchess 🦉🦀

[![Crates.io: owlchess](https://img.shields.io/crates/v/owlchess.svg)](https://crates.io/crates/owlchess)
[![Documentation](https://img.shields.io/docsrs/owlchess/latest)](https://docs.rs/owlchess)
[![Build](https://github.com/alex65536/owlchess/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/alex65536/owlchess/actions/workflows/build.yml)

Yet another chess crate for Rust, with emphasis on speed and safety. Primarily designed for various
chess GUIs and tools, it's also possible to use Owlchess to build a fast chess engine.

The code is mostly derived from my chess engine [SoFCheck](https://github.com/alex65536/sofcheck),
but rewritten in Rust with regard to safety.

This crate supports core chess functionality:

- generate moves
- make moves
- calculate game outcome
- parse and format boards in [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation)
- parse and format moves in UCI and [SAN](https://en.wikipedia.org/wiki/Algebraic_notation_\(chess\))

## Features

_Fast_: chessboard is built upon [Magic Bitboards](https://www.chessprogramming.org/Magic_Bitboards),
which is a fast way to generate moves and determine whether the king is in check.

_Safe_: the library prevents you from creating an invalid board or making an invalid move. While such
safety is usually a good thing, it is enforces by runtime checks, which can slow down your program. For
example, validation is `owlchess::moves::make_move` makes this function about 30-50% slower. So, if
performance really matters, you may use unsafe APIs for speedup.

## Examples

### Generating moves

```rust
use owlchess::{Board, movegen::legal};

fn main() {
    // Create a board from FEN
    let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        .unwrap();

    // Generate legal moves
    let moves = legal::gen_all(&board);
    assert_eq!(moves.len(), 20);
}
```

### Making moves from UCI notation

```rust
use owlchess::{Board, Move};

fn main() {
    // Create a board with initial position
    let board = Board::initial();

    // Create a legal move from UCI notation
    let mv = Move::from_uci_legal("e2e4", &board).unwrap();

    // Create a new board with move `mv` made on it
    let board = board.make_move(mv).unwrap();
    assert_eq!(
        board.as_fen(),
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
            .to_string(),
    );
}
```

### Playing games

The example below illustrates a `MoveChain`, which represents a chess game. Unlike `Board`, `MoveChain` keeps
the history of moves and is able to detect draw by repetitions.

```rust
use owlchess::{Outcome, types::OutcomeFilter, Color, WinReason, MoveChain};

fn main() {
    // Create a `MoveChain` from initial position
    let mut chain = MoveChain::new_initial();

    // Push the moves into `MoveChain` as UCI strings
    chain.push_uci("g2g4").unwrap();
    chain.push_uci("e7e5").unwrap();
    chain.push_uci("f2f3").unwrap();
    chain.push_uci("d8h4").unwrap();

    // Calculate current game outcome
    chain.set_auto_outcome(OutcomeFilter::Strict);
    assert_eq!(
        chain.outcome(),
        &Some(Outcome::Win {
            side: Color::Black,
            reason: WinReason::Checkmate,
        }),
    );
}
```

### Other

Some examples are located in the [`chess/examples`](chess/examples) directory and crate documentation.
They may give you more ideas on how to use the crate.

## Rust version

This crate is currently tested only with Rust 1.61 or higher, but can possibly work with older versions.
Rust versions before 1.51 are definitely not supported, as we use [`arrayvec`](https://github.com/bluss/arrayvec)
as dependency.

## Comparison with other crates

There are two well-known chess crates in Rust: [`chess`](https://github.com/jordanbray/chess) and
[`shakmaty`](https://github.com/niklasf/shakmaty). These crates are compared with `owlchess` below.

### Comparison with `chess`

Compared to `chess`, `owlchess` provides more features:

- distinction between various game outcomes
- draw by insufficient material
- formatting SAN moves
- more safety: the board is guaranteed to be valid, as you cannot make an illegal move without `unsafe`
- more detailed errors in case of failures

Still, `chess` has some advantages:

- faster legal move generator (`owlchess` relies on fast semilegal move generator instead)
- more mature code base

But, in many applications, it's enough to have a fast semilegal move generator and fast move validation, so
real difference in performance may be less than expected.

### Comparison with `shakmaty`

Compared to `shakmaty`, `owlchess` has the following advantages:

- more safety: the board is guaranteed to be valid, as you cannot make an illegal move without `unsafe`
- has support for draw by repetitions
- uses more liberal MIT license instead of GPLv3

Still, `shakmaty` has some advantages:

- faster legal move generator
- moves are applied faster (especially if Zobrist hashing is not used)
- supports many chess variants

### Benchmarks

There is a [separate repo](https://github.com/alex65536/chess_bench) with benchmarks for different
chess implementations in Rust.

## License

This repository is licensed under the MIT License. See [`LICENSE`](LICENSE) for more details.
