// Simple command-line application to play chess

use owlchess::{
    board::PrettyStyle,
    chain::{GameStatusPolicy, NumberPolicy},
    moves::Style,
    types::OutcomeFilter,
    Color, Move, MoveChain, GameStatus,
};
use std::io::{self, BufRead, Write};

fn main() {
    let mut stdin = io::stdin().lock();

    let mut chain = MoveChain::new_initial();

    loop {
        if let Some(outcome) = chain.outcome() {
            println!("Game finished: {}, {}", GameStatus::from(outcome), outcome);
            println!("Notation:");
            println!(
                "{}",
                chain.styled(NumberPolicy::FromBoard, Style::San, GameStatusPolicy::Show)
            );
            break;
        }

        println!("{}", chain.last().pretty(PrettyStyle::Ascii));
        let side = match chain.last().side() {
            Color::White => "White",
            Color::Black => "Black",
        };
        print!("{} move ({}): ", side, chain.last().raw().move_number);
        io::stdout().flush().unwrap();
        let mut s = String::new();
        stdin.read_line(&mut s).unwrap();
        let s = s.trim();

        // Note that we could just call `MoveChain::push_san()` here directly. But we want
        // to show more features here, so separate parsing a move and making it.
        let mv = match Move::from_san(s, chain.last()) {
            Ok(mv) => mv,
            Err(e) => {
                println!("Bad move: {}", e);
                println!("");
                continue;
            }
        };

        // Move is definitely legal after `Move::from_san()`, so just `unwrap()` instead of
        // error checking.
        chain.push(mv).unwrap();

        println!("");

        // Check whether the game must be terminated.
        chain.set_auto_outcome(OutcomeFilter::Strict);
    }
}
