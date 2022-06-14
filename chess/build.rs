use std::io::{self, BufWriter, Write};
use std::{env, fs, path::Path};

use owlchess_base::geometry;
use owlchess_base::types::{Cell, Color, Coord, File, Piece};
use rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

struct Zobrist {
    pieces: [[u64; 64]; Cell::MAX_INDEX],
    move_side: u64,
    castling: [u64; 16],
    enpassant: [u64; 64],
    castling_kingside: [u64; 2],
    castling_queenside: [u64; 2],
}

impl Zobrist {
    fn generate<R: RngCore>(gen: &mut R) -> Zobrist {
        let pieces = {
            let mut res = [[0_u64; 64]; Cell::MAX_INDEX];
            for sub in res.iter_mut().skip(1) {
                for x in sub {
                    *x = gen.next_u64();
                }
            }
            res
        };
        Zobrist {
            pieces,
            move_side: gen.next_u64(),
            castling: [(); 16].map(|_| gen.next_u64()),
            enpassant: [(); 64].map(|_| gen.next_u64()),
            castling_kingside: [Color::White, Color::Black].map(|c| {
                let rook = Cell::from_parts(c, Piece::Rook);
                let king = Cell::from_parts(c, Piece::King);
                let rank = geometry::castling_rank(c);
                pieces[king.index()][Coord::from_parts(File::E, rank).index()]
                    ^ pieces[king.index()][Coord::from_parts(File::G, rank).index()]
                    ^ pieces[rook.index()][Coord::from_parts(File::H, rank).index()]
                    ^ pieces[rook.index()][Coord::from_parts(File::F, rank).index()]
            }),
            castling_queenside: [Color::White, Color::Black].map(|c| {
                let rook = Cell::from_parts(c, Piece::Rook);
                let king = Cell::from_parts(c, Piece::King);
                let rank = geometry::castling_rank(c);
                pieces[king.index()][Coord::from_parts(File::E, rank).index()]
                    ^ pieces[king.index()][Coord::from_parts(File::C, rank).index()]
                    ^ pieces[rook.index()][Coord::from_parts(File::A, rank).index()]
                    ^ pieces[rook.index()][Coord::from_parts(File::D, rank).index()]
            }),
        }
    }

    fn generate_default() -> Zobrist {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x800D_BA5E_5EED_1234_u64);
        Self::generate(&mut rng)
    }

    fn output<W: Write>(&self, w: &mut W) -> io::Result<()> {
        writeln!(w, "pub const PIECES: [[u64; 64]; Cell::MAX_INDEX] = [")?;
        for (i, sub) in self.pieces.iter().enumerate() {
            writeln!(w, "    /*{:2}*/ [", i)?;
            for (i, hsh) in sub.iter().enumerate() {
                writeln!(w, "        /*{:2}*/ {:#x},", i, hsh)?;
            }
            writeln!(w, "    ],")?;
        }
        writeln!(w, "];\n")?;

        writeln!(w, "pub const MOVE_SIDE: u64 = {:#x};\n", self.move_side)?;

        writeln!(w, "pub const CASTLING: [u64; 16] = [")?;
        for (i, sub) in self.castling.iter().enumerate() {
            writeln!(w, "    /*{:2}*/ {:#x},", i, sub)?;
        }
        writeln!(w, "];\n")?;

        writeln!(w, "pub const ENPASSANT: [u64; 64] = [")?;
        for (i, sub) in self.enpassant.iter().enumerate() {
            writeln!(w, "    /*{:2}*/ {:#x},", i, sub)?;
        }
        writeln!(w, "];\n")?;

        writeln!(
            w,
            "pub const CASTLING_KINGSIDE: [u64; 2] = [{:#x}, {:#x}];",
            self.castling_kingside[0], self.castling_kingside[1]
        )?;
        writeln!(
            w,
            "pub const CASTLING_QUEENSIDE: [u64; 2] = [{:#x}, {:#x}];",
            self.castling_queenside[0], self.castling_queenside[1]
        )?;

        Ok(())
    }
}

fn gen_zobrist(out_path: &Path) -> io::Result<()> {
    Zobrist::generate_default().output(&mut BufWriter::new(&mut fs::File::create(out_path)?))?;
    Ok(())
}

fn main() -> io::Result<()> {
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = env::var("OUT_DIR").unwrap();

    gen_zobrist(&Path::new(&out_dir).join("zobrist.rs"))?;

    Ok(())
}
