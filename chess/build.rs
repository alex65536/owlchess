use std::path::Path;
use std::{env, io};

use owlchess_base::bitboard::Bitboard;

use rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

pub fn default_gen() -> impl RngCore {
    Xoshiro256PlusPlus::seed_from_u64(0x800D_BA5E_5EED_1234_u64)
}

fn print_bitboards<W: io::Write>(w: &mut W, name: &str, bs: &[Bitboard]) -> io::Result<()> {
    writeln!(w, "const {}: [Bitboard; {}] = [", name, bs.len())?;
    for (i, b) in bs.iter().enumerate() {
        writeln!(w, "    /*{:2}*/ bb(0x{:016x}),", i, b.as_raw())?;
    }
    writeln!(w, "];")?;
    Ok(())
}

mod zobrist {
    use std::io::{self, BufWriter, Write};
    use std::{fs, path::Path};

    use owlchess_base::geometry;
    use owlchess_base::types::{Cell, Color, Coord, File, Piece};
    use rand_core::RngCore;

    struct Zobrist {
        pieces: [[u64; 64]; Cell::COUNT],
        move_side: u64,
        castling: [u64; 16],
        enpassant: [u64; 64],
        castling_kingside: [u64; 2],
        castling_queenside: [u64; 2],
    }

    impl Zobrist {
        fn generate<R: RngCore>(gen: &mut R) -> Zobrist {
            let pieces = {
                let mut res = [[0_u64; 64]; Cell::COUNT];
                for sub in res.iter_mut().skip(1) {
                    for x in sub {
                        *x = gen.next_u64();
                    }
                }
                res
            };
            let castling = {
                let base = [(); 4].map(|_| gen.next_u64());
                let mut res = [0_u64; 16];
                for (i, val) in res.iter_mut().enumerate() {
                    for (j, base_val) in base.iter().enumerate() {
                        if (i >> j) & 1 != 0 {
                            *val ^= base_val;
                        }
                    }
                }
                res
            };
            Zobrist {
                pieces,
                move_side: gen.next_u64(),
                castling,
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
            Self::generate(&mut super::default_gen())
        }

        fn output<W: Write>(&self, w: &mut W) -> io::Result<()> {
            writeln!(w, "const PIECES: [[u64; 64]; Cell::COUNT] = [")?;
            for (i, sub) in self.pieces.iter().enumerate() {
                writeln!(w, "    /*{:2}*/ [", i)?;
                for (i, hsh) in sub.iter().enumerate() {
                    writeln!(w, "        /*{:2}*/ {:#x},", i, hsh)?;
                }
                writeln!(w, "    ],")?;
            }
            writeln!(w, "];\n")?;

            writeln!(w, "pub const MOVE_SIDE: u64 = {:#x};\n", self.move_side)?;

            writeln!(w, "const CASTLING: [u64; 16] = [")?;
            for (i, sub) in self.castling.iter().enumerate() {
                writeln!(w, "    /*{:2}*/ {:#x},", i, sub)?;
            }
            writeln!(w, "];\n")?;

            writeln!(w, "const ENPASSANT: [u64; 64] = [")?;
            for (i, sub) in self.enpassant.iter().enumerate() {
                writeln!(w, "    /*{:2}*/ {:#x},", i, sub)?;
            }
            writeln!(w, "];\n")?;

            writeln!(
                w,
                "const CASTLING_KINGSIDE: [u64; 2] = [{:#x}, {:#x}];",
                self.castling_kingside[0], self.castling_kingside[1]
            )?;
            writeln!(
                w,
                "const CASTLING_QUEENSIDE: [u64; 2] = [{:#x}, {:#x}];",
                self.castling_queenside[0], self.castling_queenside[1]
            )?;

            Ok(())
        }
    }

    pub fn gen(out_path: &Path) -> io::Result<()> {
        Zobrist::generate_default().output(&mut BufWriter::new(&fs::File::create(out_path)?))?;
        Ok(())
    }
}

mod near_attacks {
    use std::io::{self, BufWriter, Write};
    use std::{fs, path::Path};

    use owlchess_base::bitboard::Bitboard;
    use owlchess_base::types::Coord;

    fn generate_directed<const N: usize>(d_file: [isize; N], d_rank: [isize; N]) -> [Bitboard; 64] {
        let mut res = [Bitboard::EMPTY; 64];
        for c in Coord::iter() {
            let mut bb = Bitboard::EMPTY;
            for (&delta_file, &delta_rank) in d_file.iter().zip(d_rank.iter()) {
                if let Some(nc) = c.shift(delta_file, delta_rank) {
                    bb.set(nc);
                }
            }
            res[c.index()] = bb;
        }
        res
    }

    pub fn gen(out_path: &Path) -> io::Result<()> {
        let f = fs::File::create(out_path)?;
        let mut w = BufWriter::new(&f);

        super::print_bitboards(
            &mut w,
            "KING_ATTACKS",
            &generate_directed([-1, -1, -1, 0, 0, 1, 1, 1], [-1, 0, 1, -1, 1, -1, 0, 1]),
        )?;
        writeln!(&mut w)?;
        super::print_bitboards(
            &mut w,
            "KNIGHT_ATTACKS",
            &generate_directed([-2, -2, -1, -1, 2, 2, 1, 1], [-1, 1, -2, 2, -1, 1, -2, 2]),
        )?;
        writeln!(&mut w)?;
        super::print_bitboards(
            &mut w,
            "WHITE_PAWN_ATTACKS",
            &generate_directed([-1, 1], [-1, -1]),
        )?;
        writeln!(&mut w)?;
        super::print_bitboards(
            &mut w,
            "BLACK_PAWN_ATTACKS",
            &generate_directed([-1, 1], [1, 1]),
        )?;

        Ok(())
    }
}

mod magic {
    use std::io::{self, BufWriter, Write};
    use std::{cmp, fs, path::Path};

    use owlchess_base::bitboard::Bitboard;
    use owlchess_base::bitboard_consts;
    use owlchess_base::types::Coord;
    use rand_core::RngCore;

    const FILE_FRAME: Bitboard = Bitboard::from_raw(0xff000000000000ff);
    const RANK_FRAME: Bitboard = Bitboard::from_raw(0x8181818181818181);
    const DIAG_FRAME: Bitboard = Bitboard::from_raw(0xff818181818181ff);

    struct Offsets {
        ranges: [(usize, usize); 64],
        total: usize,
    }

    trait Magic {
        const NAME: &'static str;
        const SHIFTS: &'static [(isize, isize)];

        /// If `None`, then the magics are generated automatically in this build script, slowing it down.
        /// Otherwise, the build takes pre-generated magics from here and just validates them. Note that
        /// in this case, it is still impossible to build with incorrect magics.
        ///
        /// Generating magics takes ~20 seconds of build time if the build script runs unoptimized. So,
        /// `BishopMagic` and `RookMagic` implementations of this trait use pre-generated tables.
        const MAGICS: Option<[u64; 64]>;

        fn build_mask(c: Coord) -> Bitboard;
        fn build_post_mask(c: Coord) -> Bitboard;

        /// Determine pointers for shared rook and bishop arrays
        /// For rooks we share two cells (one entry for both `c1` and `c2`)
        /// For bishops the number of shared cells is equal to four
        /// To find more details, see https://www.chessprogramming.org/Magic_Bitboards#Sharing_Attacks
        fn init_offsets() -> Offsets;

        fn get_mask_size(c: Coord) -> usize {
            Self::build_mask(c).len() as usize
        }

        fn get_shift(c: Coord) -> usize {
            64 - Self::get_mask_size(c)
        }
    }

    struct BishopMagic;
    struct RookMagic;

    impl Magic for RookMagic {
        const NAME: &'static str = "ROOK";
        const SHIFTS: &'static [(isize, isize)] = &[(0, 1), (0, -1), (-1, 0), (1, 0)];

        #[rustfmt::skip]
        const MAGICS: Option<[u64; 64]> = Some([
            0x24800140001480a0, 0x0040002001409002, 0x0080200010008088, 0x0080048210010800, 0x0080040102804800, 0x0080020004008009, 0x2080010021800a00, 0xa08004418004a900,
            0x81008000e0400094, 0x0264c00460085001, 0x0185004100142003, 0x60020010200a0040, 0x6021000411000801, 0x0182000890046200, 0x1404000204084510, 0x0082801a40800100,
            0x08108c8000204000, 0x0d20010100804000, 0x02d0002004002802, 0x8008008030000a80, 0x0000110004480100, 0x0085808024002200, 0x10a8410100040200, 0x40008200008401c3,
            0x004004888000416d, 0x09201000400029c0, 0x5120080040401000, 0x0001002100100108, 0x82080008800c0080, 0x0016000200481114, 0x0100080400611002, 0x0231000100034282,
            0x5208814010800020, 0x06ca884102002200, 0x0001002001001040, 0x0010010008080080, 0x4000080080800400, 0x0800040080800200, 0x0000500a0c000801, 0x0000a88402000041,
            0x4108804000208000, 0x0009402010004001, 0x0002100020008080, 0x00d2002029420010, 0x10c2000408120020, 0x0002000448920010, 0x1010220001008080, 0x0000440440820001,
            0x0904e08000400080, 0x005100608a400100, 0x0081004020001500, 0x2214218812004200, 0x8000a50010080100, 0x4042001004480200, 0x0400083082490400, 0x9020404420850200,
            0x0020430010218001, 0x4201810020104001, 0x2020002100408891, 0x000010002c210901, 0x0002000c20381002, 0x10050002080c0001, 0x180a008308102204, 0x8000104082a30402,
        ]);

        fn build_mask(c: Coord) -> Bitboard {
            ((bitboard_consts::file(c.file()) & !FILE_FRAME)
                | (bitboard_consts::rank(c.rank()) & !RANK_FRAME))
                & !Bitboard::from_coord(c)
        }

        fn build_post_mask(c: Coord) -> Bitboard {
            bitboard_consts::file(c.file()) ^ bitboard_consts::rank(c.rank())
        }

        fn init_offsets() -> Offsets {
            let mut ranges = [(0, 0); 64];
            let mut total = 0;
            for c1 in Coord::iter() {
                let c2 = unsafe { Coord::from_index_unchecked(c1.index() ^ 9) };
                if c1.index() > c2.index() {
                    continue;
                }
                let max_len = cmp::max(Self::get_mask_size(c1), Self::get_mask_size(c2));
                let range = (total, total + (1 << max_len));
                ranges[c1.index()] = range;
                ranges[c2.index()] = range;
                total = range.1;
            }
            Offsets { ranges, total }
        }
    }

    impl Magic for BishopMagic {
        const NAME: &'static str = "BISHOP";
        const SHIFTS: &'static [(isize, isize)] = &[(-1, 1), (-1, -1), (1, -1), (1, 1)];

        #[rustfmt::skip]
        const MAGICS: Option<[u64; 64]> = Some([
            0x0420140106022200, 0x0010040808c64000, 0x0031030602040400, 0x0004440289000402, 0x8801104084200242, 0x24010120100c2512, 0x2080a40108400742, 0x0802004402080284,
            0x4a400404840c2408, 0x0580180101040300, 0x0860284802408080, 0x0010482a04201480, 0x4821042420000400, 0x0040020202208100, 0x04000c010108220c, 0x0902020202096c0a,
            0x40c8454222040400, 0x0008006002040046, 0x80100044440020c0, 0x0888000406420800, 0x000c000081a00200, 0x0002002109052009, 0x011cc04088045000, 0x0058a00101081200,
            0x0008880040100122, 0x0050306004050200, 0x0280300002028a00, 0x1020080081004208, 0x04008a0004010404, 0x8114010130900480, 0x0201040802020140, 0x4012002100c70810,
            0x0048041000401210, 0x06008809000c1030, 0x08a4020108a20400, 0x1400200800010504, 0x0701440400014100, 0x1000848100020110, 0x0041820088620800, 0x0411061a0054804a,
            0x028a012088002011, 0x1100840120240880, 0xa083003802040404, 0x140002023401a200, 0x4000c81d00400c00, 0x0002408107000200, 0x002d081800401104, 0x8009010102108100,
            0x7892024220040208, 0x0040410088200282, 0x1128082211101001, 0x0200100220880010, 0x1022401022020000, 0x002820600e008248, 0x0840080604004100, 0x0804054806008404,
            0x0005240402015000, 0x1000629401080201, 0xa000080080882110, 0x20000010808c0400, 0x0408c04a40028220, 0x0180066244100080, 0x40140a06080a0400, 0x0040418109020080,
        ]);

        fn build_mask(c: Coord) -> Bitboard {
            (bitboard_consts::DIAG[c.diag()] ^ bitboard_consts::ANTIDIAG[c.antidiag()])
                & !DIAG_FRAME
        }

        fn build_post_mask(c: Coord) -> Bitboard {
            bitboard_consts::DIAG[c.diag()] ^ bitboard_consts::ANTIDIAG[c.antidiag()]
        }

        fn init_offsets() -> Offsets {
            let mut ranges = [(0, 0); 64];
            let mut total = 0;

            // We consider 16 groups of shared bishop cells. A single group contains four cells with
            // coordinates `c + i * offs` for all `i = 0..3`.
            const STARTS: [usize; 16] = [0, 1, 32, 33, 2, 10, 18, 26, 34, 42, 50, 58, 6, 7, 38, 39];
            const OFFSETS: [usize; 16] = [8, 8, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1, 8, 8, 8, 8];
            for (&c, &offs) in STARTS.iter().zip(OFFSETS.iter()) {
                let items = [0, 1, 2, 3].map(|i| Coord::from_index(c + i * offs));
                let max_len = items.into_iter().map(Self::get_mask_size).max().unwrap();
                let range = (total, total + (1 << max_len));
                for item in items {
                    ranges[item.index()] = range;
                }
                total = range.1;
            }

            Offsets { ranges, total }
        }
    }

    fn is_valid_magic_const<M: Magic>(coord: Coord, magic: u64) -> bool {
        let mask = M::build_mask(coord);
        let shift = mask.len() as usize;
        let submask_cnt = 1_u64 << shift;
        let mut used = vec![false; submask_cnt as usize];
        for submask in 0..submask_cnt {
            let occupied = mask.deposit_bits(submask);
            let idx = (occupied.as_raw().wrapping_mul(magic) >> (64 - shift)) as usize;
            if used[idx] {
                return false;
            }
            used[idx] = true;
        }
        true
    }

    fn gen_sparse_number<R: RngCore>(r: &mut R) -> u64 {
        let mut res = 0;
        for _ in Coord::iter() {
            res <<= 1;
            if r.next_u64() % 8 == 0 {
                res |= 1;
            }
        }
        res
    }

    fn gen_magic_consts<M: Magic, R: RngCore>(r: &mut R) -> [u64; 64] {
        if let Some(res) = M::MAGICS {
            for c in Coord::iter() {
                assert!(is_valid_magic_const::<M>(c, res[c.index()]));
            }
            return res;
        }

        let mut res = [0; 64];
        for c in Coord::iter() {
            let cur = &mut res[c.index()];
            loop {
                *cur = gen_sparse_number(r);
                if is_valid_magic_const::<M>(c, *cur) {
                    break;
                }
            }
        }
        res
    }

    fn write_magic_tables<M: Magic, W: Write>(
        w: &mut W,
        magic_consts: [u64; 64],
    ) -> io::Result<()> {
        let off = M::init_offsets();

        writeln!(w, "const MAGIC_CONSTS_{}: [u64; 64] = [", M::NAME)?;
        for (i, b) in magic_consts.iter().enumerate() {
            writeln!(w, "    /*{:2}*/ 0x{:016x},", i, b)?;
        }
        writeln!(w, "];")?;

        writeln!(w)?;

        writeln!(w, "const MAGIC_SHIFTS_{}: [u64; 64] = [", M::NAME)?;
        for c in Coord::iter() {
            writeln!(w, "    /*{:2}*/ {},", c.index(), M::get_shift(c))?;
        }
        writeln!(w, "];")?;

        writeln!(w)?;

        writeln!(w, "static MAGIC_{}: [MagicEntry; 64] = [", M::NAME)?;
        for c in Coord::iter() {
            let i = c.index();
            writeln!(
                w,
                "    /*{:2}*/ MagicEntry {{mask: bb(0x{:016x}), post_mask: bb(0x{:016x}), lookup: &MAGIC_LOOKUP_{}[{}]}},",
                i,
                M::build_mask(c).as_raw(),
                M::build_post_mask(c).as_raw(),
                M::NAME,
                off.ranges[i].0,
            )?;
        }
        writeln!(w, "];")?;

        writeln!(w)?;

        let lookups = {
            let mut lookups = vec![Bitboard::EMPTY; off.total];
            for c in Coord::iter() {
                let mask = M::build_mask(c);
                let magic = magic_consts[c.index()];
                let shift = mask.len() as usize;
                let submask_cnt = 1_u64 << shift;
                for submask in 0..submask_cnt {
                    let occupied = mask.deposit_bits(submask);
                    let idx = (occupied.as_raw().wrapping_mul(magic) >> (64 - shift)) as usize;
                    let res = &mut lookups[idx + off.ranges[c.index()].0];
                    for &(delta_file, delta_rank) in M::SHIFTS {
                        let mut p = c;
                        while let Some(new_p) = p.shift(delta_file, delta_rank) {
                            res.set(new_p);
                            if occupied.has(new_p) {
                                break;
                            }
                            p = new_p;
                        }
                    }
                    res.unset(c);
                }
            }
            lookups
        };

        writeln!(
            w,
            "static MAGIC_LOOKUP_{}: [Bitboard; {}] = [",
            M::NAME,
            lookups.len()
        )?;
        for (i, b) in lookups.iter().enumerate() {
            writeln!(w, "    /*{}*/ bb(0x{:016x}),", i, b.as_raw())?;
        }
        writeln!(w, "];")?;

        Ok(())
    }

    fn gen_magic_tables<M: Magic, W: Write, R: RngCore>(w: &mut W, r: &mut R) -> io::Result<()> {
        write_magic_tables::<M, _>(w, gen_magic_consts::<M, _>(r))
    }

    pub fn gen(out_path: &Path) -> io::Result<()> {
        let f = fs::File::create(out_path)?;
        let mut w = BufWriter::new(&f);
        gen_magic_tables::<BishopMagic, _, _>(&mut w, &mut super::default_gen())?;
        writeln!(w)?;
        gen_magic_tables::<RookMagic, _, _>(&mut w, &mut super::default_gen())?;
        Ok(())
    }
}

mod between {
    use std::io::{self, BufWriter, Write};
    use std::{fs, path::Path};

    use owlchess_base::bitboard::Bitboard;
    use owlchess_base::bitboard_consts;
    use owlchess_base::types::Coord;

    fn bishop(mask: impl Fn(Coord) -> Bitboard) -> [Bitboard; 64] {
        let mut res = [Bitboard::EMPTY; 64];
        for c in Coord::iter() {
            let val = bitboard_consts::DIAG[c.diag()] | bitboard_consts::ANTIDIAG[c.antidiag()];
            res[c.index()] = val & mask(c);
        }
        res
    }

    fn rook(mask: impl Fn(Coord) -> Bitboard) -> [Bitboard; 64] {
        let mut res = [Bitboard::EMPTY; 64];
        for c in Coord::iter() {
            let val = bitboard_consts::file(c.file()) | bitboard_consts::rank(c.rank());
            res[c.index()] = val & mask(c);
        }
        res
    }

    fn not_eq(c: Coord) -> Bitboard {
        !Bitboard::from_coord(c)
    }

    fn less(c: Coord) -> Bitboard {
        Bitboard::from((1u64 << c.index()).wrapping_sub(1))
    }

    fn less_eq(c: Coord) -> Bitboard {
        less(c) | Bitboard::from_coord(c)
    }

    fn greater(c: Coord) -> Bitboard {
        !less_eq(c)
    }

    pub fn gen(out_path: &Path) -> io::Result<()> {
        let f = fs::File::create(out_path)?;
        let mut w = BufWriter::new(&f);

        super::print_bitboards(&mut w, "BISHOP_LT", &bishop(less))?;
        writeln!(&mut w)?;
        super::print_bitboards(&mut w, "BISHOP_GT", &bishop(greater))?;
        writeln!(&mut w)?;
        super::print_bitboards(&mut w, "BISHOP_NE", &bishop(not_eq))?;

        writeln!(&mut w)?;

        super::print_bitboards(&mut w, "ROOK_LT", &rook(less))?;
        writeln!(&mut w)?;
        super::print_bitboards(&mut w, "ROOK_GT", &rook(greater))?;
        writeln!(&mut w)?;
        super::print_bitboards(&mut w, "ROOK_NE", &rook(not_eq))?;

        Ok(())
    }
}

fn main() -> io::Result<()> {
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = env::var("OUT_DIR").unwrap();

    zobrist::gen(&Path::new(&out_dir).join("zobrist.rs"))?;
    near_attacks::gen(&Path::new(&out_dir).join("near_attacks.rs"))?;
    magic::gen(&Path::new(&out_dir).join("magic.rs"))?;
    between::gen(&Path::new(&out_dir).join("between.rs"))?;

    Ok(())
}
