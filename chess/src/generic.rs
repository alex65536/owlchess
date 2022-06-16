use crate::types;

pub trait Color {
    const COLOR: types::Color;
    const CASTLING_OFFSET: usize;
    type Inv: Color;
}

pub struct White;
pub struct Black;

impl Color for White {
    const COLOR: types::Color = types::Color::White;
    const CASTLING_OFFSET: usize = 56;
    type Inv = Black;
}

impl Color for Black {
    const COLOR: types::Color = types::Color::Black;
    const CASTLING_OFFSET: usize = 0;
    type Inv = White;
}
