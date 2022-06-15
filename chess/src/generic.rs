use crate::types;

pub trait Color {
    const COLOR: types::Color;
    const CASTLING_OFFSET: usize;
}

pub struct White;
pub struct Black;

impl Color for White {
    const COLOR: types::Color = types::Color::White;
    const CASTLING_OFFSET: usize = 56;
}

impl Color for Black {
    const COLOR: types::Color = types::Color::Black;
    const CASTLING_OFFSET: usize = 0;
}
