use std::fmt::Display;


#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(u8)]
pub enum Piece {
    One = 0,
    Two = 1,
    Three = 2,
    None = 3

}

impl From<usize> for Piece {
    fn from(piece: usize) -> Self {
        match piece {
            1 => Piece::One,
            2 => Piece::Two,
            3 => Piece::Three,
            _ => Piece::None,

        }

    }

}

impl Display for Piece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Piece::One => write!(f, "1"),
            Piece::Two => write!(f, "2"),
            Piece::Three => write!(f, "3"),
            Piece::None => write!(f, "0"),

        }

    }

}
