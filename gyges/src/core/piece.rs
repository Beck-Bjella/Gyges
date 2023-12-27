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

impl ToString for Piece {
    fn to_string(&self) -> String {
        match self {
            Piece::One => "1".to_string(),
            Piece::Two => "2".to_string(),
            Piece::Three => "3".to_string(),
            Piece::None => "0".to_string()

        }

    }

}
