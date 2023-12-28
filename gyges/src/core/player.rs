use std::ops::Not;
use std::fmt::Display;

/// Enum to represent a player on the board
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum Player  {
    One = 0, 
    Two = 1, 

}

impl Player {
    /// Returns the other player
    pub fn other(self) -> Player {
        !self

    }
    
    /// Returns the multipler used for evaluation
    pub fn eval_multiplier(&self) -> f64 {
        match self {
            Player::One => {
                1.0

            }
            Player::Two => {
                -1.0

            }

        }

    }

}

impl Display for Player {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Player::One => write!(f, "P1"),
            Player::Two => write!(f, "P2"),

        }

    }

}

impl Not for Player {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            Player::One => {
                Player::Two

            }
            Player::Two => {
                Player::One
                
            }

        }

    }

}
