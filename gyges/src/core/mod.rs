//! All the core components of the game. 
//! 
//! This module contains the representation for the player, pieces, and squares. It also contains various masks for bitboards and bit twiddles.
//! 

pub mod bit_twiddles;
pub mod masks;

use std::ops::{Not, Add, AddAssign, Sub, SubAssign, Mul, MulAssign};
use std::fmt::Display;

/// Represents a player in the game. 
/// 
/// Player 1 is always the player at the bottom of the board, and player 2 is always the player at the top.
/// This is due to the nature of the game not having defined pieces for each player.
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


/// Represents a piece on the board.
/// 
/// A piece can take one of three different forms: a one, a two, or a three. 
/// A piece can also be empty (it dosent exist), which is represented by None.
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

/// Represents a square on the board.
/// 
/// A square is represented by a number from 0 to 37. This number is where the square is on the board.
/// The mapping of these numbers to their corosponding position is the same as the mapping documented on the [boardstate] struct.
/// 
/// [boardstate]: 
#[derive(PartialEq, Debug, Clone, Copy)]
pub struct SQ(pub u8);

impl SQ {
    pub const NONE: SQ = SQ(100);

    pub const P1_GOAL: SQ = SQ(36);
    pub const P2_GOAL: SQ = SQ(37);

    pub const GOALS: [SQ; 2] = [SQ(36), SQ(37)];

    pub fn bit(&self) -> u64 {
        1 << self.0

    }

    pub fn in_bounds(&self) -> bool {
        self.0 < 38

    }

    fn on_top_edge(&self) -> bool {
        self.0 + 6 > 35 

    }

    fn on_bottom_edge(&self) -> bool {
        self.0 - 6 > 35 

    }

    fn on_right_edge(&self) -> bool {
        self.0 == 5 || self.0 == 11 || self.0 == 17 || self.0 == 23 || self.0 == 29 || self.0 == 35

    }

    fn on_left_edge(&self) -> bool {
        self.0 == 0 || self.0 == 6 || self.0 == 12 || self.0 == 18 || self.0 == 24 || self.0 == 30

    }

}

impl Display for SQ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0 as usize)

    }
    
}

// impl math operations
impl Add<usize> for SQ {
    type Output = SQ;

    fn add(self, rhs: usize) -> SQ {
        SQ(self.0 + rhs as u8)

    }

}

impl Add<SQ> for SQ {
    type Output = SQ;

    fn add(self, rhs: SQ) -> SQ {
        SQ(self.0 + rhs.0)

    }

}

impl AddAssign<usize> for SQ {
    fn add_assign(&mut self, rhs: usize) {
        self.0 += rhs as u8

    }

}

impl AddAssign<SQ> for SQ {
    fn add_assign(&mut self, rhs: SQ) {
        self.0 += rhs.0

    }

}

impl Sub<usize> for SQ {
    type Output = SQ;

    fn sub(self, rhs: usize) -> SQ {
        SQ(self.0 - rhs as u8)

    }

}

impl Sub<SQ> for SQ {
    type Output = SQ;

    fn sub(self, rhs: SQ) -> SQ {
        SQ(self.0 - rhs.0)

    }

}

impl SubAssign<usize> for SQ {
    fn sub_assign(&mut self, rhs: usize) {
        self.0 -= rhs as u8

    }

}

impl SubAssign<SQ> for SQ {
    fn sub_assign(&mut self, rhs: SQ) {
        self.0 -= rhs.0

    }

}

impl Mul<usize> for SQ {
    type Output = SQ;

    fn mul(self, rhs: usize) -> SQ {
        SQ(self.0 * rhs as u8)

    }

}

impl Mul<SQ> for SQ {
    type Output = SQ;

    fn mul(self, rhs: SQ) -> SQ {
        SQ(self.0 * rhs.0)

    }

}

impl MulAssign<usize> for SQ {
    fn mul_assign(&mut self, rhs: usize) {
        self.0 *= rhs as u8

    }

}

impl MulAssign<SQ> for SQ {
    fn mul_assign(&mut self, rhs: SQ) {
        self.0 *= rhs.0

    }

}

pub const READABLE_SQS: [&str; 38] = [
    "a1", "b1", "c1", "d1", "e1", "f1",
    "a2", "b2", "c2", "d2", "e2", "f2",
    "a3", "b3", "c3", "d3", "e3", "f3",
    "a4", "b4", "c4", "d4", "e4", "f4",
    "a5", "b5", "c5", "d5", "e5", "f5",
    "a6", "b6", "c6", "d6", "e6", "f6",
    "P1", "P2"
    
];
