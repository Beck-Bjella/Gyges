//! Representaion of a board's state, functions to manipulate it, and bitboards.
//! 
//! This module contains the main BoardState struct.
//! 

pub mod bitboard;

use std::fmt::Display;

#[doc(inline)]
use crate::board::bitboard::*;

use crate::core::*;
use crate::core::masks::*;
use crate::moves::*;
use crate::tools::zobrist::*;

/// Represents the state of the board. Contains the board data, bitboards, and hash.
/// 
/// The board data is 38 element array. Each of these elements represent the pieces at each position on the board.
/// It also contains a bitboard representing all of the pieces on the board, and a hash of the boardstate.
/// This hash is computed using the concept of Zobrist hashing. [Chess Zobrist Example](https://www.chessprogramming.org/Zobrist_Hashing)
/// 
/// It is important to note that throughout the program, player 1 is always the player at the bottom of the board, and player 2 is always the player at the top. 
/// This is also explained in the [player] enum.
/// 
/// 
/// # Position Mapping
/// 
/// Each of the positions on the board can be mapped to a number from 0 to 37. This is done as follows:
/// 
/// ```
///
///              (Player 2)       
/// 
///                +----+
///                | 37 |
///                +----+
/// 
///    + ----------------------------+
/// R6 | 30 | 31 | 32 | 33 | 34 | 35 |
///    + ----------------------------+
/// R5 | 24 | 25 | 26 | 27 | 28 | 29 |
///    + ----------------------------+
/// R4 | 18 | 19 | 20 | 21 | 22 | 23 |
///    + ----------------------------+
/// R3 | 12 | 13 | 14 | 15 | 16 | 17 |
///    + ----------------------------+
/// R2 | 6  | 7  | 8  | 9  | 10 | 11 |
///    + ----------------------------+
/// R1 | 0  | 1  | 2  | 3  | 4  | 5  |
///    +-----------------------------+
///      A    B    C    D    E    F
///                 
///                +----+
///                | 36 |
///                +----+
///               
///              (Player 1)
/// 
/// ```
/// 
/// This mapping applys to many compnents of the program, such as the [bitboard]. Each of the respestive bits on the bitboard can be mapped to these
/// same positions. The [SQ] struct is used to represent these positions.
/// 
/// [boardstate]: 
/// [bitboard]: 
/// [SQ]: 
/// [player]:
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct BoardState {
    pub data: [Piece; 38],
    pub piece_bb: BitBoard,
    pub player: Player,
    hash: u64,

}

impl BoardState {
    /// Makes a move on the board and returns the new board. Updates the hash and bitboards accordingly.
    pub fn make_move(self, mv: &Move) -> BoardState {
        let mut new_state = self;
        new_state.player = new_state.player.other();

        let step1 = mv.data[0];
        let step2 = mv.data[1];
        let step3 = mv.data[2];
        
        if mv.flag == MoveType::Drop {
            new_state.hash ^= ZOBRIST_HASH_DATA[step1.1.0 as usize][self.piece_at(step1.1) as usize];

            new_state.hash ^= ZOBRIST_HASH_DATA[step2.1.0 as usize][self.piece_at(step2.1) as usize];
            new_state.hash ^= ZOBRIST_HASH_DATA[step2.1.0 as usize][step2.0 as usize];

            new_state.hash ^= ZOBRIST_HASH_DATA[step3.1.0 as usize][step3.0 as usize];

            new_state.place(step1.0, step1.1);
            new_state.place(step2.0, step2.1);
            new_state.place(step3.0, step3.1);

            new_state.piece_bb.clear_bit(step1.1.0 as usize);
            new_state.piece_bb.set_bit(step3.1.0 as usize);

        } else if mv.flag == MoveType::Bounce {
            new_state.hash ^= ZOBRIST_HASH_DATA[step1.1.0 as usize][self.piece_at(step1.1) as usize];

            new_state.hash ^= ZOBRIST_HASH_DATA[step2.1.0 as usize][step2.0 as usize];

            new_state.place(step1.0, step1.1);
            new_state.place(step2.0, step2.1);

            new_state.piece_bb.clear_bit(step1.1.0 as usize);
            new_state.piece_bb.set_bit(step2.1.0 as usize);

        }
        
        new_state

    }
    
    /// Places a piece on the board.
    /// Performs no checks and does not update bitboards or hash.
    pub fn place(&mut self, piece: Piece, square: SQ) {
        self.data[square.0 as usize] = piece;

    }

    /// Removes a piece from the board.
    /// Performs no checks and does not update bitboards or hash.
    pub fn remove(&mut self, square: SQ) {
        self.data[square.0 as usize] = Piece::None;

    }

    /// Returns the piece at a given square.
    pub fn piece_at(&self, square: SQ) -> Piece {
        self.data[square.0 as usize]

    }

    /// Checks if the board is valid and panics if it is not.
    /// A board is valid as long as it has all 12 pieces still on it.
    pub fn check_valid(&self){
        if self.piece_bb.pop_count() != 12 {
            println!("Board Data: {:?}", self.data);
            panic!("Invalid board state: {} pieces", self.piece_bb.pop_count());

        }

    }
    
    /// Returns the active lines for each player.
    #[inline(always)]
    pub fn get_active_lines(&self) -> [usize; 2] {
        let player_1_active_line = (self.piece_bb.bit_scan_forward() as f64 / 6.0).floor() as usize;
        let player_2_active_line = (self.piece_bb.bit_scan_reverse() as f64 / 6.0).floor() as usize;
        
        [player_1_active_line, player_2_active_line]

    }
    
    /// Returns a BitBoard of all the squares that a player can drop a piece on.
    #[inline(always)]
    pub fn get_drops(&self, active_lines: [usize; 2], player: Player) -> BitBoard {
        !self.piece_bb & (FULL ^ BACK_ZONES[player.other() as usize][active_lines[player.other() as usize]])

    }
    
    /// Returns the hash of the board state.
    pub fn hash(&self) -> u64 {
        self.hash ^ PLAYER_HASH_DATA[self.player as usize]

    }

}

impl Display for BoardState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.data[37] == Piece::None {
            writeln!(f, "                .")?;

        } else {
            writeln!(f, "                {}", self.data[37].to_string())?;

        }
        writeln!(f, " ")?;
        writeln!(f, " ")?;

        for y in (0..6).rev() {
            for x in 0..6 {
                if self.data[y * 6 + x] == Piece::None {
                    write!(f, "    .")?;
                } else {
                    write!(f, "    {}", self.data[y * 6 + x].to_string())?;

                }
               
            }
            writeln!(f, " ")?;
            writeln!(f, " ")?;

        }

        writeln!(f, " ")?;
        if self.data[36] == Piece::None {
            writeln!(f, "                .")?;

        } else {
            writeln!(f, "                {}", self.data[36].to_string())?;

        }

        Result::Ok(())

    }

}

impl From<[usize; 38]> for BoardState {
    fn from(array_data: [usize; 38]) -> Self {

        let mut data: [Piece; 38] = [Piece::None; 38];
        for (i, piece) in array_data.iter().enumerate().take(36) {
            data[i] = Piece::from(*piece);

        }

        let mut piece_bb = BitBoard::EMPTY;
        for (i, piece) in data.iter().enumerate().take(36) {
            if *piece != Piece::None {
                piece_bb.set_bit(i);

            }

        }

        let hash = get_uni_hash(data);

        BoardState {
            data,
            piece_bb,
            player: Player::One,
            hash

        }

    }

}

impl From<&str> for BoardState {
   fn from(value: &str) -> BoardState {
        let array_data: [usize; 38] = {
            let mut arr: [usize; 38] = [0; 38];
            for (i, c) in value.chars().take(38).enumerate() {
                arr[i] = c.to_digit(10).unwrap() as usize;
            }
            arr

        };

        BoardState::from(array_data)
    
    }

}

impl Default for BoardState {
    fn default() -> Self {
        BoardState::from(STARTING_BOARD)

    }

}

/// Array representing a starting boardstate.
pub const STARTING_BOARD: [usize; 38] = [
    3, 2, 1, 1, 2, 3,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    3, 2, 1, 1, 2, 3,
    0, 0

];

/// Array representing the boardstate used for benchmarking.
pub const BENCH_BOARD: [usize; 38] = [
    2, 0, 2, 1, 1, 0,
    0, 0, 0, 3, 3, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 2, 0, 0,
    0, 0, 0, 1, 0, 0,
    0, 3, 0, 1, 2, 3,
    0, 0

];

