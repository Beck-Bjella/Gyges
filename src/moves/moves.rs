use std::cmp::Ordering;

use crate::board::bitboard::BitBoard;
use crate::board::{board::*};
use crate::moves::move_gen::*;
use crate::search::evaluation::*;

use crate::tools::tt::*;
use crate::consts::*;


/// Designates the type of move.
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum MoveType {
    Drop,
    Bounce,
    None

}

/// Structure that defines a move
#[derive(PartialEq, Debug, Clone, Copy)]
pub struct Move {
    pub data: [usize; 6],
    pub flag: MoveType,

}

impl Move {
    /// Create a new move from its indivudal components.
    pub fn new(data: [usize; 6], flag: MoveType) -> Move {
        Move {
            data,
            flag,

        }

    }

    /// Creates a new null move.
    pub fn new_null() -> Move {
        Move {
            data: [NULL; 6],
            flag: MoveType::None,

        }

    }

    /// Checks if a move is null.
    pub fn is_null(&self) -> bool {
        self.data == [NULL; 6] && self.flag == MoveType::None

    }

    /// Checks if a move wins the game
    pub fn is_win(&self) -> bool {
        if self.flag == MoveType::Bounce {
            return (self.data[3] == PLAYER_1_GOAL) || (self.data[3] == PLAYER_2_GOAL)

        } else {
            return (self.data[5] == PLAYER_1_GOAL) || (self.data[5] == PLAYER_2_GOAL)

        }

    }

}

impl From<TTMove> for Move {
    fn from(mv: TTMove) -> Self {
        let mut data = [0; 6];
        for i in 0..mv.data.len() {
            data[i] = mv.data[i] as usize

        }

        Move {
            data,
            flag: mv.flag

        }

    }
}

/// Structure that defines a move for the tt table.
#[derive(PartialEq, Debug, Clone, Copy)]
pub struct TTMove {
    pub data: [u8; 6],
    pub flag: MoveType,

}

impl From<Move> for TTMove {
    fn from(mv: Move) -> Self {
        let mut data = [0; 6];
        for i in 0..mv.data.len() {
            data[i] = mv.data[i] as u8

        }

        TTMove {
            data,
            flag: mv.flag

        }

    }
}

/// Structure that defines a rootmove.
#[derive(PartialEq, Debug, Clone, Copy)]
pub struct RootMove {
    pub mv: Move,
    pub score: f64,
    pub threats: usize,
    pub ply: i8

}

impl RootMove {
    /// Creates a rootmove from its indivudal components.
    pub fn new(mv: Move, score: f64, ply: i8, threats: usize) -> RootMove {
        RootMove {
            mv,
            score,
            ply,
            threats

        }

    }

    /// Creates a null rootmove.
    pub fn new_null() -> RootMove {
        RootMove {
            mv: Move::new_null(),
            score: 0.0,
            ply: 0,
            threats: 0

        }

    }

    /// Checks if a rootmove is null.
    pub fn is_null(&self) -> bool {
        self.mv.is_null() && self.score == 0.0 && self.ply == 0

    }

    /// Sets the score and the search ply of a rootmove.
    pub fn set_score_and_ply(&mut self, score: f64, ply: i8) {
        self.score = score;
        self.ply = ply;

    }

    pub fn as_human(&self) -> String {
        if self.mv.flag == MoveType::Bounce {
            return human_cord(self.mv.data[1]) + &human_cord(self.mv.data[3]);

        } else {
            human_cord(self.mv.data[1]) + &human_cord(self.mv.data[3]) + &human_cord(self.mv.data[5])

        }

    }

    pub fn as_ugi(&self) -> String {
        if self.mv.flag == MoveType::Bounce {
            return self.mv.data[1].to_string() + "|" + &self.mv.data[3].to_string();

        } else {
            return self.mv.data[1].to_string() + "|" + &self.mv.data[3].to_string() + "|" + &self.mv.data[5].to_string()

        }

    }

}

/// Orders a list of moves.
pub fn order_moves(moves: Vec<Move>, board: &mut BoardState, player: f64, pv: &Vec<Entry>) -> Vec<Move> {
    // For every move calculate a value to sort it by.
    let mut moves_to_sort: Vec<(Move, f64)> = moves.into_iter().map(|mv| {
        let mut sort_val: f64 = 0.0;
        let mut new_board = board.make_move(&mv);
        
        // If move is in the PV then sort it first.
        for e in pv {
            if Move::from(e.bestmove ) == mv {
                sort_val = 500_000.0;
                return (mv, sort_val);

            }

        }

        // If move is not the PV then guess how good it is.
        sort_val -= unsafe { valid_move_count(&mut new_board, -player)} as f64;
        
        // If a move has less then 5 threats then penalize it.
        let threat_count = unsafe{ valid_threat_count(&mut new_board, player) };
        if threat_count <= 5_usize {
            sort_val -= 1000.0 * (5 - threat_count) as f64;

        }

        // Lower the score if there are pieces that cant reach anything or are unreachable on your active line.
        // sort_val -= activeline_cant_reach(board, player) as f64 * 1000.0;
        // sort_val -= activeline_unreachable(board, player) as f64 * 500.0;

        // Lower the score of moves that leave the piece where it cant reach anything.
        // let end_pos: usize = if mv.flag == MoveType::Drop { 
        //     mv.data[5] 

        // } else { 
        //     mv.data[3] 

        // };
        // let end_type = new_board.data[end_pos];
    
        
        // if piece_cant_reach(board, end_pos, end_type) {
        //     sort_val -= (4 - end_type) as f64 * 1000.0

        // }

        (mv, sort_val)

    }).collect();

    // Sort the moves based on their predicted values.
    moves_to_sort.sort_by(|a, b| {
        if a.1 > b.1 {
            Ordering::Less
            
        } else if a.1 == b.1 {
            Ordering::Equal

        } else {
            Ordering::Greater

        }

    });

    // Collect the moves.
    let ordered_moves: Vec<Move> = moves_to_sort.into_iter().map(|x| x.0).collect();

    ordered_moves
 
}
