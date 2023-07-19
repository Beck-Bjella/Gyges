use std::fmt::Display;

use crate::board::bitboard::*;
use crate::tools::zobrist::*;
use crate::moves::moves::*;
use crate::consts::*;

#[derive(Clone, Copy, PartialEq)]
pub struct BoardState {
    pub data: [usize; 38],
    pub piece_bb: BitBoard,
    pub player: f64,
    hash: u64

}

impl BoardState {
    pub fn new() -> BoardState {
        BoardState {
            data: [0; 38],
            piece_bb: BitBoard(0),
            player: 0.0,
            hash: 0

        }
        
    }

    pub fn set(&mut self, rank5: [usize; 6], rank4: [usize; 6], rank3: [usize; 6], rank2: [usize; 6], rank1: [usize; 6], rank0: [usize; 6], goal_data: [usize; 2], player: f64) {
        self.data[..6].copy_from_slice(&rank0);
        self.data[6..12].copy_from_slice(&rank1);
        self.data[12..18].copy_from_slice(&rank2);
        self.data[18..24].copy_from_slice(&rank3);
        self.data[24..30].copy_from_slice(&rank4);
        self.data[30..36].copy_from_slice(&rank5);
        
        self.data[PLAYER_1_GOAL] = goal_data[0];
        self.data[PLAYER_2_GOAL] = goal_data[1];

        self.hash = get_uni_hash(self.data);

        self.player = player;
        
        for i in 0..36 {
            if self.data[i] != 0 {
                self.piece_bb.set_bit(i);

            }

        }

    }

    pub fn from(data: [usize; 38], player: f64) -> BoardState {
        let hash = get_uni_hash(data);

        let mut piece_bb = BitBoard(0);
        for (i, piece) in data.iter().enumerate().take(36) {
            if *piece != 0 {
                piece_bb.set_bit(i);

            }

        }

        BoardState {
            data,
            piece_bb,
            player,
            hash

        }

    }

    pub fn make_move(self, mv: &Move) -> BoardState {
        let mut data = self.data;
        let mut piece_bb = self.piece_bb;
        let mut hash: u64 = self.hash;
        let player = self.player * -1.0;

        let step1 = [mv.data[0], mv.data[1]];
        let step2 = [mv.data[2], mv.data[3]];
        let step3 = [mv.data[4], mv.data[5]];
        
        if mv.flag == MoveType::Drop {
            hash ^= ZOBRIST_HASH_DATA[step1[1]][self.data[step1[1]]];

            hash ^= ZOBRIST_HASH_DATA[step2[1]][self.data[step2[1]]];
            hash ^= ZOBRIST_HASH_DATA[step2[1]][step2[0]];

            hash ^= ZOBRIST_HASH_DATA[step3[1]][step3[0]];
            
            data[step1[1]] = step1[0];

            data[step2[1]] = step2[0];
        
            data[step3[1]] = step3[0];

            piece_bb.clear_bit(step1[1]);
            piece_bb.set_bit(step3[1]);
            
        } else if mv.flag == MoveType::Bounce {
           
            hash ^= ZOBRIST_HASH_DATA[step1[1]][self.data[step1[1]]];

            hash ^= ZOBRIST_HASH_DATA[step2[1]][step2[0]];
            
            data[step1[1]] = step1[0];

            data[step2[1]] = step2[0];

            piece_bb.clear_bit(step1[1]);

            piece_bb.set_bit(step2[1]);

        }
        
        BoardState {
            data,
            piece_bb,
            player,
            hash

        }


    }

    pub fn make_null(self) -> BoardState {
        let data = self.data;
        let piece_bb = self.piece_bb;
        let hash = self.hash;
        let player = self.player * -1.0;

        BoardState {
            data,
            piece_bb,
            player,
            hash

        }

    }

    pub fn get_active_lines(&self) -> [usize; 2] {
        let player_1_active_line = (self.piece_bb.bit_scan_forward() as f64 / 6.0).floor() as usize;
        let player_2_active_line = (self.piece_bb.bit_scan_reverse() as f64 / 6.0).floor() as usize;
        
        [player_1_active_line, player_2_active_line]

    }

    pub fn get_drops(&self, active_lines: [usize; 2], player: f64) -> BitBoard {
        if player == PLAYER_1 {
            (FULL ^ OPP_BACK_ZONE[active_lines[1]]) & !self.piece_bb

        } else {
            (FULL ^ PLAYER_BACK_ZONE[active_lines[0]]) & !self.piece_bb
                
        }

    }
    
    pub fn hash(&self) -> u64 {
        if self.player == PLAYER_1 {
            self.hash ^ PLAYER_1_HASH

        } else {
            self.hash ^ PLAYER_2_HASH

        }

    }

    pub fn flip(&mut self) {
        let mut temp_data: [usize; 38] = [0; 38];
    
        for y in 0..6 {
            for x in 0..6 {
                let piece = self.data[y * 6 + x];
    
                temp_data[((5 - y) * 6) + (5 - x)] = piece;
    
            }
    
        }
    
        self.data = temp_data;
    
    }

}

impl Display for BoardState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.data[37] == 0 {
            writeln!(f, "                .")?;

        } else {
            writeln!(f, "                {}", self.data[37])?;

        }
        writeln!(f, " ")?;
        writeln!(f, " ")?;

        for y in (0..6).rev() {
            for x in 0..6 {
                if self.data[y * 6 + x] == 0 {
                    write!(f, "    .")?;
                } else {
                    write!(f, "    {}", self.data[y * 6 + x])?;

                }
               
            }
            writeln!(f, " ")?;
            writeln!(f, " ")?;

        }

        writeln!(f, " ")?;
        if self.data[36] == 0 {
            writeln!(f, "                .")?;

        } else {
            writeln!(f, "                {}", self.data[36])?;

        }

        Result::Ok(())

    }

}

pub const COL_LETTER: [&str; 6] = ["a", "b", "c", "d", "e", "f"];

pub fn human_cord(pos: usize) -> String {
    let y = (pos as f64 / 6.0).floor() as usize;
    let x: usize = pos - (y * 6);

    COL_LETTER[x].to_string() + &(y + 1).to_string()

} 
