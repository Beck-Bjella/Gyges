use std::fmt::Display;

use crate::board::bitboard::*;
use crate::tools::zobrist::*;
use crate::moves::moves::*;
use crate::consts::*;

#[derive(Clone, Copy, PartialEq)]
pub struct BoardState {
    pub data: [usize; 38],
    pub peice_board: BitBoard,
    pub player: f64,
    hash: u64

}

impl BoardState {
    pub fn new() -> BoardState {
        BoardState {
            data: [0; 38],
            peice_board: BitBoard(0),
            player: 0.0,
            hash: 0

        }
        
    }

    pub fn set(&mut self, rank5: [usize; 6], rank4: [usize; 6], rank3: [usize; 6], rank2: [usize; 6], rank1: [usize; 6], rank0: [usize; 6], goal_data: [usize; 2], player: f64) {
        for x in 0..6 {
            self.data[x] = rank0[x];
            self.data[x + 6] = rank1[x];
            self.data[x + 12] = rank2[x];
            self.data[x + 18] = rank3[x];
            self.data[x + 24] = rank4[x];
            self.data[x + 30] = rank5[x];

        }
        
        self.data[PLAYER_1_GOAL] = goal_data[0];
        self.data[PLAYER_2_GOAL] = goal_data[1];

        self.hash = get_uni_hash(self);

        self.player = player;
        
        for i in 0..36 {
            if self.data[i] != 0 {
                self.peice_board .set_bit(i);

            }

        }

    }

    pub fn make_move(self, mv: &Move) -> BoardState {
        let mut data = self.data.clone();
        let mut peice_board = self.peice_board.clone();
        let mut hash: u64 = self.hash.clone();
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

            peice_board.clear_bit(step1[1]);
            peice_board.set_bit(step3[1]);
            
        } else if mv.flag == MoveType::Bounce {
           
            hash ^= ZOBRIST_HASH_DATA[step1[1]][self.data[step1[1]]];

            hash ^= ZOBRIST_HASH_DATA[step2[1]][step2[0]];
            
            data[step1[1]] = step1[0];

            data[step2[1]] = step2[0];

            peice_board.clear_bit(step1[1]);

            peice_board.set_bit(step2[1]);

        }
        
        return BoardState {
            data,
            peice_board,
            player,
            hash

        }


    }

    pub fn make_null(self) -> BoardState {
        let data = self.data.clone();
        let peice_board = self.peice_board.clone();
        let hash = self.hash.clone();
        let player = self.player * -1.0;

        return BoardState {
            data,
            peice_board,
            player,
            hash

        }

    }

    pub fn get_active_lines(&self) -> [usize; 2] {
        let player_1_active_line = (self.peice_board.bit_scan_forward() as f64 / 6.0).floor() as usize;
        let player_2_active_line = (self.peice_board.bit_scan_reverse() as f64 / 6.0).floor() as usize;
        
        [player_1_active_line, player_2_active_line]

    }

    pub fn get_drops(&self, active_lines: [usize; 2], player: f64) -> BitBoard {
        if player == PLAYER_1 {
            return (FULL ^ OPP_BACK_ZONE[active_lines[1]]) & !self.peice_board;

        } else {
            return (FULL ^ PLAYER_BACK_ZONE[active_lines[0]]) & !self.peice_board;
                
        }

    }

    pub fn is_valid(&self) {
        let mut one_count = 0;
        let mut two_count = 0;
        let mut three_count = 0;

        for i in 0..36 {
            if self.data[i] == 1 {
                one_count += 1;

            } else if self.data[i] == 2 {
                two_count += 1;

            } else if self.data[i] == 3 {
                three_count += 1;

            } 


        }

        if !(one_count == 4 && two_count == 4 && three_count == 4) {
            panic!("ERROR INVALD BOARD");

        }

    }
    
    pub fn hash(&self) -> u64 {
        if self.player == PLAYER_1 {
            return self.hash ^ PLAYER_1_HASH;

        } else {
            return self.hash ^ PLAYER_2_HASH;

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
        writeln!(f, " ")?;
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
        writeln!(f, " ")?;

        return Result::Ok(());

    }

}
