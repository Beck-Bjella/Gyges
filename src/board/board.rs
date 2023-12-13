use crate::board::bitboard::*;
use crate::moves::moves::*;
use crate::consts::*;
use crate::tools::zobrist::*;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct BoardState {
    pub data: [usize; 38],
    pub piece_bb: BitBoard,
    pub player: f64,
    hash: u64,

}

impl BoardState {
    pub fn make_move(self, mv: &Move) -> BoardState {
        let mut data = self.data;
        let mut piece_bb = self.piece_bb;
        let player = self.player * -1.0;
        let mut hash = self.hash;

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

}

impl From<[usize; 38]> for BoardState {
    fn from(data: [usize; 38]) -> Self {
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
            player: 1.0,
            hash: hash

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

        return BoardState::from(array_data);
    
    }

}
