use rand::Rng;

use crate::board::BoardState;
pub struct ZobristHasher {
    pub hash_data: [[u64; 3]; 36],
    pub player_1_hash: u64,
    pub player_2_hash: u64

}

impl ZobristHasher {
    pub fn new() -> ZobristHasher {
        let mut rng = rand::thread_rng();

        let mut data: [[u64; 3]; 36] = [[0; 3]; 36];
        
        for i in 0..36 {
            data[i] = [rng.gen(), rng.gen(), rng.gen()];
        
        }

        let player_1_hash: u64 = rng.gen();
        let player_2_hash: u64 = rng.gen();

        return ZobristHasher {
            hash_data: data,
            player_1_hash: player_1_hash,
            player_2_hash: player_2_hash,

        }

    }

    pub fn get_hash(&self, board: &mut BoardState, player: f64) -> u64 {
        let mut hash = 0;

        if player == 1.0 {
            hash ^= self.player_1_hash;

        } else {
            hash ^= self.player_2_hash;

        }

        for i in 0..36 {
            if board.data[i] != 0 {
                let piece_type = board.data[i];
                hash ^= self.hash_data[i][piece_type - 1];

            }

        }

        return hash;

    }

}