use crate::board::*;
use crate::move_generation::*;

pub fn get_evalulation(board: &mut BoardState) -> f64 {
    let mut score: f64 = 0.0;

    //  Move Counts

    let player_1_move_count = valid_move_count_2(board, 1) as f64;
    let player_2_moves_count = valid_move_count_2(board, 2) as f64;

    score += player_1_move_count;
    score -= player_2_moves_count;

    // Attacking and Threating Pieces

    let mut attacking_pieces = 0;
    let mut threating_pieces = 0;

    for x in 0..6 {
        if board.data[6 + x] == 2 || board.data[6 + x] == 3 {
            attacking_pieces += 1;

        }
        if board.data[12 + x] == 3 {
            attacking_pieces += 1;

        }

        if board.data[24 + x] == 2 || board.data[24 + x] == 3 {
            threating_pieces += 1;

        }
        if board.data[18 + x] == 3 {
            threating_pieces += 1;

        }

    }

    score += 150.0 * (attacking_pieces as f64 - threating_pieces as f64);

    // Keeping ones behind twos

    let mut good_pairs = 0;
    let mut bad_pairs = 0;

    for i in 0..36 {
        if board.data[i] == 1 {
            if i + 6 <= 35 {
                if board.data[i + 6] == 2 {
                    good_pairs += 1;
                }
            } else if i - 6 <= 35 {
                if board.data[i - 6] == 2 {
                    bad_pairs += 1;
                }
            }

        }
    }

    score += 200.0 * (good_pairs as f64 - bad_pairs as f64);

    return score;

}   


