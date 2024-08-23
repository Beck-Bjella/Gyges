use std::cmp::Ordering;

use gyges::moves::*;


/// A sortable list of RootMove's
/// 
/// Very simmilar to using `Vec<RootMove>` except implements custom functions for setup and sorting.
/// 
#[derive(Clone)]
pub struct RootMoveList {
    pub moves: Vec<RootMove>,

}

impl RootMoveList {
    pub fn new() -> RootMoveList {
        RootMoveList {
            moves: vec![],

        }

    }
    
    /// Sorts the RootMoveList by whatever move has the greatest score.
    pub fn sort(&mut self) {
        self.moves.sort_by(|a, b| {
            if a.score > b.score {
                Ordering::Less
                
            } else if a.score == b.score {
                Ordering::Equal

            } else {
                Ordering::Greater
    
            }
    
        });

    }

    /// Updates the score and ply fields of a specific move.
    pub fn update_move(&mut self, mv: Move, score: f64, ply: i8) {
        for root_move in self.moves.iter_mut() {
            if root_move.mv == mv {
                root_move.set_score_and_ply(score, ply);

            }

        }

        self.sort();

    }

    /// Returns the first move
    pub fn first(&self) -> RootMove {
        self.moves[0]

    }

}

impl From<RootMoveList> for Vec<Move> {
    fn from(val: RootMoveList) -> Vec<Move> {
        let moves: Vec<Move> = val.moves.into_iter().map( |mv| {
            mv.mv

        }).collect();

        moves

    }

}

impl Default for RootMoveList {
    fn default() -> Self {
        Self::new()

    }

}
