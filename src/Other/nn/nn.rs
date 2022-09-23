extern crate neuroflow;
use neuroflow::FeedForward;
use neuroflow::data::DataSet;
use neuroflow::activators::Type::Tanh;

use std::path::Path;    
use std::fs::OpenOptions;
use std::io::prelude::*;

use rand::Rng;

use crate::board::BoardState;

pub struct NN {
    pub network: FeedForward,
    pub dataset: DataSet,

}

impl NN {
    pub fn new() -> NN {
        let dataset = DataSet::from_csv("src/Other/nn/move_count_dataset.csv.csv").unwrap();

        return NN {
            network: FeedForward::new(&[36, 36, 36, 36, 36, 1]),
            dataset: dataset,

        };

    }

    pub fn train(&mut self) {
        let mut new_net = FeedForward::new(&[36, 36, 36, 36, 36, 1]);
        new_net.activation(Tanh).learning_rate(0.01).train(&self.dataset, 500000);

        self.network = new_net;
        
    }

    pub fn infrence(&mut self, board: &mut BoardState) -> f64 {
        let result = self.network.calc(&board.to_training_data());    
        return result[0];

    }

}

pub fn generate_move_count_data() {
    loop {
        let mut board = BoardState::new();
        board.set_rank([1, 0, 0 ,3, 2, 1], 5);
        board.set_rank([0 ,0 ,0, 0, 1, 1], 4);
        board.set_rank([0 ,0 ,0, 3, 0, 0], 3);
        board.set_rank([0 ,2 ,0 ,0, 0, 0], 2);
        board.set_rank([0 ,0, 3, 0, 3, 0], 1);
        board.set_rank([0 ,2 ,0 ,2, 0, 0], 0);
        board.set_goals([0, 0]);
    
        board.set_rank(get_random_rank(), 5);
        board.set_rank(get_random_rank(), 0);
     
        let mut nn = NN::new();
        nn.dataset = DataSet::from_csv("src/Other/nn/move_count_dataset.csv.csv").unwrap();
        nn.train();
    
        println!("{}", nn.infrence(&mut board));
        println!("{}", valid_move_count_2(&mut board, 1));
    
        let mut data_collection_file = OpenOptions::new().write(true).append(true).open("data2.csv").unwrap();
    
    
        let mut current_player = 1.0;
        let mut winner = 0.0;
        loop {
            // DATA COLLECTION FOR CURRENT BOARD
    
            let move_count = valid_move_count_2(&mut board, 1);
            for i in 0..36 {
                let data = &board.data[i];
                write!(data_collection_file, "{data}, ").expect("ERROR!");
            }    
            writeln!(data_collection_file, "- , {move_count}").expect("ERROR!");
    
            // ===========
    
            let results = negamax_normal.iterative_deepening_search(&mut board, 3);
            
            println!("===========");
            let mut best_move: (Move, f64) = (Move([NULL, NULL, NULL, NULL, NULL, NULL]), -f64::INFINITY);
            for result in results {
                println!("{:?}", result.best_move);
    
                if result.best_move.1 == f64::NEG_INFINITY{
                    break;
    
                } else if result.best_move.1 == f64::INFINITY {
                    best_move = result.best_move;
                    break;
    
                } else {
                    best_move = result.best_move;
    
                }
    
            }
    
            if best_move.1 == f64::NEG_INFINITY {
                println!("PLAYER {} WON", -current_player);
                winner = -current_player;
                break;
    
            } else if best_move.1 == f64::INFINITY {
                println!("PLAYER {} WON", current_player);
                winner = current_player;
                break;
    
            }
    
            board.make_move(&best_move.0);
    
            println!("BEST: {:?}", best_move);
            println!("===========");
    
            board.flip();
            current_player *= -1.0;
    
        }
    
        if winner == -1.0 {
            board.flip();
        }
        board.print();
        println!("WINNER");
    
    
    }
}





fn get_random_rank() -> [usize; 6] {
    let mut rng = rand::thread_rng();

    let mut random_rank: [usize; 6] = [0; 6];
    let mut pieces: Vec<usize> = vec![1, 2, 3, 2, 1, 3];
    for i in 0..6 {
        random_rank[i] = pieces.remove(rng.gen_range(0..pieces.len()));

    }

    return random_rank;
    negamax_normal.eval_type = "NORMAL".to_string();

}
