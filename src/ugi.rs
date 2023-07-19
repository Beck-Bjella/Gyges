use std::io::{self, BufRead};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use crate::moves::move_gen::controlled_squares;
use crate::tools::tt::init_tt;
use crate::board::{board::*, self};
use crate::consts::*;
use crate::search::searcher::*;
use crate::search::evaluation::*;

use crate::moves::move_gen::per_piece_move_counts;

pub struct Ugi {
    pub searcher_stop: Option<Sender<bool>>,
    pub search_options: SearchOptions

}

impl Ugi {
    pub fn new() -> Ugi {
        Ugi {
            searcher_stop: Option::None,
            search_options: SearchOptions::new()

        }

    }

    pub fn start(&mut self) {
        let stdin = io::stdin();
        let mut lines = stdin.lock().lines();
    
        loop {
            let line = lines.next().unwrap().unwrap();
            let commands: Vec<&str> = line.split_whitespace().collect();
    
            match commands.get(0).copied() {
                Some("ugi") => {
                    println!("id name nova");
                    println!("id author beck bjella");
    
                }
                Some("test") => {
                    println!("wall strength:{}", wall_strength(&mut self.search_options.board));
                    println!("wall offset: {}", wall_depth_offset(&mut self.search_options.board));
                    println!("p1 wall score: {}", p1_wall_score(&mut self.search_options.board));
                    println!("p2 wall score: {}", p2_wall_score(&mut self.search_options.board));

                    println!("p1 one conn: {}", ones_connectivity_score(&mut self.search_options.board, PLAYER_1));
                    println!("p2 one conn: {}", ones_connectivity_score(&mut self.search_options.board, PLAYER_2));

                }
                Some("setoption") => {
                    
    
                }
                Some("setpos") => {
                    let board_str = *commands.get(1).unwrap();

                    let array_data: [usize; 38] = {
                        let mut arr = [0; 38];
                        for (i, c) in board_str.chars().take(38).enumerate() {
                            arr[i] = c.to_digit(10).unwrap() as usize;
                        }
                        arr

                    };
        
                    let board: BoardState = BoardState::from(array_data, PLAYER_1);
                    self.search_options.board = board;
                    
                }
                Some("gamepos") => {
                    let mut board = BoardState::new();
                    board.set(
                        [0, 2, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 3, 0, 0, 0],
                        [0, 0, 2, 0, 0, 0],
                        [0, 0, 1, 2, 0, 0],
                        [1, 3, 3, 2, 0, 3],
                        [0, 0],
                        PLAYER_1
                    );

                    self.search_options.board = board;
                        
                }
                Some("testpos") => {
                    let mut board = BoardState::new();
                    board.set(
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 3, 1, 3, 3],
                        [0, 0, 2, 1, 1, 1],
                        [0, 0, 0, 2, 2, 2],
                        [0, 3, 0, 0, 0, 0],
                        [0, 0],
                        PLAYER_1
                    );

                    self.search_options.board = board;

                }
                Some("showpos") => {
                    println!("{}", self.search_options.board);
                    
                }
                Some("go") => {
                    self.go();
    
                }
                Some("stop") => {
                    self.stop();
    
                }
                Some("quit") => {
                    break;
    
                }
                _ => {}
    
            }
    
        }
    
    }
    
    pub fn go(&mut self) {
        init_tt(2usize.pow(24));

        let (ss, sr): (Sender<bool>, Receiver<bool>) = mpsc::channel();
        self.searcher_stop = Some(ss);

        let search_options = self.search_options.clone();

        thread::spawn(move || {
            let mut searcher: Searcher = Searcher::new(sr, search_options);
            searcher.iterative_deepening_search();
            
        });
    
    }

    pub fn stop(&mut self) {
        if self.searcher_stop.is_some() {
            _ = self.searcher_stop.clone().unwrap().send(true);

        }

    }

}
