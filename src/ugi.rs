use std::io::{self, BufRead};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use crate::tools::tt::init_tt;
use crate::board::board::*;
use crate::consts::*;
use crate::search::searcher::*;

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
            let raw_commands: Vec<&str> = line.split_whitespace().collect();

            let mut data: Vec<&str> = vec![];
            for cmd in raw_commands {
                data.push(cmd.clone());

            }
    
            match data[0] {
                "ugi" => {
                    println!("id name nova");
                    println!("id author beck bjella");

                    println!("ugiok");

                }
                "setoption" => {
                    match data[1] {
                        "maxply" => {
                            self.search_options.maxply = data[2].parse().unwrap();

                        }
                        _ => {}

                    }

                    println!("ugiok");
                    
                }
                "setpos" => {
                    let board_str = data[1];
                    let board = str_to_board(board_str, PLAYER_1);

                    self.search_options.board = board;
                    
                }
                "go" => {
                    self.go();
    
                }
                "stop" => {
                    self.stop();
    
                }
                "quit" => {
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

pub fn str_to_board(data: &str, player: f64) -> BoardState {
    let array_data: [usize; 38] = {
        let mut arr = [0; 38];
        for (i, c) in data.chars().take(38).enumerate() {
            arr[i] = c.to_digit(10).unwrap() as usize;
        }
        arr

    };

    BoardState::from(array_data, player)

}
