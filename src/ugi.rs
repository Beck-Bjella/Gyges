use std::io::{self};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use crate::tools::tt::init_tt;
use crate::board::board::*;
use crate::search::searcher::*;

pub struct Ugi {
    pub searcher_stop: Option<Sender<bool>>,
    pub search_options: SearchOptions,
    pub searching_thread: Option<thread::JoinHandle<()>>,

}

impl Ugi {
    pub fn new() -> Ugi {
        Ugi {
            searcher_stop: Option::None,
            search_options: SearchOptions::new(),
            searching_thread: Option::None,

        }

    }

    pub fn start(&mut self) {
        self.init();

        let stdin = io::stdin();
        loop {
            let mut input = String::new();
            stdin.read_line(&mut input).expect("Failed to read line from stdin");

            let mut raw_commands: Vec<&str> = input.split_whitespace().collect();
            if raw_commands.len() == 0 {
                continue;

            }

            let base_cmd = raw_commands.remove(0);
            match base_cmd {
                "ugi" => {
                    self.ugi();

                },
                "isready" => {
                    println!("readyok");

                },
                "setoption" => {
                    let name = raw_commands[0].clone();
                    let value = raw_commands[1].clone();

                    match name {
                        "MaxPly" => {
                            self.search_options.maxply = value.parse().unwrap();

                        }
                        _ => {
                            println!("Unknown option: {}", name);

                        }

                    }

                }
                "setpos" => {
                    let board_str = raw_commands[0].clone();
                    self.search_options.board = BoardState::from(board_str);

                },
                "go" => {
                    self.go();

                },
                "stop" => {
                    self.stop();

                },
                "quit" => {
                    break;

                },
                _ => {
                    println!("Unknown command: {}", base_cmd);

                }

            }
           
        }

        self.stop();
    
    }

    pub fn init(&mut self) {
        init_tt(2usize.pow(24));

    }
    
    pub fn go(&mut self) {
        let (ss, sr): (Sender<bool>, Receiver<bool>) = mpsc::channel();
        self.searcher_stop = Some(ss);

        let search_options = self.search_options.clone();

        self.searching_thread = Some(thread::spawn(move || {
            let mut searcher: Searcher = Searcher::new(sr, search_options);
            searcher.iterative_deepening_search();
            
        }));
    
    }

    pub fn stop(&mut self) {
        if self.searcher_stop.is_some() && self.searching_thread.is_some() {
            _ = self.searcher_stop.clone().unwrap().send(true);
            self.searching_thread.take().unwrap().join().unwrap();
            
        }

        self.searcher_stop = Option::None;
        self.searching_thread = Option::None;
    
    }

    pub fn ugi(&mut self) {
        println!("id name nova");
        println!("id author beck-bjella");                
        println!("option MaxPly");
        println!("ugiok");

    }
    
}

pub fn info_output(search_data: SearchData) {
    print!("info ");
    print!("ply {} ", search_data.ply);
    print!("bestmove {} ", search_data.best_move.as_ugi());
    print!("score {} ", search_data.best_move.score);
    println!("time {} ", search_data.search_time);

}

pub fn best_move_output(search_data: SearchData) {
    println!("bestmove {}", search_data.best_move.as_ugi());

}
