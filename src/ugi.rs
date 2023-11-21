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
        let stdin = io::stdin();

        loop {
            let mut input = String::new();
            stdin.read_line(&mut input).expect("Failed to read line from stdin");
            
            let raw_commands: Vec<&str> = input.split_whitespace().collect();
            let mut data: Vec<&str> = vec![];
            for cmd in raw_commands {
                data.push(cmd.clone());

            }

            if data.len() == 0 {
                continue;

            }
    
            match data[0] {
                "ugi" => {
                    id_output();
                    
                }
                "setoption" => {
                    self.set_option(data[1].to_string(), data[2].to_string());
                    
                }
                "setpos" => {
                    self.set_position(data[1]);

                }
                "go" => {
                    self.go();
    
                }
                "search" => {
                    self.search(data[1].to_string());
    
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

        self.searching_thread = Some(thread::spawn(move || {
            let mut searcher: Searcher = Searcher::new(sr, search_options);
            searcher.iterative_deepening_search();
            
        }));
    
    }

    pub fn search(&mut self, ply_string: String) {
        let ply = ply_string.parse().unwrap();

        init_tt(2usize.pow(24));

        let (ss, sr): (Sender<bool>, Receiver<bool>) = mpsc::channel();
        self.searcher_stop = Some(ss);

        let search_options = self.search_options.clone();

        self.searching_thread = Some(thread::spawn(move || {
            let mut searcher: Searcher = Searcher::new(sr, search_options);
            searcher.search_depth(ply);
            
        }));
    
    }

    pub fn stop(&mut self) {
        if self.searcher_stop.is_some() && self.searching_thread.is_some() {
            _ = self.searcher_stop.clone().unwrap().send(true);
            self.searching_thread.take().unwrap().join().unwrap();

        } else {
            ugiok_output();

        }

        self.searcher_stop = Option::None;
        self.searching_thread = Option::None;
    
    }

    pub fn set_position(&mut self, board_str: &str) {
        self.search_options.board = BoardState::from(board_str);

        ugiok_output();

    }

    pub fn set_option(&mut self, option: String, value: String) {
        match option.as_str() {
            "maxply" => {
                self.search_options.maxply = value.parse().unwrap();

            }
            _ => {}

        }

        ugiok_output();

    }
    
}

pub fn info_output(search_data: SearchData) {
    print!("info ");
    print!("ply {} ", search_data.ply);
    print!("bestmove {} ", search_data.best_move.as_ugi());
    print!("score {} ", search_data.best_move.score);
    println!("time {} ", search_data.search_time);

}

pub fn id_output() {
    println!("id name nova");
    println!("id author beck-bjella");

}

pub fn ugiok_output() {
    println!("ugiok");

}
