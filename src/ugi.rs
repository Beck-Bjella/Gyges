use std::io::{self};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use crate::board::board::*;
use crate::search::searcher::*;

pub struct Ugi {
    searching_thread: Option<thread::JoinHandle<()>>,
    searcher_stop: Option<Sender<bool>>,
    searching: bool,
    search_options: SearchOptions,
   
}

impl Ugi {
    pub fn new() -> Ugi {
        Ugi {
            searching_thread: Option::None,
            searcher_stop: Option::None,
            searching: false, 
            search_options: SearchOptions::new(),
            
        }

    }

    pub fn start(&mut self) {
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
                        "MaxTime" => {
                            self.search_options.maxtime = Some(value.parse().unwrap());

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

    pub fn go(&mut self) {
        self.searching = true;

        let (ss, sr): (Sender<bool>, Receiver<bool>) = mpsc::channel();
        self.searcher_stop = Some(ss);

        let search_options = self.search_options.clone();

        self.searching_thread = Some(thread::spawn(move || {
            let mut searcher: Searcher = Searcher::new(sr, search_options);
            searcher.iterative_deepening_search();
            
        }));
    
    }

    pub fn stop(&mut self) {
        if self.searching {
            _ = self.searcher_stop.clone().unwrap().send(true);
            self.searching_thread.take().unwrap().join().unwrap();
            
        }

        self.searcher_stop = Option::None;
        self.searching_thread = Option::None;
        self.searching = false;
    
    }

    pub fn ugi(&mut self) {
        println!("id name nova");
        println!("id author beck-bjella");                
        println!("option MaxPly");
        println!("option MaxTime");
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
