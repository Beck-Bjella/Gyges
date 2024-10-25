//! Ugi implimentation.
//! 



use std::io;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use rayon;

use gyges::board::*;

use crate::search::*;
use crate::consts::*;

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
        self.init();

        println!("Gyges UGI Engine v1.0.2");

        let stdin = io::stdin();
        loop {
            let mut input = String::new();
            stdin.read_line(&mut input).expect("Failed to read line from stdin");
            
            let trimmed = input.trim();
            
            let raw_commands: Vec<&str> = trimmed.split_whitespace().collect();
            if raw_commands.is_empty() {
                continue;

            }

            match raw_commands.first() {
                Some(&"ugi") => {
                    println!("id name Helios");
                    println!("id author beck-bjella");                
                    println!("option maxply");
                    println!("option maxtime");
                    println!("option tt_enabled");
                    println!("ugiok");

                },
                Some(&"isready") => {
                    println!("readyok");

                },
                Some(&"setoption") => {
                    self.parse_option(trimmed, raw_commands);

                }
                Some(&"setpos") => {
                    self.parse_position(trimmed, raw_commands)

                },
                Some(&"showpos") => {
                    println!("{}", self.search_options.board);

                },
                Some(&"go") => {
                    self.go();

                },
                Some(&"stop") => {
                    self.stop();

                },
                Some(&"quit") => {
                    break;

                },
                Some(_) | None => {
                    println!("Unknown Command: '{}'", trimmed);

                },

            }
           
        }

        self.stop();
        unsafe{ tt().de_alloc() };
    
    }

    pub fn init(&self) {
        // Initialize transposition table
        init_tt(2usize.pow(22)); // 400 MB

        // Configure rayon
        rayon::ThreadPoolBuilder::new().num_threads(4).build_global().unwrap();

    }

    pub fn parse_option(&mut self, trimmed: &str, raw_commands: Vec<&str>) {
        match raw_commands.get(1) {
            Some(&"maxPly") => {
                if let Some(value_str) = raw_commands.get(2) {
                    self.search_options.maxply = Some(value_str.parse::<i8>().unwrap());

                } else {
                    println!("Unknown Command: '{}'", trimmed);

                }
                
            }
            Some(&"maxTime") => {
                if let Some(value_str) = raw_commands.get(2) {
                    self.search_options.maxtime = Some(value_str.parse::<f64>().unwrap());

                } else {
                    println!("Unknown Command: '{}'", trimmed);

                }

            }
            Some(_) | None => {
                println!("Unknown Command: '{}'", trimmed);

            }

        }

    }

    pub fn parse_position(&mut self, trimmed: &str, raw_commands: Vec<&str>) {
        match raw_commands.get(1) {
            Some(&"start") => {
                self.search_options.board = BoardState::from(STARTING_BOARD);

            },
            Some(&"bench") => {
                self.search_options.board = BoardState::from(BENCH_BOARD);

            },
            Some(&"test") => {
                self.search_options.board = BoardState::from(TEST_BOARD);

            }
            Some(&"data") => {
                if let Some(board_str) = raw_commands.get(2) {
                    self.search_options.board = BoardState::from(*board_str);

                } else {
                    println!("Unknown Command: '{}'", trimmed);

                }

            },
            Some(_) | None => {
                println!("Unknown Command: '{}'", trimmed);

            }

        }

    }

    pub fn go(&mut self) {
        unsafe { tt().reset() }; // Reset tt before new search

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

}

impl Default for Ugi {
    fn default() -> Self {
        Self::new()

    }

}

pub fn info_output(search_data: SearchData, search_stats: SearchStats) {
    print!("info ");
    print!("ply {} ", search_data.ply);
    print!("bestmove {} ", search_data.best_move);
    print!("score {} ", search_data.best_move.score);
    print!("nodes {} ", search_stats.nodes);
    print!("nps {} ", search_stats.nps);
    print!("time {} ", search_stats.search_time);
    print!("pv ");
    for mv in search_data.pv.iter() {
        print!("{} ", mv);

    }
    println!()

}

pub fn best_move_output(search_data: SearchData) {
    println!("bestmove {}", search_data.best_move);

}
