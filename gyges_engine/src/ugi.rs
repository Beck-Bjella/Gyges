//! Ugi implimentation.
//! 

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::thread;

use gyges::{MoveGen, board::*};

use crate::search::*;
use crate::consts::*;

pub struct Ugi {
    /// The orchestrator thread that owns the [Searcher] and runs it.
    /// Set when a search is running, joined on stop.
    main_thread: Option<thread::JoinHandle<()>>,
    /// Shared stop flag — flipped by `stop` UGI command (or by the
    /// orchestrator when IDS exits) and polled by every searcher.
    stop_signal: Option<Arc<AtomicBool>>,
    searching: bool,
    search_options: SearchOptions,

}

impl Ugi {
    pub fn new() -> Ugi {
        Ugi {
            main_thread: Option::None,
            stop_signal: Option::None,
            searching: false,
            search_options: SearchOptions::new(),

        }

    }

    /// Initializes the engine
    pub fn init(&self) {
        // Initialize transposition table
        init_tt(2usize.pow(22)); // 400 MB

        // Load default network
        let _ = network::load_network("./weights/default.bin");

    }

    /// Starts the UGI loop
    pub fn start(&mut self) {
        self.init();

        println!("Gyges UGI Engine v1.2.0");

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
                    println!("id author Beck-Bjella");
                    println!("option maxPly");
                    println!("option maxTime");
                    println!("option maxNodes");
                    println!("option randomize");
                    println!("option nn");
                    println!("option weightsPath");
                    println!("option threads");
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
                Some(&"eval") => {
                    let mut mg = MoveGen::default();
                    let ctx = evaluation::EvaluationContext::new(&mut self.search_options.board, &mut mg);
                    ctx.print();

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
            Some(&"maxNodes") => {
                if let Some(value_str) = raw_commands.get(2) {
                    self.search_options.maxnodes = Some(value_str.parse::<usize>().unwrap());

                } else {
                    println!("Unknown Command: '{}'", trimmed);

                }

            }
            Some(&"randomize") => {
                if let Some(value_str) = raw_commands.get(2) {
                    match *value_str {
                        "true" | "1" | "on" => self.search_options.randomize = true,
                        "false" | "0" | "off" => self.search_options.randomize = false,
                        _ => println!("Unknown Command: '{}'", trimmed),
                    }

                } else {
                    println!("Unknown Command: '{}'", trimmed);

                }

            }
            Some(&"nn") => {
                if let Some(value_str) = raw_commands.get(2) {
                    match *value_str {
                        "true" | "1" | "on" => self.search_options.nn = true,
                        "false" | "0" | "off" => self.search_options.nn = false,
                        _ => println!("Unknown Command: '{}'", trimmed),
                    }

                } else {
                    println!("Unknown Command: '{}'", trimmed);

                }

            }
            Some(&"threads") => {
                if let Some(value_str) = raw_commands.get(2) {
                    match value_str.parse::<usize>() {
                        Ok(n) if n >= 1 => self.search_options.threads = n,
                        _ => println!("Unknown Command: '{}'", trimmed),
                    }

                } else {
                    println!("Unknown Command: '{}'", trimmed);

                }

            }
            Some(&"weightsPath") => {
                if let Some(path) = raw_commands.get(2) {
                    if let Err(e) = network::load_network(path) {
                        println!("info string weight load failed: {}", e);

                    }

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
            Some(&"draw") => {
                self.search_options.board = BoardState::from(DRAW_BOARD);

            },
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
        // Join any previous search before touching the TT — threads from the
        // last `go` may still be unwinding when the next command arrives.
        self.stop();
        unsafe { tt().reset() }; // Reset tt before new search

        // Display eval type
        match (self.search_options.nn, network::network_name()) {
            (true, Some(name)) => println!("info string NN evaluation using {} enabled", name),
            _ => println!("info string classical evaluation enabled"),

        }

        self.searching = true;

        let stop_signal = Arc::new(AtomicBool::new(false));
        self.stop_signal = Some(stop_signal.clone());

        let options = self.search_options.clone();
        let signal = stop_signal.clone();
        self.main_thread = Some(thread::spawn(move || {
            let mut searcher = Searcher::new(options, signal.clone());
            searcher.run();
            signal.store(true, AtomicOrdering::Relaxed);

        }));

    }

    pub fn stop(&mut self) {
        if self.searching {
            if let Some(signal) = &self.stop_signal {
                signal.store(true, AtomicOrdering::Relaxed);

            }

            if let Some(handle) = self.main_thread.take() {
                handle.join().unwrap();

            }

        }

        self.stop_signal = Option::None;
        self.main_thread = Option::None;
        self.searching = false;

    }

}

impl Default for Ugi {
    fn default() -> Self {
        Self::new()

    }

}

pub fn info_output(search_data: SearchData) {
    print!("info ");
    print!("ply {} ", search_data.ply);
    print!("bestmove {} ", search_data.best_move);
    print!("score {:.3} ", search_data.best_move.score);
    print!("nodes {} ", search_data.nodes);
    print!("nps {:.3} ", search_data.nps);
    print!("time {:.3} ", search_data.elapsed_time);
    print!("pv ");
    for mv in search_data.pv.iter() {
        print!("{} ", mv);

    }
    println!()

}

pub fn best_move_output(search_data: SearchData) {
    if search_data.is_draw {
        println!("bestmove null score {:.3} time {:.3}", search_data.best_move.score, search_data.elapsed_time);

    } else {
        println!("bestmove {} score {:.3} time {:.3}", search_data.best_move, search_data.best_move.score, search_data.elapsed_time);

    }

}
