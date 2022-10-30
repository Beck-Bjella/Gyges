#[macro_use]
mod macros;

mod board;
mod bitboard;
mod bit_twiddles;
mod move_generation;
mod evaluation;
mod engine;
mod transposition_tables;
mod zobrist;

use crate::board::*;
use crate::engine::*;
use crate::move_generation::*;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;


// ====================== GUI ======================

use macroquad::prelude::*;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;

fn window_conf() -> Conf {
    Conf {
        window_title: "Gyges Engine".to_owned(),
        window_height: 900,
        window_width: 1000,
        window_resizable: false,
        ..Default::default() 
    }

}

pub fn draw_three_piece(x: f32, y: f32, radius: f32) {
    draw_circle(x, y, radius, BROWN);
    draw_circle(x, y, radius * (130.0/150.0), BEIGE);
    draw_circle(x, y, radius * (110.0/150.0), BROWN);
    draw_circle(x, y, radius * (90.0/150.0), BEIGE);
    draw_circle(x, y, radius * (70.0/150.0), BROWN);
    draw_circle(x, y, radius * (50.0/150.0), BEIGE);

}

pub fn draw_two_piece(x: f32, y: f32, radius: f32) {
    draw_circle(x, y, radius, BROWN);
    draw_circle(x, y, radius * (130.0/150.0), BEIGE);
    draw_circle(x, y, radius * (110.0/150.0), BROWN);
    draw_circle(x, y, radius * (90.0/150.0), BEIGE);

}

pub fn draw_one_piece(x: f32, y: f32, radius: f32) {
    draw_circle(x, y, radius, BROWN);
    draw_circle(x, y, radius * (130.0/150.0), BEIGE);

}

pub fn draw_statisitics(data: Vec<(&str, String)>) {
    let vertical_spacing = 60;
    let horizontal_position = 725.0;
    let vertical_position = 50.0;
    let font_size = 25.0;

    let mut count = 0;
    for stat in data {
        draw_text(&stat.0, horizontal_position, vertical_position + ((count * vertical_spacing) as f32), font_size, BLACK);
        draw_text(&stat.1, horizontal_position, vertical_position + 20.0 + ((count * vertical_spacing) as f32), font_size, BLACK);

        count += 1;

    }

}

#[derive(PartialEq, Clone)]
pub enum PieceType {
    One,
    Two,
    Three

}

#[derive(Clone)]
pub struct Piece {
    pub pos: (f32, f32),
    pub radus: f32,
    pub piece_type: PieceType,
    pub being_dragged: bool,
    pub being_dropped: bool

}

impl Piece {
    pub fn new(x: f32, y: f32, piece_type: PieceType) -> Piece {
        return Piece {
            pos: (x, y),
            radus: 30.0,
            piece_type: piece_type,
            being_dragged: false,
            being_dropped: false
    
        };

    }

    pub fn is_touching_point(&self, point_x: f32, point_y: f32) -> bool {
        if point_x <= self.pos.0 + self.radus && point_x >= self.pos.0 - self.radus {
            if point_y <= self.pos.1 + self.radus && point_y >= self.pos.1 - self.radus {
                return true;
            
            }

        }

        return false;

    }

    pub fn draw(&self) {
        if self.piece_type == PieceType::Three {
            draw_three_piece(self.pos.0, self.pos.1, self.radus);

        } else if self.piece_type == PieceType::Two {
            draw_two_piece(self.pos.0, self.pos.1, self.radus);

        } else if self.piece_type == PieceType::One {
            draw_one_piece(self.pos.0, self.pos.1, self.radus);

        }

    }

}

pub struct DrawableBoard {
    boardstate: BoardState,

    pieces: Vec<Piece>,
    piece_snap_positions: Vec<(f32, f32)>,
    pos: (f32, f32),
    board_pos: (f32, f32),
    held_piece_idx: usize,
    is_still: bool

}

impl DrawableBoard {
    pub fn new(x: f32, y: f32) -> DrawableBoard {
        let board_pos = (x + 50.0, y + 150.0);

        let mut pieces: Vec<Piece> = vec![];
        pieces.push(Piece::new(x + 100.0, y + 200.0, PieceType::Three));
        pieces.push(Piece::new(x + 200.0, y + 200.0, PieceType::Two));
        pieces.push(Piece::new(x + 300.0, y + 200.0, PieceType::One));
        pieces.push(Piece::new(x + 400.0, y + 200.0, PieceType::One));
        pieces.push(Piece::new(x + 500.0, y + 200.0, PieceType::Two));
        pieces.push(Piece::new(x + 600.0, y + 200.0, PieceType::Three));
        pieces.push(Piece::new(x + 100.0, y + 700.0, PieceType::Three));
        pieces.push(Piece::new(x + 200.0, y + 700.0, PieceType::Two));
        pieces.push(Piece::new(x + 300.0, y + 700.0, PieceType::One));
        pieces.push(Piece::new(x + 400.0, y + 700.0, PieceType::One));
        pieces.push(Piece::new(x + 500.0, y + 700.0, PieceType::Two));
        pieces.push(Piece::new(x + 600.0, y + 700.0, PieceType::Three));

        let mut boardstate = BoardState::new();
        boardstate.set( [3, 2, 1 ,1, 2, 3],
                        [0 ,0 ,0, 0, 0, 0],
                        [0 ,0 ,0, 0, 0, 0],
                        [0 ,0 ,0 ,0, 0, 0],
                        [0 ,0, 0, 0, 0, 0],
                        [3 ,2 ,1 ,1, 2, 3],
                        [0, 0]);

        let mut piece_snap_positions = vec![];
        for y in (0..6).rev() {
            for x in 0..6 {
                let pos = (board_pos.0 + 50.0 + (x * 100) as f32, board_pos.1 + 50.0 + (y * 100) as f32);
                piece_snap_positions.push(pos);

            }

        }
        
        piece_snap_positions.push((x + 350.0, y + 825.0));
        piece_snap_positions.push((x + 350.0, y + 75.0));

        return DrawableBoard {
            boardstate: boardstate,

            pieces: pieces,
            piece_snap_positions: piece_snap_positions,
            pos: (x, y),
            board_pos: board_pos,
            held_piece_idx: usize::MAX,
            is_still: true

        };

    }

    pub fn draw(&self) {
        draw_circle(self.pos.0, self.pos.1, 5.0, RED);
        draw_rectangle_lines(self.pos.0, self.pos.1, 700.0, 900.0, 5.0, RED);

        for i in 0..7 {
            draw_line(self.board_pos.0 + i as f32 * 100.0, self.board_pos.1, self.board_pos.0 + i as f32 * 100.0, self.board_pos.1 + 600.0, 5.0, GRAY);
            draw_line(self.board_pos.0, self.board_pos.1 + i as f32 * 100.0, self.board_pos.0 + 600.0, self.board_pos.1 + i as f32 * 100.0, 5.0, GRAY);
    
        }
    
        draw_rectangle_lines(self.pos.0 + 300.0, self.pos.1 + 25.0, 100.0, 100.0, 10.0, GRAY);
        draw_rectangle_lines(self.pos.0 + 300.0, self.pos.1 + 775.0, 100.0, 100.0, 10.0, GRAY);

        for piece in self.pieces.iter() {
            piece.draw();

        }

    }

    pub fn draw_move(&mut self, mv: Move) {
        if mv.flag == MoveType::Bounce {
            self.draw_arrow(mv.data[1], mv.data[3]);

        } else if mv.flag == MoveType::Drop {
            self.draw_arrow(mv.data[1], mv.data[3]);
            self.draw_arrow(mv.data[3], mv.data[5]);
            
        }

    }

    fn draw_arrow(&mut self, boardpos_1: usize, boardpos_2: usize) {
        let xy_pos_1 = self.piece_snap_positions[boardpos_1];
        let xy_pos_2 = self.piece_snap_positions[boardpos_2];

        draw_line(xy_pos_1.0, xy_pos_1.1, xy_pos_2.0, xy_pos_2.1, 2.5, BLACK);
        draw_circle(xy_pos_2.0, xy_pos_2.1, 5.0, BLACK)

    }

    pub fn update(&mut self) {
        let mouse_pos = mouse_position();

        if !self.is_still {
            let mut piece = self.pieces[self.held_piece_idx].clone();

            if is_mouse_button_released(MouseButton::Left) {
                if piece.being_dragged {
                    let mut closest_pos = (0.0, 0.0);
                    let mut closest_distance = f32::INFINITY;
    
                    for point in self.piece_snap_positions.iter() {
                        let distance: f32 = ((mouse_pos.0 - point.0).abs() + (mouse_pos.1 - point.1).abs()).sqrt();
                        if distance <= closest_distance {
                            closest_distance = distance;
                            closest_pos = point.clone();
    
                        }
    
                    }

                    let mut dropped = false;
                    let mut new_held_piece = usize::MAX;
                    for piece_idx in 0..12 {
                        let mut temp_piece = self.pieces[piece_idx].clone();

                        if temp_piece.pos == closest_pos {
                            dropped = true;
                            new_held_piece = piece_idx;

                            temp_piece.being_dropped = true;
                            self.pieces[piece_idx] = temp_piece;
                            
                            break;
    
                        }
                        
                    }

                    piece.being_dragged = false;
                    piece.pos = closest_pos;

                    self.pieces[self.held_piece_idx] = piece;
                    if !dropped {
                        self.held_piece_idx = usize::MAX;
                        self.is_still = true;

                    } else {
                        self.held_piece_idx = new_held_piece;
                        
                    }
                   
                } else if piece.being_dropped {
                    let mut closest_pos = (0.0, 0.0);
                    let mut closest_distance = f32::INFINITY;
    
                    'snaps: for point in self.piece_snap_positions.iter() {
                        for piece_idx in 0..12 {
                            let temp_piece = self.pieces[piece_idx].clone();
        
                            if &temp_piece.pos == point {
                                continue 'snaps;
        
                            }
                            
                        }

                        let distance: f32 = ((mouse_pos.0 - point.0).abs() + (mouse_pos.1 - point.1).abs()).sqrt();
                        if distance <= closest_distance {
                            closest_distance = distance;
                            closest_pos = point.clone();
    
                        }
    
                    }
                    
                    piece.pos = closest_pos;
                    piece.being_dropped = false;

                    self.pieces[self.held_piece_idx] = piece;
                    self.held_piece_idx = usize::MAX;

                    self.is_still = true;

                }

            } else {
                piece.pos = mouse_pos;
                
                self.pieces[self.held_piece_idx] = piece;

            }
 
        } else {
            let mut piece_selected = false;
            for piece_idx in 0..12 {
                let mut piece = self.pieces[piece_idx].clone();
    
                if piece.is_touching_point(mouse_pos.0, mouse_pos.1) && is_mouse_button_pressed(MouseButton::Left) && !piece.being_dropped && !piece.being_dragged {
                    piece_selected = true;

                    piece.being_dragged = true;
                    self.held_piece_idx = piece_idx;

                    self.pieces[piece_idx] = piece;

                    self.is_still = false;

                    break;

                }
    
            }
            
            if !piece_selected {
                for i in 0..38 {
                    self.boardstate.data[i] = 0;

                }

                for (snap_pos_idx, snap_pos) in self.piece_snap_positions.iter().enumerate() {
                    for piece_idx in 0..12 {
                        let piece = self.pieces[piece_idx].clone();

                        if &piece.pos == snap_pos {
                            if piece.piece_type == PieceType::One {
                                self.boardstate.data[snap_pos_idx] = 1

                            } else if piece.piece_type == PieceType::Two {
                                self.boardstate.data[snap_pos_idx] = 2

                            } else if piece.piece_type == PieceType::Three {
                                 self.boardstate.data[snap_pos_idx] = 3

                            }

                        }

                    }

                }

            }

        }

    }

}

#[macroquad::main(window_conf)]
async fn main() {
    let mut drawable_board = DrawableBoard::new(0.0, 0.0);

    let (board_sender, board_reciver): (Sender<SearchInput>, Receiver<SearchInput>) = mpsc::channel();
    let (stop_sender, stop_reciver): (Sender<bool>, Receiver<bool>) = mpsc::channel();
    let (results_sender, results_reciver): (Sender<SearchData>, Receiver<SearchData>) = mpsc::channel();

    thread::spawn(move || {
        let mut engine = Engine::new(board_reciver, stop_reciver, results_sender);
        engine.start();
        
    });

    let mut current_best_search = SearchData::new();
    let mut previous_board_state: BoardState = BoardState::new();
    loop {
        clear_background(BEIGE);

        drawable_board.update();
        drawable_board.draw();

        drawable_board.draw_move(current_best_search.best_move);

        let depth_text = current_best_search.depth.to_string();
        let gameover_text = current_best_search.game_over.to_string();
        let winner_text = current_best_search.winner.to_string();
        let time_searched_text = current_best_search.search_time.to_string();
        let average_branching_factor_text = current_best_search.average_branching_factor.to_string();
        let nodes_text = current_best_search.nodes.to_string();
        let leafs_text = current_best_search.leafs.to_string();
        let nps_text = current_best_search.nps.to_string();
        let lps_text = current_best_search.lps.to_string();
        let tt_hits_text = current_best_search.tt_hits.to_string();
        let tt_exacts_text = current_best_search.tt_exacts.to_string();
        let tt_cuts_text = current_best_search.tt_cuts.to_string();
        let alphabeta_cuts_text = current_best_search.beta_cuts.to_string();

        let data = vec![
            ("Depth", depth_text),
            ("GAMEOVER", gameover_text),
            ("WINNER", winner_text),
            ("_____________________", String::from("")),
            ("Time Searched", time_searched_text),
            ("Average Branching Factor", average_branching_factor_text),
            ("Nodes Searched", nodes_text),
            ("NPS", nps_text),
            ("Leafs Searched", leafs_text),
            ("LPS", lps_text),
            ("TT Hits", tt_hits_text),
            ("TT Exacts", tt_exacts_text),
            ("TT Cuts", tt_cuts_text),
            ("Alphabeta Cuts", alphabeta_cuts_text),
            
        ];

        draw_statisitics(data);

        let current_board = drawable_board.boardstate.clone();
        if current_board != previous_board_state {            
            current_best_search = SearchData::new();
    
            _ = stop_sender.send(true);
            _ = board_sender.send(SearchInput::new(current_board, MAX_SEARCH_PLY));
            
        }
        previous_board_state = current_board;
        
        let results = results_reciver.try_recv();
        match results {
            Ok(_) => {
                let unwraped = results.unwrap();
                current_best_search = unwraped;
                
            },
            Err(TryRecvError::Disconnected) => {},
            Err(TryRecvError::Empty) => {}

        }
        
        next_frame().await

    }

}

// ====================== GUI ======================
