use min_max::*;
use std::cmp::Ordering;

static ONE_PIECE: usize = 1;
static TWO_PIECE: usize = 2;
static THREE_PIECE: usize = 3;

static PLAYER_1_GOAL: usize = 7;
static PLAYER_2_GOAL: usize = 8;

struct Board {
    data: [[usize; 6]; 6],
    goals: [usize; 2],

    one_moves: [[usize; 1]; 4],
    two_moves: [[usize; 2]; 12],
    three_moves: [[usize; 3]; 36]
 
}

impl Board {
    fn new() -> Board {
        return Board {
            data: [ [0 ,0, 0 ,0, 0, 0],
                    [0 ,0 ,0 ,0, 0, 0],
                    [0 ,0 ,0 ,0, 0, 0],
                    [0 ,0 ,0 ,0, 0, 0],
                    [0 ,0, 0 ,0, 0, 0], 
                    [0 ,0 ,0 ,0, 0, 0]],
            goals: [0, 0],

            one_moves: [[0], [1], [2], [3]],
            two_moves: [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 1], [2, 2], [2, 3], [3, 2], [3, 3], [3, 0], [0, 3]],
            three_moves: [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 2], [1, 2, 1], [2, 1, 1], [1, 2, 2], [2, 1, 2], [2, 2, 1], [2, 2, 3], [2, 3, 2], [3, 2, 2], [2, 3, 3], [3, 2, 3], [3, 3, 2], [3, 3, 0], [3, 0, 3], [0, 3, 3], [3, 0, 0], [0, 3, 0], [0, 0, 3], [3, 0, 1], [1, 0, 3], [0, 1, 2], [2, 1, 0], [1, 2, 3], [3, 2, 1], [2, 3, 0], [0, 3, 2]],
    
        };
        
    }

    fn set(&mut self, data: [[usize; 6]; 6], goals: [usize; 2]) {
        self.data = data;
        self.goals = goals;

    }

    fn _print(&self) {
        println!(" ");
        if self.goals[0] == 0 {
            println!("                .");

        } else {
            println!("                {}", self.goals[0]);

        }
        println!(" ");
        println!(" ");

        for y in 0..6 {
            for x in 0..6 {
                if self.data[y][x] == 0 {
                    print!("    .");
                } else {
                    print!("    {}", self.data[y][x]);

                }
               
            }
            println!(" ");
            println!(" ");
        }

        println!(" ");
        if self.goals[1] == 0 {
            println!("                .");

        } else {
            println!("                {}", self.goals[1]);

        }
        println!(" ");
    
    }
    
    fn evalulate(&mut self) -> usize {
        let starting_value = usize::MAX / 2;
        let player_1_move_count = self.valid_move_count(1);
        let player_2_move_count = self.valid_move_count(2);

        let score = starting_value + (player_2_move_count - player_1_move_count);

        return score;
    
    }

    fn make_move(&mut self, mv: &[usize; 9]) {
        let step1 = [mv[6], mv[7], mv[8]];
        let step2 = [mv[3], mv[4], mv[5]];
        let step3 = [mv[0], mv[1], mv[2]];
        
        if step1[1] != 9 {
            self.data[step1[2]][step1[1]] = step1[0];
            self.data[step2[2]][step2[1]] = step2[0];
            self.data[step3[2]][step3[1]] = step3[0];

        } else {
            if step3[1] == PLAYER_1_GOAL {
                self.goals[0] = step3[0];
                self.data[step2[2]][step2[1]] = 0;

            } else if step3[1] == PLAYER_2_GOAL {
                self.goals[1] = step3[0];
                self.data[step2[2]][step2[1]] = 0;

            } else {
                self.data[step2[2]][step2[1]] = 0;
                self.data[step3[2]][step3[1]] = step3[0];

            }

        }

    }

    fn undo_move(&mut self, mv: &[usize; 9]) {
        let step1 = [mv[6], mv[7], mv[8]];
        let step2 = [mv[3], mv[4], mv[5]];
        let step3 = [mv[0], mv[1], mv[2]];

        if step1[1] != 9 {
            self.data[step3[2]][step3[1]] = step1[0];
            self.data[step2[2]][step2[1]] = step3[0];
            self.data[step1[2]][step1[1]] = step2[0];
            
        } else {
            if self.goals[0] != 0 {
                self.goals[0] = 0;
                self.data[step2[2]][step2[1]] = step3[0];

            } else if self.goals[1] != 0 {
                self.goals[1] = 0;
                self.data[step2[2]][step2[1]] = step3[0];

            } else {
                self.data[step2[2]][step2[1]] = step3[0];
                self.data[step3[2]][step3[1]] = 0;

            }

        }

    }
    
    fn active_lines(&self) -> [usize; 2] {
        let mut player_1_set = false;
    
        let mut player_1_active_line = 9;
        let mut player_2_active_line = 9;
    
        for y in 0..6 {
            for x in 0..6 {
                if self.data[y][x] != 0 {
                    if !player_1_set {
                        player_1_active_line = y;
                        player_1_set = true;
    
                    }
                    player_2_active_line = y;
                    
                }
            }
        }
    
        return [player_1_active_line, player_2_active_line];
    
    }

    fn mini_max(&mut self, mut alpha: usize, mut beta: usize, is_maximisizing: bool, depth: i8) -> usize {
        if depth == 0 {
            let score = self.evalulate();
            return score;
    
        }

        if is_maximisizing {
            if self.valid_threat_count(2) > 0 {
                return usize::MAX;
        
            }

            let current_moves = self.valid_moves(2);
            
            let mut max_eval: usize = usize::MIN;
            let mut used_moves: Vec<[usize; 9]> = vec![];
            for mv in current_moves.iter() {
                if used_moves.contains(mv) {
                    continue;
                }
                
                used_moves.push(*mv);

                self.make_move(&mv);
    
                let curr_eval: usize = self.mini_max(alpha, beta, false, depth - 1);
                
                self.undo_move(&mv);

                if curr_eval > max_eval {
                    max_eval = curr_eval;

                } 

                alpha = max!(alpha, curr_eval);
                if beta <= alpha {
                    break
    
                }

            }
            
            return max_eval;
    
        } else {
            if self.valid_threat_count(1) > 0 {
                return usize::MIN;
        
            }

            let current_moves = self.valid_moves(1);
            
            let mut min_eval: usize = usize::MAX;
            let mut used_moves: Vec<[usize; 9]> = vec![];
            for mv in current_moves.iter() {
                if used_moves.contains(mv) {
                    continue;
                }
    
                used_moves.push(*mv);
    
                self.make_move(&mv);
                
                let curr_eval: usize = self.mini_max(alpha, beta, true, depth - 1);

                self.undo_move(&mv);

                if curr_eval < min_eval {
                    min_eval = curr_eval;

                } 

                beta = min!(beta, curr_eval);
                if beta <= alpha {
                    break;
    
                }

            }

            return min_eval;
    
        }
    
    }

    fn get_best_move(&mut self, depth: i8) -> (usize, [usize; 9], f64) {
        let start_time = std::time::Instant::now();

        self.is_valid();

        let mut current_moves = self.valid_moves(2);
        current_moves = self.order_moves(current_moves);

        for mv in current_moves.iter() {
            if mv[1] == PLAYER_1_GOAL {
                return (usize::MAX, *mv, start_time.elapsed().as_secs_f64());   

            }

        }

        let mut alpha = usize::MIN;
        let beta = usize::MAX;
        let mut max_eval: [usize; 2] = [usize::MIN, usize::MIN];
        
        let mut used_moves: Vec<[usize; 9]> = vec![];
        for (move_idx, mv) in current_moves.iter().enumerate() {
            if used_moves.contains(mv) {
                println!("Indix {} is a dupe", move_idx);
                continue;
            }
    
            self.make_move(&mv);

            if self.is_tie() {
                self.undo_move(&mv);
                continue;

            }

            used_moves.push(*mv);
            
            println!("Starting Index {}", move_idx);
            
            let temp_eval: usize = self.mini_max(alpha, beta, false, depth - 1);
            let eval: [usize; 2] = [temp_eval, move_idx];

            self.undo_move(&mv);

            if eval[0] > max_eval[0] {
                max_eval = eval;

            }

            alpha = max!(alpha, eval[0]);
            if beta <= alpha {
                break

            }

        }

        return (max_eval[0], current_moves[max_eval[1]], start_time.elapsed().as_secs_f64());
        
    }

    fn is_valid(&self) {
        let mut one_count = 0;
        let mut two_count = 0;
        let mut three_count = 0;

        for y in 0..6 {
            for x in 0..6 {
                if self.data[y][x] == 1 {
                    one_count += 1;

                } else if self.data[y][x] == 2 {
                    two_count += 1;

                } else if self.data[y][x] == 3 {
                    three_count += 1;

                } 
                

            }

        }

        if !(one_count == 4 && two_count == 4 && three_count == 4) {
            
        panic!("ERROR INVALD BOARD");

        }

    }

    fn is_tie(&mut self) -> bool {
        let player_1_move_count = self.valid_move_count(1);
        let player_2_move_count = self.valid_move_count(2);

        if player_1_move_count == 0 || player_2_move_count == 0 {
            return true;

        }

        return false;

    }

    fn order_moves(&mut self, moves: Vec<[usize; 9]>) -> Vec<[usize; 9]> {
        let mut moves_to_sort: Vec<[usize; 2]> = vec![];
        
        for (mv_idx, mv) in moves.iter().enumerate() {
            let mut predicted_score = 0;

            self.make_move(&mv);

            let threats = self.valid_threat_count(2);

            self.undo_move(&mv);

            predicted_score -= threats;
    
            if mv[7] == 9 {
                predicted_score -= 100;
    
            } else if mv[7] != 9 {
                predicted_score -= 200;
    
            }
    
            moves_to_sort.push([predicted_score, mv_idx]);
            
        }
    
        moves_to_sort.sort_by(|a, b| {
            if a[0] < b[0] {
                Ordering::Less
                
            } else if a[0] == b[0] {
                Ordering::Equal
    
            } else {
                Ordering::Greater
    
            }
    
        });
    
        let mut ordered_moves: Vec<[usize; 9]> = vec![];
    
        for item in &moves_to_sort {
            ordered_moves.push(moves[item[1]].clone());
            
        }
    
        return ordered_moves;
    
    }

    fn _set_starting_line(&mut self, line: [usize; 6], player: i8) {
        if player == 1 {
            for x in 0..6 {
                self.data[0][x] = line[x];

            }

        } else if player == 2 {
            for x in 0..6 {
                self.data[5][x] = line[x];

            }

        }

    }

    fn _flip(&mut self) {
        let mut temp_data: [[usize; 6]; 6] = [  [0 ,0 ,0 ,0, 0, 0],
                                                [0 ,0 ,0 ,0, 0, 0],
                                                [0 ,0 ,0 ,0, 0, 0],
                                                [0 ,0 ,0 ,0, 0, 0],
                                                [0 ,0 ,0 ,0, 0, 0], 
                                                [0 ,0 ,0 ,0, 0, 0]];
    
        for y in 0..6 {
            for x in 0..6 {
                let piece = self.data[y][x];
    
                temp_data[5 - y][5 - x] = piece;
    
            }
    
        }
    
        self.data = temp_data;
    
    }

    // ------------------------- Move Gen -------------------------

    fn valid_moves(&mut self, player: i8) -> Vec<[usize; 9]> {
        let active_lines = self.active_lines();
    
        if player == 1 {
            let mut player_1_drops: Vec<[usize; 2]> = vec![];
            for y in 0..active_lines[1] + 1 {
                for x in 0..6 {
                    if self.data[y][x] == 0 {
                        player_1_drops.push([x, y]);
    
                    }
    
                }
    
            }
            
            
            let mut player_1_moves: Vec<[usize; 9]> = vec![];
            
            for x in 0..6 {
                if self.data[active_lines[0]][x] != 0 {
                    let starting_piece: [usize; 2] = [x, active_lines[0]];
                    let starting_piece_type: usize = self.data[starting_piece[1]][starting_piece[0]];

                    let mut previous_path: Vec<[i8; 2]> = vec![[x as i8, active_lines[0] as i8]];
                    let mut previous_banned_bounces: Vec<[i8; 2]> = vec![];

                    player_1_drops.push(starting_piece);
                    
                    self.data[starting_piece[1]][starting_piece[0]] = 0;

                    let mut moves = self.get_piece_moves(&starting_piece, starting_piece_type, &starting_piece, starting_piece_type, &mut previous_path, &mut previous_banned_bounces, 1, &player_1_drops);
                    player_1_moves.append(&mut moves);

                    self.data[starting_piece[1]][starting_piece[0]] = starting_piece_type;

                    player_1_drops.pop();

                }
    
            }
    
            return player_1_moves;
    
        } else {
            let mut player_2_drops: Vec<[usize; 2]> = vec![];
            for y in (active_lines[0]..6).rev() {
                for x in 0..6 {
                    if self.data[y][x] == 0 {
                        player_2_drops.push([x, y]);
    
                    }
    
                }
                
            }
    
            let mut player_2_moves: Vec<[usize; 9]> = vec![];
    
            for x in 0..6 {
                if self.data[active_lines[1]][x] != 0 {
                    let starting_piece: [usize; 2] = [x, active_lines[1]];
                    let starting_piece_type: usize = self.data[starting_piece[1]][starting_piece[0]];

                    let mut previous_path: Vec<[i8; 2]> = vec![[x as i8, active_lines[1] as i8]];
                    let mut previous_banned_bounces: Vec<[i8; 2]> = vec![];

                    player_2_drops.push(starting_piece);
                    
                    self.data[starting_piece[1]][starting_piece[0]] = 0;

                    let mut moves = self.get_piece_moves(&starting_piece, starting_piece_type, &starting_piece, starting_piece_type, &mut previous_path, &mut previous_banned_bounces, 2, &player_2_drops);
                    player_2_moves.append(&mut moves);

                    self.data[starting_piece[1]][starting_piece[0]] = starting_piece_type;

                    player_2_drops.pop();

                }
    
            }
    
            return player_2_moves;
    
        }
    
    }

    fn get_piece_moves(&self, current_piece: &[usize; 2], current_piece_type: usize, starting_piece: &[usize; 2], starting_piece_type: usize, previous_path: &mut Vec<[i8; 2]>, previous_banned_bounces: &mut Vec<[i8; 2]>, player: i8, current_player_drops: &Vec<[usize; 2]>) -> Vec<[usize; 9]> {
        let mut final_moves: Vec<[usize; 9]> = vec![];
        
        if current_piece_type == ONE_PIECE {
            for path in self.one_moves.iter() {
                let mut current_x: i8 = current_piece[0] as i8;
                let mut current_y: i8 = current_piece[1] as i8;
        
                let mut current_path: Vec<[i8; 2]> = vec![];
        
                'step: for step_idx in 0..path.len() {
                    let step = path[step_idx];
    
                    for (item_idx, item) in previous_path.iter().enumerate() {
                        if item == &[current_x, current_y] {
                            let mut temp_curr_x = current_x;
                            let mut temp_curr_y = current_y;
    
                            match step as i8{
                                0=> temp_curr_x -= 1,
                                1=> temp_curr_y -= 1,
                                2=> temp_curr_x  += 1,
                                3=> temp_curr_y  += 1,
                                _=> println!("ERROR"),
                              }
                        
                            if item_idx != 0 && previous_path[item_idx - 1] == [temp_curr_x, temp_curr_y] {
                                break 'step;
    
                            }
    
                            if item_idx != previous_path.len() - 1 && previous_path[item_idx + 1] == [temp_curr_x, temp_curr_y] {
                                break 'step;
    
                            }
    
                        }
    
                    }

                    match step as i8{
                        0=> current_x -= 1,
                        1=> current_y -= 1,
                        2=> current_x  += 1,
                        3=> current_y  += 1,
                        _=> println!("ERROR"),
                    }
    
                    if  current_x < 0 || current_x > 5 {
                        break;
                    } 
        
                    if step_idx == path.len() - 1 {
                        if current_y == 6 && player == 1 {
                            final_moves.push([starting_piece_type, PLAYER_2_GOAL, PLAYER_2_GOAL, 0, starting_piece[0], starting_piece[1], 9, 9, 9]);
        
                        } else if current_y == -1 && player == 2{
                            final_moves.push([starting_piece_type, PLAYER_1_GOAL, PLAYER_1_GOAL, 0, starting_piece[0], starting_piece[1], 9, 9, 9]);
                            
                        }
        
                    }
    
                    if current_y < 0 || current_y > 5 {
                        break 'step;
                    } 
    
                    if step_idx != path.len() - 1 {
                        if self.data[current_y as usize][current_x as usize] != 0 {
                            break 'step;
    
                        } 
                    }
    
                    current_path.push([current_x, current_y]);
    
                    if step_idx == path.len() - 1 {
                        if self.data[current_y as usize][current_x as usize] != 0 {
                            if !(previous_banned_bounces.contains(&[current_x ,current_y])) {
                                previous_banned_bounces.push([current_x ,current_y]);

                                for item in &current_path {
                                    previous_path.push(*item)
    
                                }
    
                                for drop in current_player_drops.iter() {
                                    final_moves.push([self.data[current_y as usize][current_x as usize], drop[0], drop[1], starting_piece_type, current_x as usize ,current_y as usize, 0, starting_piece[0], starting_piece[1]]);
    
                                }
                                
                                let mut moves: Vec<[usize; 9]> = self.get_piece_moves(&[current_x as usize ,current_y as usize], self.data[current_y as usize][current_x as usize], starting_piece, starting_piece_type, previous_path, previous_banned_bounces, player, current_player_drops);
                                final_moves.append(&mut moves);
                                
                                previous_banned_bounces.pop();

                                for _ in &current_path {
                                    previous_path.pop();
    
                                }

                            }
    
                        } else {
                            if &[current_x as usize,  current_y as usize] != starting_piece {
                                final_moves.push([starting_piece_type, current_x as usize, current_y as usize, 0, starting_piece[0], starting_piece[1], 9, 9, 9]);

                            }

                        }
    
                    } 
    
                }
                
            }
    
        } else if current_piece_type == TWO_PIECE {
            for path in self.two_moves.iter() {
                let mut current_x: i8 = current_piece[0] as i8;
                let mut current_y: i8 = current_piece[1] as i8;
        
                let mut current_path: Vec<[i8; 2]> = vec![];
        
                'step: for step_idx in 0..path.len() {
                    let step = path[step_idx];
    
                    for (item_idx, item) in previous_path.iter().enumerate() {
                        if item == &[current_x, current_y] {
                            let mut temp_curr_x = current_x;
                            let mut temp_curr_y = current_y;
    
                            match step as i8{
                                0=> temp_curr_x -= 1,
                                1=> temp_curr_y -= 1,
                                2=> temp_curr_x  += 1,
                                3=> temp_curr_y  += 1,
                                _=> println!("ERROR"),
                              }
                        
                            if item_idx != 0 && previous_path[item_idx - 1] == [temp_curr_x, temp_curr_y] {
                                break 'step;
    
                            }
    
                            if item_idx != previous_path.len() - 1 && previous_path[item_idx + 1] == [temp_curr_x, temp_curr_y] {
                                break 'step;
    
                            }
    
                        }
    
                    }

                    match step as i8{
                        0=> current_x -= 1,
                        1=> current_y -= 1,
                        2=> current_x  += 1,
                        3=> current_y  += 1,
                        _=> println!("ERROR"),
                    }
    
                    if  current_x < 0 || current_x > 5 {
                        break;
                    } 
        
                    if step_idx == path.len() - 1 {
                        if current_y == 6 && player == 1 {
                            final_moves.push([starting_piece_type, PLAYER_2_GOAL, PLAYER_2_GOAL, 0, starting_piece[0], starting_piece[1], 9, 9, 9]);
        
                        } else if current_y == -1 && player == 2{
                            final_moves.push([starting_piece_type, PLAYER_1_GOAL, PLAYER_1_GOAL, 0, starting_piece[0], starting_piece[1], 9, 9, 9]);
                            
                        }
        
                    }
    
                    if current_y < 0 || current_y > 5 {
                        break 'step;
                    } 
    
                    if step_idx != path.len() - 1 {
                        if self.data[current_y as usize][current_x as usize] != 0 {
                            break 'step;
    
                        } 
                    }
    
                    current_path.push([current_x, current_y]);
    
                    if step_idx == path.len() - 1 {
                        if self.data[current_y as usize][current_x as usize] != 0 {
                            if !(previous_banned_bounces.contains(&[current_x ,current_y])) {
                                previous_banned_bounces.push([current_x ,current_y]);

                                for item in &current_path {
                                    previous_path.push(*item)
    
                                }
    
                                for drop in current_player_drops.iter() {
                                    final_moves.push([self.data[current_y as usize][current_x as usize], drop[0], drop[1], starting_piece_type, current_x as usize ,current_y as usize, 0, starting_piece[0], starting_piece[1]]);
    
                                }
                                
                                let mut moves: Vec<[usize; 9]> = self.get_piece_moves(&[current_x as usize ,current_y as usize], self.data[current_y as usize][current_x as usize], starting_piece, starting_piece_type, previous_path, previous_banned_bounces, player, current_player_drops);
                                final_moves.append(&mut moves);
                                
                                previous_banned_bounces.pop();

                                for _ in &current_path {
                                    previous_path.pop();
    
                                }

                            }
    
                        } else {
                            if &[current_x as usize,  current_y as usize] != starting_piece {
                                final_moves.push([starting_piece_type, current_x as usize, current_y as usize, 0, starting_piece[0], starting_piece[1], 9, 9, 9]);

                            }

                        }
    
                    } 
    
                }
                
            }
    
        } else if current_piece_type == THREE_PIECE {
            for path in self.three_moves.iter() {
                let mut current_x: i8 = current_piece[0] as i8;
                let mut current_y: i8 = current_piece[1] as i8;
        
                let mut current_path: Vec<[i8; 2]> = vec![];
        
                'step: for step_idx in 0..path.len() {
                    let step = path[step_idx];
    
                    for (item_idx, item) in previous_path.iter().enumerate() {
                        if item == &[current_x, current_y] {
                            let mut temp_curr_x = current_x;
                            let mut temp_curr_y = current_y;
    
                            match step as i8{
                                0=> temp_curr_x -= 1,
                                1=> temp_curr_y -= 1,
                                2=> temp_curr_x  += 1,
                                3=> temp_curr_y  += 1,
                                _=> println!("ERROR"),
                              }
                        
                            if item_idx != 0 && previous_path[item_idx - 1] == [temp_curr_x, temp_curr_y] {
                                break 'step;
    
                            }
    
                            if item_idx != previous_path.len() - 1 && previous_path[item_idx + 1] == [temp_curr_x, temp_curr_y] {
                                break 'step;
    
                            }
    
                        }
    
                    }

                    match step as i8{
                        0=> current_x -= 1,
                        1=> current_y -= 1,
                        2=> current_x  += 1,
                        3=> current_y  += 1,
                        _=> println!("ERROR"),
                    }
    
                    if  current_x < 0 || current_x > 5 {
                        break;
                    } 
        
                    if step_idx == path.len() - 1 {
                        if current_y == 6 && player == 1 {
                            final_moves.push([starting_piece_type, PLAYER_2_GOAL, PLAYER_2_GOAL, 0, starting_piece[0], starting_piece[1], 9, 9, 9]);
        
                        } else if current_y == -1 && player == 2{
                            final_moves.push([starting_piece_type, PLAYER_1_GOAL, PLAYER_1_GOAL, 0, starting_piece[0], starting_piece[1], 9, 9, 9]);
                            
                        }
        
                    }
    
                    if current_y < 0 || current_y > 5 {
                        break 'step;
                    } 
    
                    if step_idx != path.len() - 1 {
                        if self.data[current_y as usize][current_x as usize] != 0 {
                            break 'step;
    
                        } 
                    }
    
                    current_path.push([current_x, current_y]);
    
                    if step_idx == path.len() - 1 {
                        if self.data[current_y as usize][current_x as usize] != 0 {
                            if !(previous_banned_bounces.contains(&[current_x ,current_y])) {
                                previous_banned_bounces.push([current_x ,current_y]);

                                for item in &current_path {
                                    previous_path.push(*item)
    
                                }
    
                                for drop in current_player_drops.iter() {
                                    final_moves.push([self.data[current_y as usize][current_x as usize], drop[0], drop[1], starting_piece_type, current_x as usize ,current_y as usize, 0, starting_piece[0], starting_piece[1]]);
    
                                }
                                
                                let mut moves: Vec<[usize; 9]> = self.get_piece_moves(&[current_x as usize ,current_y as usize], self.data[current_y as usize][current_x as usize], starting_piece, starting_piece_type, previous_path, previous_banned_bounces, player, current_player_drops);
                                final_moves.append(&mut moves);
                                
                                previous_banned_bounces.pop();

                                for _ in &current_path {
                                    previous_path.pop();
    
                                }

                            }
    
                        } else {
                            if &[current_x as usize,  current_y as usize] != starting_piece {
                                final_moves.push([starting_piece_type, current_x as usize, current_y as usize, 0, starting_piece[0], starting_piece[1], 9, 9, 9]);

                            }

                        }
    
                    } 
    
                }
                
            }
    
        } 

        return final_moves;
    
    }
    
    // ------------------------- Move Count -------------------------

    fn valid_move_count(&mut self, player: i8) -> usize {
        let active_lines = self.active_lines();
    
        if player == 1 {
            let mut player_1_drop_count: i8 = 1;
            for y in 0..active_lines[1] + 1 {
                for x in 0..6 {
                    if self.data[y][x] == 0 {
                        player_1_drop_count += 1;
    
                    }
    
                }
    
            }
            
            
            let mut player_1_move_count: usize = 0;
            
            for x in 0..6 {
                if self.data[active_lines[0]][x] != 0 {
                    let starting_piece: [usize; 2] = [x, active_lines[0]];
                    let starting_piece_type: usize = self.data[starting_piece[1]][starting_piece[0]];

                    let mut previous_path: Vec<[i8; 2]> = vec![[x as i8, active_lines[0] as i8]];
                    let mut previous_banned_bounces: Vec<[i8; 2]> = vec![];
                                    
                    self.data[starting_piece[1]][starting_piece[0]] = 0;

                    let move_count = self.get_piece_move_count(&starting_piece, starting_piece_type, &starting_piece, starting_piece_type, &mut previous_path, &mut previous_banned_bounces, 1, player_1_drop_count);
                    player_1_move_count += move_count;

                    self.data[starting_piece[1]][starting_piece[0]] = starting_piece_type;

                }
    
            }
    
            return player_1_move_count;
    
        } else {
            let mut player_2_drop_count: i8 = 1;
            for y in (active_lines[0]..6).rev() {
                for x in 0..6 {
                    if self.data[y][x] == 0 {
                        player_2_drop_count += 1;
    
                    }
    
                }
                
            }
    
            let mut player_2_move_count: usize = 0;
    
            for x in 0..6 {
                if self.data[active_lines[1]][x] != 0 {
                    let starting_piece: [usize; 2] = [x, active_lines[1]];
                    let starting_piece_type: usize = self.data[starting_piece[1]][starting_piece[0]];

                    let mut previous_path: Vec<[i8; 2]> = vec![[x as i8, active_lines[1] as i8]];
                    let mut previous_banned_bounces: Vec<[i8; 2]> = vec![];
                    
                    self.data[starting_piece[1]][starting_piece[0]] = 0;

                    let move_count = self.get_piece_move_count(&starting_piece, starting_piece_type, &starting_piece, starting_piece_type, &mut previous_path, &mut previous_banned_bounces, 2, player_2_drop_count);
                    player_2_move_count += move_count;

                    self.data[starting_piece[1]][starting_piece[0]] = starting_piece_type;

                }
    
            }
    
            return player_2_move_count;
    
        }
    
    }

    fn get_piece_move_count(&self, current_piece: &[usize; 2], current_piece_type: usize, starting_piece: &[usize; 2], starting_piece_type: usize, previous_path: &mut Vec<[i8; 2]>, previous_banned_bounces: &mut Vec<[i8; 2]>, player: i8, current_player_drop_count: i8) -> usize {
        let mut final_move_count: usize = 0;
        
        if current_piece_type == ONE_PIECE {
            for path in self.one_moves.iter() {
                let mut current_x: i8 = current_piece[0] as i8;
                let mut current_y: i8 = current_piece[1] as i8;
        
                let mut current_path: Vec<[i8; 2]> = vec![];
        
                'step: for step_idx in 0..path.len() {
                    let step = path[step_idx];
    
                    for (item_idx, item) in previous_path.iter().enumerate() {
                        if item == &[current_x, current_y] {
                            let mut temp_curr_x = current_x;
                            let mut temp_curr_y = current_y;
    
                            match step as i8{
                                0=> temp_curr_x -= 1,
                                1=> temp_curr_y -= 1,
                                2=> temp_curr_x  += 1,
                                3=> temp_curr_y  += 1,
                                _=> println!("ERROR"),
                              }
                        
                            if item_idx != 0 && previous_path[item_idx - 1] == [temp_curr_x, temp_curr_y] {
                                break 'step;
    
                            }
    
                            if item_idx != previous_path.len() - 1 && previous_path[item_idx + 1] == [temp_curr_x, temp_curr_y] {
                                break 'step;
    
                            }
    
                        }
    
                    }

                    match step as i8{
                        0=> current_x -= 1,
                        1=> current_y -= 1,
                        2=> current_x  += 1,
                        3=> current_y  += 1,
                        _=> println!("ERROR"),
                    }
    
                    if current_x < 0 || current_x > 5 {
                        break;
                    } 
        
                    if step_idx == path.len() - 1 {
                        if current_y == 6 && player == 1 {
                            final_move_count += 1;
        
                        } else if current_y == -1 && player == 2{
                            final_move_count += 1;
                            
                        }
        
                    }
    
                    if current_y < 0 || current_y > 5 {
                        break 'step;
                    } 
    
                    if step_idx != path.len() - 1 {
                        if self.data[current_y as usize][current_x as usize] != 0 {
                            break 'step;
    
                        } 
                    }
    
                    current_path.push([current_x, current_y]);
    
                    if step_idx == path.len() - 1 {
                        if self.data[current_y as usize][current_x as usize] != 0 {
                            if !(previous_banned_bounces.contains(&[current_x ,current_y])) {
                                previous_banned_bounces.push([current_x ,current_y]);

                                for item in &current_path {
                                    previous_path.push(*item)
    
                                }
    
                                for _ in 0..current_player_drop_count {
                                    final_move_count += 1;
                                }
                                
                                let move_count: usize = self.get_piece_move_count(&[current_x as usize ,current_y as usize], self.data[current_y as usize][current_x as usize], starting_piece, starting_piece_type, previous_path, previous_banned_bounces, player, current_player_drop_count);
                                final_move_count += move_count;
                                
                                previous_banned_bounces.pop();

                                for _ in &current_path {
                                    previous_path.pop();
    
                                }

                            }
    
                        } else {
                            if &[current_x as usize,  current_y as usize] != starting_piece {
                                final_move_count += 1;

                            }

                        }
    
                    } 
    
                }
                
            }
    
        } else if current_piece_type == TWO_PIECE {
            for path in self.two_moves.iter() {
                let mut current_x: i8 = current_piece[0] as i8;
                let mut current_y: i8 = current_piece[1] as i8;
        
                let mut current_path: Vec<[i8; 2]> = vec![];
        
                'step: for step_idx in 0..path.len() {
                    let step = path[step_idx];
    
                    for (item_idx, item) in previous_path.iter().enumerate() {
                        if item == &[current_x, current_y] {
                            let mut temp_curr_x = current_x;
                            let mut temp_curr_y = current_y;
    
                            match step as i8{
                                0=> temp_curr_x -= 1,
                                1=> temp_curr_y -= 1,
                                2=> temp_curr_x  += 1,
                                3=> temp_curr_y  += 1,
                                _=> println!("ERROR"),
                              }
                        
                            if item_idx != 0 && previous_path[item_idx - 1] == [temp_curr_x, temp_curr_y] {
                                break 'step;
    
                            }
    
                            if item_idx != previous_path.len() - 1 && previous_path[item_idx + 1] == [temp_curr_x, temp_curr_y] {
                                break 'step;
    
                            }
    
                        }
    
                    }

                    match step as i8{
                        0=> current_x -= 1,
                        1=> current_y -= 1,
                        2=> current_x  += 1,
                        3=> current_y  += 1,
                        _=> println!("ERROR"),
                    }
    
                    if  current_x < 0 || current_x > 5 {
                        break;
                    } 
        
                    if step_idx == path.len() - 1 {
                        if current_y == 6 && player == 1 {
                            final_move_count += 1;
        
                        } else if current_y == -1 && player == 2{
                            final_move_count += 1;
                            
                        }
        
                    }
    
                    if current_y < 0 || current_y > 5 {
                        break 'step;
                    } 
    
                    if step_idx != path.len() - 1 {
                        if self.data[current_y as usize][current_x as usize] != 0 {
                            break 'step;
    
                        } 
                    }
    
                    current_path.push([current_x, current_y]);
    
                    if step_idx == path.len() - 1 {
                        if self.data[current_y as usize][current_x as usize] != 0 {
                            if !(previous_banned_bounces.contains(&[current_x ,current_y])) {
                                previous_banned_bounces.push([current_x ,current_y]);

                                for item in &current_path {
                                    previous_path.push(*item)
    
                                }
    
                                for _ in 0..current_player_drop_count {
                                    final_move_count += 1;
                                }
                                
                                let move_count: usize = self.get_piece_move_count(&[current_x as usize ,current_y as usize], self.data[current_y as usize][current_x as usize], starting_piece, starting_piece_type, previous_path, previous_banned_bounces, player, current_player_drop_count);
                                final_move_count += move_count;
                                
                                previous_banned_bounces.pop();

                                for _ in &current_path {
                                    previous_path.pop();
    
                                }

                            }
    
                        } else {
                            if &[current_x as usize,  current_y as usize] != starting_piece {
                                final_move_count += 1;

                            }

                        }
    
                    } 
    
                }
                
            }
    
        } else if current_piece_type == THREE_PIECE {
            for path in self.three_moves.iter() {
                let mut current_x: i8 = current_piece[0] as i8;
                let mut current_y: i8 = current_piece[1] as i8;
        
                let mut current_path: Vec<[i8; 2]> = vec![];
        
                'step: for step_idx in 0..path.len() {
                    let step = path[step_idx];
    
                    for (item_idx, item) in previous_path.iter().enumerate() {
                        if item == &[current_x, current_y] {
                            let mut temp_curr_x = current_x;
                            let mut temp_curr_y = current_y;
    
                            match step as i8{
                                0=> temp_curr_x -= 1,
                                1=> temp_curr_y -= 1,
                                2=> temp_curr_x  += 1,
                                3=> temp_curr_y  += 1,
                                _=> println!("ERROR"),
                              }
                        
                            if item_idx != 0 && previous_path[item_idx - 1] == [temp_curr_x, temp_curr_y] {
                                break 'step;
    
                            }
    
                            if item_idx != previous_path.len() - 1 && previous_path[item_idx + 1] == [temp_curr_x, temp_curr_y] {
                                break 'step;
    
                            }
    
                        }
    
                    }

                    match step as i8{
                        0=> current_x -= 1,
                        1=> current_y -= 1,
                        2=> current_x  += 1,
                        3=> current_y  += 1,
                        _=> println!("ERROR"),
                    }
    
                    if  current_x < 0 || current_x > 5 {
                        break;
                    } 
        
                    if step_idx == path.len() - 1 {
                        if current_y == 6 && player == 1 {
                            final_move_count += 1;
        
                        } else if current_y == -1 && player == 2{
                            final_move_count += 1;
                            
                        }
        
                    }
    
                    if current_y < 0 || current_y > 5 {
                        break 'step;
                    } 
    
                    if step_idx != path.len() - 1 {
                        if self.data[current_y as usize][current_x as usize] != 0 {
                            break 'step;
    
                        } 
                    }
    
                    current_path.push([current_x, current_y]);
    
                    if step_idx == path.len() - 1 {
                        if self.data[current_y as usize][current_x as usize] != 0 {
                            if !(previous_banned_bounces.contains(&[current_x ,current_y])) {
                                previous_banned_bounces.push([current_x ,current_y]);

                                for item in &current_path {
                                    previous_path.push(*item)
    
                                }
    
                                for _ in 0..current_player_drop_count {
                                    final_move_count += 1;
                                }
                                
                                let move_count: usize = self.get_piece_move_count(&[current_x as usize ,current_y as usize], self.data[current_y as usize][current_x as usize], starting_piece, starting_piece_type, previous_path, previous_banned_bounces, player, current_player_drop_count);
                                final_move_count += move_count;
                                
                                previous_banned_bounces.pop();

                                for _ in &current_path {
                                    previous_path.pop();
    
                                }

                            }
    
                        } else {
                            if &[current_x as usize,  current_y as usize] != starting_piece {
                                final_move_count += 1;

                            }

                        }
    
                    } 
    
                }
                
            }
    
        }
    
        return final_move_count;
    
    }

    // ------------------------- Threat Count -------------------------

    fn valid_threat_count(&mut self, player: i8) -> usize {
        let active_lines = self.active_lines();
    
        if player == 1 {
            let mut player_1_threat_count: usize = 0;
            
            for x in 0..6 {
                if self.data[active_lines[0]][x] != 0 {
                    let starting_piece: [usize; 2] = [x, active_lines[0]];
                    let starting_piece_type: usize = self.data[starting_piece[1]][starting_piece[0]];

                    let mut previous_path: Vec<[i8; 2]> = vec![[x as i8, active_lines[0] as i8]];
                    let mut previous_banned_bounces: Vec<[i8; 2]> = vec![];
                    
                    self.data[starting_piece[1]][starting_piece[0]] = 0;

                    let move_count = self.get_piece_threat_count(&starting_piece, starting_piece_type, &starting_piece, starting_piece_type, &mut previous_path, &mut previous_banned_bounces, 1);
                    player_1_threat_count += move_count;

                    self.data[starting_piece[1]][starting_piece[0]] = starting_piece_type;

                }
    
            }
    
            return player_1_threat_count;
    
        } else {
            let mut player_2_threat_count: usize = 0;
    
            for x in 0..6 {
                if self.data[active_lines[1]][x] != 0 {
                    let starting_piece: [usize; 2] = [x, active_lines[1]];
                    let starting_piece_type: usize = self.data[starting_piece[1]][starting_piece[0]];

                    let mut previous_path: Vec<[i8; 2]> = vec![[x as i8, active_lines[1] as i8]];
                    let mut previous_banned_bounces: Vec<[i8; 2]> = vec![];
                    
                    self.data[starting_piece[1]][starting_piece[0]] = 0;

                    let move_count = self.get_piece_threat_count(&starting_piece, starting_piece_type, &starting_piece, starting_piece_type, &mut previous_path, &mut previous_banned_bounces, 2);
                    player_2_threat_count += move_count;

                    self.data[starting_piece[1]][starting_piece[0]] = starting_piece_type;

                }
    
            }
    
            return player_2_threat_count;
    
        }
    
    }

    fn get_piece_threat_count(&self, current_piece: &[usize; 2], current_piece_type: usize, starting_piece: &[usize; 2], starting_piece_type: usize, previous_path: &mut Vec<[i8; 2]>, previous_banned_bounces: &mut Vec<[i8; 2]>, player: i8) -> usize {
        let mut final_threat_count: usize = 0;
        
        if current_piece_type == ONE_PIECE {
            for path in self.one_moves.iter() {
                let mut current_x: i8 = current_piece[0] as i8;
                let mut current_y: i8 = current_piece[1] as i8;
        
                let mut current_path: Vec<[i8; 2]> = vec![];
        
                'step: for step_idx in 0..path.len() {
                    let step = path[step_idx];
    
                    for (item_idx, item) in previous_path.iter().enumerate() {
                        if item == &[current_x, current_y] {
                            let mut temp_curr_x = current_x;
                            let mut temp_curr_y = current_y;
    
                            match step as i8{
                                0=> temp_curr_x -= 1,
                                1=> temp_curr_y -= 1,
                                2=> temp_curr_x  += 1,
                                3=> temp_curr_y  += 1,
                                _=> println!("ERROR"),
                              }
                        
                            if item_idx != 0 && previous_path[item_idx - 1] == [temp_curr_x, temp_curr_y] {
                                break 'step;
    
                            }
    
                            if item_idx != previous_path.len() - 1 && previous_path[item_idx + 1] == [temp_curr_x, temp_curr_y] {
                                break 'step;
    
                            }
    
                        }
    
                    }

                    match step as i8{
                        0=> current_x -= 1,
                        1=> current_y -= 1,
                        2=> current_x  += 1,
                        3=> current_y  += 1,
                        _=> println!("ERROR"),
                    }
    
                    if  current_x < 0 || current_x > 5 {
                        break;
                    } 
        
                    if step_idx == path.len() - 1 {
                        if current_y == 6 && player == 1 {
                            final_threat_count += 1;
        
                        } else if current_y == -1 && player == 2{
                            final_threat_count += 1;
                            
                        }
        
                    }
    
                    if current_y < 0 || current_y > 5 {
                        break 'step;
                    } 
    
                    if step_idx != path.len() - 1 {
                        if self.data[current_y as usize][current_x as usize] != 0 {
                            break 'step;
    
                        } 
                    }
    
                    current_path.push([current_x, current_y]);
    
                    if step_idx == path.len() - 1 {
                        if self.data[current_y as usize][current_x as usize] != 0 {
                            if !(previous_banned_bounces.contains(&[current_x ,current_y])) {
                                previous_banned_bounces.push([current_x ,current_y]);
                                
                                for item in &current_path {
                                    previous_path.push(*item)
    
                                }

                                let threat_count: usize = self.get_piece_threat_count(&[current_x as usize ,current_y as usize], self.data[current_y as usize][current_x as usize], starting_piece, starting_piece_type, previous_path, previous_banned_bounces, player);
                                final_threat_count += threat_count;
                                
                                previous_banned_bounces.pop();

                                for _ in &current_path {
                                    previous_path.pop();
    
                                }

                            }
    
                        }
    
                    } 
    
                }
                
            }
    
        } else if current_piece_type == TWO_PIECE {
            for path in self.two_moves.iter() {
                let mut current_x: i8 = current_piece[0] as i8;
                let mut current_y: i8 = current_piece[1] as i8;
        
                let mut current_path: Vec<[i8; 2]> = vec![];
        
                'step: for step_idx in 0..path.len() {
                    let step = path[step_idx];
    
                    for (item_idx, item) in previous_path.iter().enumerate() {
                        if item == &[current_x, current_y] {
                            let mut temp_curr_x = current_x;
                            let mut temp_curr_y = current_y;
    
                            match step as i8{
                                0=> temp_curr_x -= 1,
                                1=> temp_curr_y -= 1,
                                2=> temp_curr_x  += 1,
                                3=> temp_curr_y  += 1,
                                _=> println!("ERROR"),
                              }
                        
                            if item_idx != 0 && previous_path[item_idx - 1] == [temp_curr_x, temp_curr_y] {
                                break 'step;
    
                            }
    
                            if item_idx != previous_path.len() - 1 && previous_path[item_idx + 1] == [temp_curr_x, temp_curr_y] {
                                break 'step;
    
                            }
    
                        }
    
                    }

                    match step as i8{
                        0=> current_x -= 1,
                        1=> current_y -= 1,
                        2=> current_x  += 1,
                        3=> current_y  += 1,
                        _=> println!("ERROR"),
                    }
    
                    if  current_x < 0 || current_x > 5 {
                        break;
                    } 
        
                    if step_idx == path.len() - 1 {
                        if current_y == 6 && player == 1 {
                            final_threat_count += 1;
        
                        } else if current_y == -1 && player == 2{
                            final_threat_count += 1;
                            
                        }
        
                    }
    
                    if current_y < 0 || current_y > 5 {
                        break 'step;
                    } 
    
                    if step_idx != path.len() - 1 {
                        if self.data[current_y as usize][current_x as usize] != 0 {
                            break 'step;
    
                        } 
                    }
    
                    current_path.push([current_x, current_y]);
    
                    if step_idx == path.len() - 1 {
                        if self.data[current_y as usize][current_x as usize] != 0 {
                            if !(previous_banned_bounces.contains(&[current_x ,current_y])) {
                                previous_banned_bounces.push([current_x ,current_y]);
                                
                                for item in &current_path {
                                    previous_path.push(*item)
    
                                }

                                let threat_count: usize = self.get_piece_threat_count(&[current_x as usize ,current_y as usize], self.data[current_y as usize][current_x as usize], starting_piece, starting_piece_type, previous_path, previous_banned_bounces, player);
                                final_threat_count += threat_count;
                                
                                previous_banned_bounces.pop();

                                for _ in &current_path {
                                    previous_path.pop();
    
                                }

                            }
    
                        }
    
                    } 
    
                }
                
            }
    
        } else if current_piece_type == THREE_PIECE {
            for path in self.three_moves.iter() {
                let mut current_x: i8 = current_piece[0] as i8;
                let mut current_y: i8 = current_piece[1] as i8;
        
                let mut current_path: Vec<[i8; 2]> = vec![];
        
                'step: for step_idx in 0..path.len() {
                    let step = path[step_idx];
    
                    for (item_idx, item) in previous_path.iter().enumerate() {
                        if item == &[current_x, current_y] {
                            let mut temp_curr_x = current_x;
                            let mut temp_curr_y = current_y;
    
                            match step as i8{
                                0=> temp_curr_x -= 1,
                                1=> temp_curr_y -= 1,
                                2=> temp_curr_x  += 1,
                                3=> temp_curr_y  += 1,
                                _=> println!("ERROR"),
                              }
                        
                            if item_idx != 0 && previous_path[item_idx - 1] == [temp_curr_x, temp_curr_y] {
                                break 'step;
    
                            }
    
                            if item_idx != previous_path.len() - 1 && previous_path[item_idx + 1] == [temp_curr_x, temp_curr_y] {
                                break 'step;
    
                            }
    
                        }
    
                    }

                    match step as i8{
                        0=> current_x -= 1,
                        1=> current_y -= 1,
                        2=> current_x  += 1,
                        3=> current_y  += 1,
                        _=> println!("ERROR"),
                    }
    
                    if  current_x < 0 || current_x > 5 {
                        break;
                    } 
        
                    if step_idx == path.len() - 1 {
                        if current_y == 6 && player == 1 {
                            final_threat_count += 1;
        
                        } else if current_y == -1 && player == 2{
                            final_threat_count += 1;
                            
                        }
        
                    }
    
                    if current_y < 0 || current_y > 5 {
                        break 'step;
                    } 
    
                    if step_idx != path.len() - 1 {
                        if self.data[current_y as usize][current_x as usize] != 0 {
                            break 'step;
    
                        } 
                    }
    
                    current_path.push([current_x, current_y]);
    
                    if step_idx == path.len() - 1 {
                        if self.data[current_y as usize][current_x as usize] != 0 {
                            if !(previous_banned_bounces.contains(&[current_x ,current_y])) {
                                previous_banned_bounces.push([current_x ,current_y]);
                                
                                for item in &current_path {
                                    previous_path.push(*item)
    
                                }

                                let threat_count: usize = self.get_piece_threat_count(&[current_x as usize ,current_y as usize], self.data[current_y as usize][current_x as usize], starting_piece, starting_piece_type, previous_path, previous_banned_bounces, player);
                                final_threat_count += threat_count;
                                
                                previous_banned_bounces.pop();

                                for _ in &current_path {
                                    previous_path.pop();
    
                                }

                            }
    
                        }
    
                    } 
    
                }
                
            }
    
        } 

        return final_threat_count;
    
    }

    // --------------------------------------------------
    

}


fn main() {
    let mut board = Board::new(); 

    board.set([   [0 ,3, 0 ,0, 2, 0],
                        [3 ,0 ,0 ,1, 1, 3],
                        [0 ,0 ,0 ,2, 0, 0],
                        [0 ,0 ,0 ,1, 0, 0],
                        [3 ,0, 2, 0, 0, 0], 
                        [1 ,2 ,0 ,0, 0, 0]], 
                    [0 ,0]);
    
    board._print();

    let data = board.get_best_move(3);

    println!("");
    println!("SCORE: {}", data.0);
    println!("MOVE: {:?}", data.1);
    println!("DONE! in {} seconds.", data.2);
    println!("");

}