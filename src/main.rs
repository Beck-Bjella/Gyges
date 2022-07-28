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
    fn print_board(&self) { 
        println!(" ");
        println!("         {}", self.goals[0]);
        println!(" ");
        for y in 0..6 {
            for x in 0..6 {
                print!("  {}", self.data[y][x]);
            }
            println!(" ");
        }
        println!(" ");
        println!("         {}",  self.goals[1]);
        println!(" ");
    
    }

    fn game_over(&self, current_player: i8) -> bool{
        if self.threat_count(current_player) > 0 {
            return true;

        }

        return false;
    
    }

    fn evalulate(&self) -> usize {
        let starting_value = usize::MAX / 2;
        let player_1_moves = self.valid_moves(1);
        let player_2_moves = self.valid_moves(2);

        let score = starting_value + (player_2_moves.len() - (player_1_moves.len() * 2));

        return score;
    
    }

    fn threat_count(&self, player: i8) -> usize {
        let mut threats = 0;
        let moves = self.valid_moves(player);
        for mv in moves {
            if mv[0][1] == PLAYER_1_GOAL || mv[0][1] == PLAYER_2_GOAL {
                threats += 1;

            }

        }

        return threats;

    }

    fn make_move(&mut self, mv: &Vec<[usize; 3]>) {
        for step in mv.iter() {
            if step[1] == PLAYER_1_GOAL {
                self.goals[0] = 1;
    
            } else if step[1] == PLAYER_2_GOAL {
                self.goals[1] = 1;
    
            } else {
                self.data[step[2]][step[1]] = step[0];
    
            }
    
        }
    
    } 

    fn undo_move(&mut self, mv: &Vec<[usize; 3]>) {
        for (step_idx, step) in mv.iter().enumerate() {
            if step_idx != mv.len() - 1 {
                let temp_step = mv[step_idx + 1];
                self.data[temp_step[2]][temp_step[1]] = step[0];

            } else {
                let temp_step = mv[0];

                if temp_step[1] == PLAYER_1_GOAL {
                    self.goals[0] = 0;

                } else if temp_step[1] == PLAYER_2_GOAL {
                    self.goals[1] = 0;

                } else {
                    self.data[temp_step[2]][temp_step[1]] = step[0];

                }

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
            if self.game_over(2){
                return usize::MAX;
        
            }

            let current_moves = self.valid_moves(2);
    
            let mut max_eval: usize = usize::MIN;
            let mut used_moves: Vec<Vec<[usize; 3]>> = Vec::new();
            for mv in current_moves.iter() {
                if used_moves.contains(mv) {
                    continue;
                }
                
                self.make_move(&mv);
    
                used_moves.push(mv.to_vec());
    
                let curr_eval: usize;
                curr_eval = self.mini_max(alpha, beta, false, depth - 1);
                
                self.undo_move(&mv);

                max_eval = max!(curr_eval, max_eval);
    
                alpha = max!(alpha, curr_eval);
                if alpha >= beta {
                    break
    
                }
                    
            }
            
            return max_eval;
    
        } else {
            if self.game_over(1){
                return usize::MIN;
        
            }

            let current_moves = self.valid_moves(1);
    
            let mut min_eval: usize = usize::MAX;
            let mut used_moves: Vec<Vec<[usize; 3]>> = Vec::new();
            for mv in current_moves.iter() {
                if used_moves.contains(mv) {
                    continue;
                }
    
                used_moves.push(mv.to_vec());
    
                self.make_move(&mv);
                
                let curr_eval: usize;
                curr_eval = self.mini_max(alpha, beta, true, depth - 1);
                
                self.undo_move(&mv);

                min_eval = min!(curr_eval, min_eval);
    
                beta = min!(beta, curr_eval);
                if beta <= alpha {
                    break;
    
                }
                    
            }
    
            return min_eval;
    
        }
    
    }

    fn get_best_move(&mut self, depth: i8) {
        let mut alpha = usize::MIN;
        let beta = usize::MAX;
        let mut max_eval: [usize; 2] = [usize::MIN, usize::MIN];
        
        let mut current_moves = self.valid_moves(2);
        current_moves = order_moves(current_moves);

        let mut used_moves: Vec<Vec<[usize; 3]>> = Vec::new();
        for (move_idx, mv) in current_moves.iter().enumerate() {
            if used_moves.contains(mv) {
                println!("Indix {} is a dupe", move_idx);
                continue;
            }
            
            used_moves.push(mv.to_vec());
    
            self.make_move(&mv);
            
            println!("Starting Index {}", move_idx);
    
            let eval = self.mini_max(alpha, beta, false, depth);

            // println!("{:?} - {} - {}", mv, eval, self._threat_count(1));

            // let threat_eval = self._threat_count(2) * 1000;
            // eval += threat_eval;
            
            self.undo_move(&mv);

            if eval >= max_eval[0] {
                max_eval = [eval, move_idx];
            }

            alpha = max!(alpha, eval);
            if alpha > beta {
                break
    
            }
        
        }
    
        println!("{:?} {:?}", current_moves[max_eval[1]], max_eval);
        
    }

    fn valid_moves(&self, player: i8) -> Vec<Vec<[usize; 3]>> {
        let active_lines = self.active_lines();
    
        if player == 1 {
            let mut player_1_drops: Vec<[usize; 2]> = Vec::new();
            for y in active_lines[0]..active_lines[1] + 1 {
                for x in 0..6 {
                    if self.data[y][x] == 0 {
                        player_1_drops.push([x, y]);
    
                    }
    
                }
    
            }
            
            
            let mut player_1_moves: Vec<Vec<[usize; 3]>> = Vec::new();
            
            for x in 0..6 {
                if self.data[active_lines[0]][x] != 0 {
                    let current_piece: [usize; 2] = [x, active_lines[0]];
                    let mut previous_path: Vec<[i8; 2]> = vec![[x as i8, active_lines[0] as i8]];
                    let mut previous_banned_bounces: Vec<[i8; 2]> = vec![[x as i8, active_lines[0] as i8]];
    
                    let mut moves =  self.get_piece_moves(&current_piece, &current_piece, &mut previous_path, &mut previous_banned_bounces, 1, &player_1_drops);
                    player_1_moves.append(&mut moves);
    
                }
    
            }
    
            return player_1_moves;
    
        } else if player == 2 {
            let mut player_2_drops: Vec<[usize; 2]> = Vec::new();
            for y in (active_lines[0]..active_lines[1] + 1).rev() {
                for x in 0..6 {
                    if self.data[y][x] == 0 {
                        player_2_drops.push([x, y]);
    
                    }
    
                }
                
            }
    
            let mut player_2_moves: Vec<Vec<[usize; 3]>> = Vec::new();
    
            for x in 0..6 {
                if self.data[active_lines[1]][x] != 0 {
                    let current_piece: [usize; 2] = [x, active_lines[1]];
                    let mut previous_path: Vec<[i8; 2]> = vec![[x as i8, active_lines[1] as i8]];
                    let mut previous_banned_bounces: Vec<[i8; 2]> = vec![[x as i8, active_lines[1] as i8]];
    
                    let mut moves = self.get_piece_moves(&current_piece, &current_piece, &mut previous_path, &mut previous_banned_bounces, 2, &player_2_drops);
                    player_2_moves.append(&mut moves);
                    
    
                }
    
            }
    
            return player_2_moves;
    
        }
    
        return Vec::new();
    
    }

    fn get_piece_moves(&self, current_piece: &[usize; 2],  starting_piece: &[usize; 2], previous_path: &mut Vec<[i8; 2]>, previous_banned_bounces: &mut Vec<[i8; 2]>, player: i8, current_player_drops: &Vec<[usize; 2]>) -> Vec<Vec<[usize; 3]>> {
        let mut final_moves: Vec<Vec<[usize; 3]>> = Vec::new();
    
        let piece: usize = self.data[current_piece[1]][current_piece[0]];
    
        if piece == ONE_PIECE {
            for path in self.one_moves.iter() {
                let mut current_x: i8 = current_piece[0] as i8;
                let mut current_y: i8 = current_piece[1] as i8;
        
                let mut current_path: Vec<[i8; 2]> = Vec::new();
        
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
                            final_moves.push(vec![[self.data[starting_piece[1]][starting_piece[0]], PLAYER_2_GOAL, PLAYER_2_GOAL], [0, starting_piece[0], starting_piece[1]]]);
        
                        } else if current_y == -1 && player == 2{
                            final_moves.push(vec![[self.data[starting_piece[1]][starting_piece[0]], PLAYER_1_GOAL, PLAYER_1_GOAL], [0, starting_piece[0], starting_piece[1]]]);
                            
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
                            let mut total_path: Vec<[i8; 2]> = previous_path.clone();
                            total_path.append(&mut current_path);
                            
                            let mut total_banned_bounces: Vec<[i8; 2]> = previous_banned_bounces.clone();
                            
                            if !(total_banned_bounces.contains(&[current_x ,current_y])) {
                                total_banned_bounces.push([current_x ,current_y]);
    
                                for drop in current_player_drops.iter() {
                                    final_moves.push(vec![[self.data[current_y as usize][current_x as usize], drop[0], drop[1]], [self.data[starting_piece[1]][starting_piece[0]], current_x as usize ,current_y as usize], [0, starting_piece[0], starting_piece[1]]]);
    
                                }
                                
                                let mut moves: Vec<Vec<[usize; 3]>> = self.get_piece_moves(&[current_x as usize ,current_y as usize], starting_piece, &mut total_path, &mut total_banned_bounces, player, current_player_drops);
                                final_moves.append(&mut moves);
    
                            }
    
                        } else {
                            final_moves.push(vec![[self.data[starting_piece[1]][starting_piece[0]], current_x as usize, current_y as usize], [0, starting_piece[0], starting_piece[1]]]);
    
                        }
    
                    } 
    
                }
                
            }
    
        } else if piece == TWO_PIECE {
            for path in self.two_moves.iter() {
                let mut current_x: i8 = current_piece[0] as i8;
                let mut current_y: i8 = current_piece[1] as i8;
        
                let mut current_path: Vec<[i8; 2]> = Vec::new();
        
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
                            final_moves.push(vec![[self.data[starting_piece[1]][starting_piece[0]], PLAYER_2_GOAL, PLAYER_2_GOAL], [0, starting_piece[0], starting_piece[1]]]);
        
                        } else if current_y == -1 && player == 2{
                            final_moves.push(vec![[self.data[starting_piece[1]][starting_piece[0]], PLAYER_1_GOAL, PLAYER_1_GOAL], [0, starting_piece[0], starting_piece[1]]]);
                            
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
                            let mut total_path: Vec<[i8; 2]> = previous_path.clone();
                            total_path.append(&mut current_path);
                            
                            let mut total_banned_bounces: Vec<[i8; 2]> = previous_banned_bounces.clone();
                            
                            if !(total_banned_bounces.contains(&[current_x ,current_y])) {
                                total_banned_bounces.push([current_x ,current_y]);
    
                                for drop in current_player_drops.iter() {
                                    final_moves.push(vec![[self.data[current_y as usize][current_x as usize], drop[0], drop[1]], [self.data[starting_piece[1]][starting_piece[0]], current_x as usize ,current_y as usize], [0, starting_piece[0], starting_piece[1]]]);
    
                                }
                                
                                let mut moves: Vec<Vec<[usize; 3]>> = self.get_piece_moves(&[current_x as usize ,current_y as usize], starting_piece, &mut total_path, &mut total_banned_bounces, player, current_player_drops);
                                final_moves.append(&mut moves);
    
                            }
    
                        } else {
                            final_moves.push(vec![[self.data[starting_piece[1]][starting_piece[0]], current_x as usize, current_y as usize], [0, starting_piece[0], starting_piece[1]]]);
    
                        }
    
                    } 
    
                }
                
            }
    
        } else if piece == THREE_PIECE {
            for path in self.three_moves.iter() {
                let mut current_x: i8 = current_piece[0] as i8;
                let mut current_y: i8 = current_piece[1] as i8;
        
                let mut current_path: Vec<[i8; 2]> = Vec::new();
        
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
                            final_moves.push(vec![[self.data[starting_piece[1]][starting_piece[0]], PLAYER_2_GOAL, PLAYER_2_GOAL], [0, starting_piece[0], starting_piece[1]]]);
        
                        } else if current_y == -1 && player == 2{
                            final_moves.push(vec![[self.data[starting_piece[1]][starting_piece[0]], PLAYER_1_GOAL, PLAYER_1_GOAL], [0, starting_piece[0], starting_piece[1]]]);
                            
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
                            let mut total_path: Vec<[i8; 2]> = previous_path.clone();
                            total_path.append(&mut current_path);
                            
                            let mut total_banned_bounces: Vec<[i8; 2]> = previous_banned_bounces.clone();
                            
                            if !(total_banned_bounces.contains(&[current_x ,current_y])) {
                                total_banned_bounces.push([current_x ,current_y]);
    
                                for drop in current_player_drops.iter() {
                                    final_moves.push(vec![[self.data[current_y as usize][current_x as usize], drop[0], drop[1]], [self.data[starting_piece[1]][starting_piece[0]], current_x as usize ,current_y as usize], [0, starting_piece[0], starting_piece[1]]]);
    
                                }
                                
                                let mut moves: Vec<Vec<[usize; 3]>> = self.get_piece_moves(&[current_x as usize ,current_y as usize], starting_piece, &mut total_path, &mut total_banned_bounces, player, current_player_drops);
                                final_moves.append(&mut moves);
    
                            }
    
                        } else {
                            final_moves.push(vec![[self.data[starting_piece[1]][starting_piece[0]], current_x as usize, current_y as usize], [0, starting_piece[0], starting_piece[1]]]);
    
                        }
    
                    } 
    
                }
                
            }
    
        }
    
        return final_moves;
    
    }
    
}

fn order_moves(moves: Vec<Vec<[usize; 3]>>) -> Vec<Vec<[usize; 3]>> {
    let mut moves_to_sort: Vec<[usize; 2]> = Vec::new();
    
    for (mv_idx, mv) in moves.iter().enumerate() {
        let mut predicted_score = 0;

        if mv.len() == 3 {
            predicted_score -= 2;

        }

        if mv.len() == 2 {
            predicted_score -= 1;

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

    let mut ordered_moves: Vec<Vec<[usize; 3]>> = Vec::new();

    for item in &moves_to_sort {
        ordered_moves.push(moves[item[1]].clone());
        
    }

    return ordered_moves;

}


fn main() {
    let mut board = Board {
        data: [ [0 ,0 ,0 ,0, 0, 0],
                [0 ,0 ,0 ,0, 0, 0],
                [0 ,0 ,0 ,0, 0, 0],
                [0 ,0 ,0 ,0, 0, 0],
                [0 ,0 ,0 ,0, 0, 0], 
                [0 ,0 ,0 ,0, 0, 0]],
        goals: [0, 0],
        one_moves: [[0], [1], [2], [3]],
        two_moves: [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 1], [2, 2], [2, 3], [3, 2], [3, 3], [3, 0], [0, 3]],
        three_moves: [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 2], [1, 2, 1], [2, 1, 1], [1, 2, 2], [2, 1, 2], [2, 2, 1], [2, 2, 3], [2, 3, 2], [3, 2, 2], [2, 3, 3], [3, 2, 3], [3, 3, 2], [3, 3, 0], [3, 0, 3], [0, 3, 3], [3, 0, 0], [0, 3, 0], [0, 0, 3], [3, 0, 1], [1, 0, 3], [0, 1, 2], [2, 1, 0], [1, 2, 3], [3, 2, 1], [2, 3, 0], [0, 3, 2]],

    };

    board.print_board();

    let start = std::time::Instant::now();

    board.get_best_move(2);

    let elapsed_time = start.elapsed();
    println!("DONE! in {} seconds.", elapsed_time.as_secs_f64());

}
