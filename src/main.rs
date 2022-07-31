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
    fn print(&self) { 
        println!(" ");
        println!("         {}", self.goals[0]);
        println!(" ");
        for y in 0..6 {
            for x in 0..6 {
                if self.data[y][x] == 0 {
                    print!("  .");
                } else {
                    print!("  {}", self.data[y][x]);

                }
               
            }
            println!(" ");
        }
        println!(" ");
        println!("         {}",  self.goals[1]);
        println!(" ");
    
    }
    
    fn evalulate(&self) -> usize {
        let starting_value = usize::MAX / 2;
        let player_1_moves = self.valid_moves(1);
        let player_2_moves = self.valid_moves(2);

        let score = starting_value + (player_2_moves.len() - player_1_moves.len());

        return score;
    
    }

    fn threat_count(&self, player: i8) -> usize {
        let mut threats = 0;
        let moves = self.valid_moves(player);
        for mv in moves {
            if player == 1 {
                if mv[0][1] == PLAYER_2_GOAL {
                    threats += 1;
    
                }
    
            } else {
                if mv[0][1] == PLAYER_1_GOAL {
                    threats += 1;
    
                }
    
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

    fn mini_max(&mut self, mut alpha: usize, mut beta: usize, is_maximisizing: bool, depth: i8) -> [usize; 2] {
        if depth == 0 {
            let score = self.evalulate();
            return [score, 0];
    
        }

        if is_maximisizing {
            if self.threat_count(2) > 0 {
                return [usize::MAX, depth as usize];
        
            }

            let current_moves = self.valid_moves(2);
    
            let mut max_eval: [usize; 2] = [usize::MIN, usize::MIN];
            let mut used_moves: Vec<Vec<[usize; 3]>> = Vec::new();
            for mv in current_moves.iter() {
                if used_moves.contains(mv) {
                    continue;
                }
                
                self.make_move(&mv);
    
                used_moves.push(mv.to_vec());
    
                let curr_eval: [usize; 2];
                curr_eval = self.mini_max(alpha, beta, false, depth - 1);
                
                self.undo_move(&mv);

                if curr_eval[0] > max_eval[0] {
                    max_eval = curr_eval;

                } else if curr_eval[0] == max_eval[0] {
                    if curr_eval[1] > max_eval[1] {
                        max_eval = curr_eval;

                    }

                }
    
                alpha = max!(alpha, curr_eval[0]);
                if alpha >= beta {
                    break
    
                }
                    
            }
            
            return max_eval;
    
        } else {
            if self.threat_count(1) > 0 {
                return [usize::MIN, depth as usize];
        
            }

            let current_moves = self.valid_moves(1);
    
            let mut min_eval: [usize; 2] = [usize::MAX, usize::MIN];
            let mut used_moves: Vec<Vec<[usize; 3]>> = Vec::new();
            for mv in current_moves.iter() {
                if used_moves.contains(mv) {
                    continue;
                }
    
                used_moves.push(mv.to_vec());
    
                self.make_move(&mv);
                
                let curr_eval: [usize; 2];
                curr_eval = self.mini_max(alpha, beta, true, depth - 1);
                
                self.undo_move(&mv);

                if curr_eval[0] < min_eval[0] {
                    min_eval = curr_eval;

                } else if curr_eval[0] == min_eval[0] {
                    if curr_eval[1] > min_eval[1] {
                        min_eval = curr_eval;

                    }

                }
    
                beta = min!(beta, curr_eval[0]);
                if beta <= alpha {
                    break;
    
                }
                    
            }
    
            return min_eval;
    
        }
    
    }

    fn get_best_move(&mut self, depth: i8) -> Vec<[usize; 3]> {
        if !self.is_valid() {
            panic!("NOT A VAILD BOARD POSITION");

        }

        let mut current_moves = self.valid_moves(2);
        current_moves = self.order_moves(current_moves);

        for mv in current_moves.iter() {
            if mv[0][1] == PLAYER_1_GOAL {
                return mv.to_vec();   

            }

        }

        let mut alpha = usize::MIN;
        let beta = usize::MAX;
        let mut max_eval: [usize; 3] = [usize::MIN; 3];
        
        let mut used_moves: Vec<Vec<[usize; 3]>> = Vec::new();
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

            used_moves.push(mv.to_vec());
            
            println!("Starting Index {}", move_idx);
            
            let eval: [usize; 3];
            let temp_eval = self.mini_max(alpha, beta, false, depth - 1);
            eval = [temp_eval[0], temp_eval[1], move_idx];
            
            self.undo_move(&mv);

            if eval[0] > max_eval[0] {
                max_eval = eval;

            } else if eval[0] == max_eval[0] {
                if eval[1] > max_eval[1] {
                    max_eval = eval;

                }

            }

            alpha = max!(alpha, eval[0]);

        }

        println!("{:?}", max_eval);
        return (&current_moves[max_eval[2]]).to_vec();
        
    }

    fn valid_moves(&self, player: i8) -> Vec<Vec<[usize; 3]>> {
        let active_lines = self.active_lines();
    
        if player == 1 {
            let mut player_1_drops: Vec<[usize; 2]> = Vec::new();
            for y in 0..active_lines[1] + 1 {
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
    
        } else {
            let mut player_2_drops: Vec<[usize; 2]> = Vec::new();
            for y in (active_lines[0]..6).rev() {
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
   
    fn is_valid(&self) -> bool{
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

        if one_count == 4 && two_count == 4 && three_count == 4 {
            return true;

        }

        return false;

    }

    fn is_tie(&self) -> bool {
        let player_1_moves = self.valid_moves(1);
        let player_2_moves = self.valid_moves(2);

        if player_1_moves.len() == 0 || player_2_moves.len() == 0 {
            return true;

        }

        return false;

    }

    fn order_moves(&mut self, moves: Vec<Vec<[usize; 3]>>) -> Vec<Vec<[usize; 3]>> {
        let mut moves_to_sort: Vec<[usize; 2]> = Vec::new();
        
        for (mv_idx, mv) in moves.iter().enumerate() {
            let mut predicted_score = 0;

            self.make_move(&mv);

            let threats = self.threat_count(2);

            self.undo_move(&mv);

            predicted_score -= threats;
    
            if mv.len() == 3 {
                predicted_score -= 200;
    
            }
    
            if mv.len() == 2 {
                predicted_score -= 100;
    
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
    
    fn _simualate_game(&mut self, depth: i8, starting_player: i8) {
        let mut player = starting_player;
    
        loop {
            println!(" -------------------- Player {} turn --------------------", player);
            println!("");
    
            println!("BEFORE ->");
            if player == 2 {
                self.print();
            } else {
                self._flip();
                self.print();
                self._flip();
                
            }
            
            let best_move = self.get_best_move(depth);
            self.make_move(&best_move);
    
            println!("AFTER ->");
            if player == 2 {
                self.print();
            } else {
                self._flip();
                self.print();
                self._flip();
                
            }
    
            if self.goals[0] == 1 {
                println!("PLAYER {} WINS", player);
                println!("");
                println!("----------------------------------------");
                break;
    
            }
    
            self._flip();
            if player == 1 {
                player = 2;
    
            } else {
                player = 1;
    
            }
    
        }
    
    }

}


fn main() {
    let mut board = Board {
        data: [ [0 ,0, 0 ,0, 3, 3],
                [2 ,3 ,0 ,1, 3, 0],
                [0 ,1 ,2 ,2, 0, 0],
                [0 ,1 ,1 ,0, 0, 0],
                [0 ,0 ,0 ,2, 0, 0], 
                [0 ,0 ,0 ,0, 0, 0]],
        goals: [0, 0],
        one_moves: [[0], [1], [2], [3]],
        two_moves: [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 1], [2, 2], [2, 3], [3, 2], [3, 3], [3, 0], [0, 3]],
        three_moves: [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 2], [1, 2, 1], [2, 1, 1], [1, 2, 2], [2, 1, 2], [2, 2, 1], [2, 2, 3], [2, 3, 2], [3, 2, 2], [2, 3, 3], [3, 2, 3], [3, 3, 2], [3, 3, 0], [3, 0, 3], [0, 3, 3], [3, 0, 0], [0, 3, 0], [0, 0, 3], [3, 0, 1], [1, 0, 3], [0, 1, 2], [2, 1, 0], [1, 2, 3], [3, 2, 1], [2, 3, 0], [0, 3, 2]],

    };
    
    board.print();

    let start = std::time::Instant::now();

    // board.simualate_game(3);
    println!("{:?}", board.get_best_move(3));

    let elapsed_time = start.elapsed();
    println!("DONE! in {} seconds.", elapsed_time.as_secs_f64());

}
