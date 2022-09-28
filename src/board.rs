use crate::move_generation::*;

#[derive(Clone)]
pub struct BoardState {
    pub data: [usize; 38],

}

impl BoardState {
    pub fn new() -> BoardState {
        return BoardState {
            data: [0; 38],

        };
        
    }

    pub fn to_training_data(&self) -> [f64; 36] {
        let mut data: [f64; 36] = [0.0; 36];
        for i in 0..36 {
            data[i] = self.data[i] as f64;
    
        }

        return data;

    }

    pub fn set_rank(&mut self, rank_data: [usize; 6], rank: usize) {
        for x in 0..6 {
            self.data[x + (rank * 6)] = rank_data[x];

        }
  
    }

    pub fn set_goals(&mut self, goal_data: [usize; 2]) {
        self.data[36] = goal_data[0];
        self.data[37] = goal_data[1];


    }

    pub fn print(&self) {
        println!(" ");
        if self.data[37] == 0 {
            println!("                .");

        } else {
            println!("                {}", self.data[37]);

        }
        println!(" ");
        println!(" ");

        for y in (0..6).rev() {
            for x in 0..6 {
                if self.data[y * 6 + x] == 0 {
                    print!("    .");
                } else {
                    print!("    {}", self.data[y * 6 + x]);

                }
               
            }
            
            println!(" ");
            println!(" ");

        }

        println!(" ");
        if self.data[36] == 0 {
            println!("                .");

        } else {
            println!("                {}", self.data[36]);

        }
        println!(" ");
    
    }

    pub fn make_move(&mut self, mv: &Move) {
        let step1 = [mv.data[0], mv.data[1]];
        let step2 = [mv.data[2], mv.data[3]];
        let step3 = [mv.data[4], mv.data[5]];
        
        if mv.flag == MoveType::Drop {
            self.data[step1[1]] = step1[0];
            self.data[step2[1]] = step2[0];
            self.data[step3[1]] = step3[0];

        } else if mv.flag == MoveType::Bounce {
            self.data[step2[1]] = step2[0];
            self.data[step1[1]] = 0;

        }

    }

    pub fn undo_move(&mut self, mv: &Move) {
        let step1 = [mv.data[0], mv.data[1]];
        let step2 = [mv.data[2], mv.data[3]];
        let step3 = [mv.data[4], mv.data[5]];

        if mv.flag == MoveType::Drop {
            self.data[step3[1]] = step1[0];
            self.data[step2[1]] = step3[0];
            self.data[step1[1]] = step2[0];
            
        } else if mv.flag == MoveType::Bounce {
            self.data[step2[1]] = 0;
            self.data[step1[1]] = step2[0];

        }

    }
    
    pub fn get_active_lines(&self) -> [usize; 2] {
        let mut player_1_set = false;
    
        let mut player_1_active_line = 9;
        let mut player_2_active_line = 9;
    
        for y in 0..6 {
            for x in 0..6 {
                if self.data[y * 6 + x] != 0 {
                    if !player_1_set {
                        player_1_active_line = y;
                        player_1_set = true;
    
                    }
                    player_2_active_line = y;
                    
                }
            }
        }
    
        return [player_1_active_line * 6, player_2_active_line * 6];
    
    }

    pub fn get_drops(&self, active_lines: [usize; 2], player: i8) -> Vec<usize> {
        let mut current_player_drops: Vec<usize> = vec![];

        if player == 2 {
            for i in active_lines[0]..36 {
                if self.data[i] == 0 {
                    current_player_drops.push(i);

                }
    
            }

        } else {
            for i in 0..active_lines[1] + 6 {
                if self.data[i] == 0 {
                    current_player_drops.push(i);

                }
                
            }

        }

        return current_player_drops;

    }

    pub fn is_valid(&self) {
        let mut one_count = 0;
        let mut two_count = 0;
        let mut three_count = 0;

        for i in 0..36 {
            if self.data[i] == 1 {
                one_count += 1;

            } else if self.data[i] == 2 {
                two_count += 1;

            } else if self.data[i] == 3 {
                three_count += 1;

            } 


        }

        if !(one_count == 4 && two_count == 4 && three_count == 4) {
            panic!("ERROR INVALD BOARD");

        }

    }
    
    pub fn flip(&mut self) {
        let mut temp_data: [usize; 38] = [0 ,0 ,0 ,0, 0, 0,
                                            0 ,0 ,0 ,0, 0, 0,
                                            0 ,0 ,0 ,0, 0, 0,
                                            0 ,0 ,0 ,0, 0, 0,
                                            0 ,0 ,0 ,0, 0, 0, 
                                            0 ,0 ,0 ,0, 0, 0,
                                            
                                            0, 0];
    
        for y in 0..6 {
            for x in 0..6 {
                let piece = self.data[y * 6 + x];
    
                temp_data[((5 - y) * 6) + (5 - x)] = piece;
    
            }
    
        }
    
        self.data = temp_data;
    
    }

}
