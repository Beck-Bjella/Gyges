
pub struct Backtracker {
    banned_positions: [bool; 36],
    movement_board: [Vec<usize>; 36],
    
}

impl Backtracker {
    pub fn new() -> Backtracker {
        let banned_positions: [bool; 36] = [false, false, false, false, false, false,
                                            false, false, false, false, false, false,
                                            false, false, false, false, false, false,
                                            false, false, false, false, false, false,
                                            false, false, false, false, false, false,
                                            false, false, false, false, false, false];

        let movement_board: [Vec<usize>; 36] = [vec![] ,vec![], vec![] ,vec![], vec![], vec![],
                                                vec![] ,vec![], vec![] ,vec![], vec![], vec![],
                                                vec![] ,vec![], vec![] ,vec![], vec![], vec![],
                                                vec![] ,vec![], vec![] ,vec![], vec![], vec![],
                                                vec![] ,vec![], vec![] ,vec![], vec![], vec![],
                                                vec![] ,vec![], vec![] ,vec![], vec![], vec![]];

        return Backtracker {
            banned_positions: banned_positions,
            movement_board: movement_board,

        }

    }

    pub fn is_banned(&mut self, pos: usize) -> bool {
        return self.banned_positions[pos];

    }

    pub fn set_banned(&mut self, pos: usize) {
        self.banned_positions[pos] = true;

    }

    pub fn set_unbanned(&mut self, pos: usize) {
        self.banned_positions[pos] = false;

    }

    pub fn push_three_path(&mut self, path: &[usize; 4]) {
        self.movement_board[path[0]].push(path[1]);

        self.movement_board[path[1]].push(path[2]);
        self.movement_board[path[1]].push(path[0]);

        self.movement_board[path[2]].push(path[3]);
        self.movement_board[path[2]].push(path[1]);

        self.movement_board[path[3]].push(path[2]);

    }
    
    pub fn pop_three_path(&mut self, path: &[usize; 4]) {
        self.movement_board[path[0]].pop();

        self.movement_board[path[1]].pop();
        self.movement_board[path[1]].pop();

        self.movement_board[path[2]].pop();
        self.movement_board[path[2]].pop();

        self.movement_board[path[3]].pop();

    }

    pub fn is_backtrack_three_path(&mut self, path: &[usize; 4]) -> bool {
        if self.movement_board[path[0]].contains(&path[1]) {
            return true;

        }

        if self.movement_board[path[1]].contains(&path[2]) {
            return true;

        }

        if self.movement_board[path[2]].contains(&path[3]) {
            return true;

        }

        return false;

    }

    pub fn push_two_path(&mut self, path: &[usize; 3]) {
        self.movement_board[path[0]].push(path[1]);

        self.movement_board[path[1]].push(path[2]);
        self.movement_board[path[1]].push(path[0]);

        self.movement_board[path[2]].push(path[1]);

    }

    pub fn pop_two_path(&mut self, path: &[usize; 3]) {
        self.movement_board[path[0]].pop();

        self.movement_board[path[1]].pop();
        self.movement_board[path[1]].pop();

        self.movement_board[path[2]].pop();

    }

    pub fn is_backtrack_two_path(&mut self, path: &[usize; 3]) -> bool {
        if self.movement_board[path[0]].contains(&path[1]) {
            return true;

        }

        if self.movement_board[path[1]].contains(&path[2]) {
            return true;

        }

        

        return false;

    }

    pub fn push_one_path(&mut self, path: &[usize; 2]) {
        self.movement_board[path[0]].push(path[1]);

        self.movement_board[path[1]].push(path[0]);

    }

    pub fn pop_one_path(&mut self, path: &[usize; 2]) {
        self.movement_board[path[0]].pop();

        self.movement_board[path[1]].pop();

    }

    pub fn is_backtrack_one_path(&mut self, path: &[usize; 2]) -> bool {
        if self.movement_board[path[0]].contains(&path[1]) {
            return true;

        }

        return false;

    }

}
