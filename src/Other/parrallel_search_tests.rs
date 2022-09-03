

pub fn sort_moves_2(mut evaluations: Vec<(usize, [usize; 6], usize, usize)>) -> Vec<[usize; 6]> {
    evaluations.sort_by(|a, b| {
        if a.0 > b.0 {
            Ordering::Less
            
        } else if a.0 == b.0 {
            Ordering::Equal

        } else {
            Ordering::Greater

        }

    });


    let mut sorted_moves = vec![];

    for item in &evaluations {
        sorted_moves.push(item.1);
        
    }

    return sorted_moves;


}

pub fn chunk_moves(moves: &Vec<[usize; 6]>, chunks: usize) -> Vec<Vec<[usize; 6]>> {
    let mut list_of_lists: Vec<Vec<[usize; 6]>> = vec![];
    for _ in 0..chunks {
        list_of_lists.push(vec![]);

    }

    let mut up = vec![];
    for i in 0..chunks {
        up.push(i);

    }

    let mut down = vec![];
    for i in (0..chunks).rev() {
        down.push(i);

    }

    let mut direction = &up;
    let mut current_index = 0;

    for item in moves {
        list_of_lists[direction[current_index]].push(*item);

        current_index += 1;

        if current_index >= direction.len() && direction == &up{
            direction = &down;
            current_index = 0;

        } else if current_index >= direction.len() && direction == &down {
            direction = &up;
            current_index = 0;

        }
        
    }

    return list_of_lists;


}

pub fn iterative_deepening_parrallel_search(board: &mut BoardState, max_depth: i8, max_time: f64) -> Vec<(usize, [usize; 6], usize, usize)> {
    let start_time = std::time::Instant::now();

    let mut moves = valid_moves(board, 2);

    let mut all_bests: Vec<(usize, [usize; 6], usize, usize)> = vec![];

    for mv in moves.iter() {
        if mv[3] == PLAYER_1_GOAL {
            all_bests.push((usize::MAX, *mv, 0, 0));
            return all_bests;

        }

    }

    let mut current_depth = 1;
    'depth: loop {
        let data: ((usize, [usize; 6], usize, usize), Vec<(usize, [usize; 6], usize, usize)>) = parallel_search(board, moves, current_depth);
        let best_move = data.0;
        let evaluations = data.1;

        all_bests.push(best_move);
        println!("{:?}", best_move);

        moves = sort_moves_2(evaluations);

        current_depth += 2;
        if current_depth > max_depth {
            break 'depth;

        }

    }

    return all_bests;

}

pub fn parallel_search(board: &mut BoardState, moves: Vec<[usize; 6]>, depth: i8) -> ((usize, [usize; 6], usize, usize), Vec<(usize, [usize; 6], usize, usize)>) {
    let mut handles = Vec::new();

    for thread_index in 0..2 {
        let cloned_board_data = board.data.clone();
        let chunked_moves = chunk_moves(&moves, 2);

        let handle = thread::spawn(move || {
            let start_time = std::time::Instant::now();

            let mut new_board = BoardState::new_from(cloned_board_data);

            let mut evaluations: Vec<(usize, [usize; 6], usize, usize)> = vec![];
            
            let mut alpha = usize::MIN;
            let beta = usize::MAX;

            let mut used_moves: Vec<[usize; 6]> = vec![];
            for mv in chunked_moves[thread_index].iter() {
                if used_moves.contains(mv) {
                    continue;

                }
        
                new_board.make_move(&mv);

                if new_board.is_tie() {
                    new_board.undo_move(&mv);
                    continue;

                }

                used_moves.push(*mv);
                
                let minimax_eval: usize = mini_max(&mut new_board, alpha, beta, false, depth - 1);
                let eval: (usize, [usize; 6], usize, usize) = (minimax_eval, *mv, valid_threat_count(&mut new_board, 2), depth as usize);

                evaluations.push(eval);

                new_board.undo_move(&mv);

                alpha = max!(alpha, eval.0);

            }

            println!("DONE! in {} seconds.", start_time.elapsed().as_secs_f64());

            return evaluations;

        });

        handles.push(handle);

    }

    let mut all_evaultaions = vec![];
    
    for handle in handles {
        let val = handle.join().unwrap();
        for item in val {
            all_evaultaions.push(item);

        }

    }

    let mut best_move: (usize, [usize; 6], usize, usize) = (usize::MIN, [NULL, NULL, NULL, NULL, NULL, NULL], 0, depth as usize);

    for eval in all_evaultaions.iter() {
        if eval.0 > best_move.0 {
            best_move = *eval;

        } else if eval.0 == best_move.0 {
            if eval.2 > best_move.2 {
                best_move = *eval;

            }

        }

    }

    return (best_move, all_evaultaions);

}
