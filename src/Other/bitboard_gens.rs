
fn board_to_bitboard(board: &BoardState) -> String {
    let mut bit_board_string = String::from("");

    let piece = board.data[37];
    let piece_string = piece.to_string().chars().nth(0).unwrap();
    bit_board_string.push(piece_string);

    let piece = board.data[36];
    let piece_string = piece.to_string().chars().nth(0).unwrap();
    bit_board_string.push(piece_string);

    for y in 0..6 {
        for x in (0..6).rev()  {

            let mut piece = board.data[(y * 6) + x];
            if piece != 0 {
                piece = 1;

            }
            let piece_string = piece.to_string().chars().nth(0).unwrap();
            bit_board_string.push(piece_string);
    
        }

    }

    return bit_board_string;

}


fn three_path_to_middle_points_bb(path: [usize; 4]) -> String {
    let mut bb: String = String::from("000000000000000000000000000000000000000000000000000000000000");

    if path[3] == PLAYER_1_GOAL || path[3] == PLAYER_2_GOAL {
        let old_to_new_cords: [i8; 36] = [30, 31, 32, 33, 34, 35,
        24, 25, 26, 27, 28, 29, 
        18, 19, 20, 21, 22, 23,
        12, 13, 14, 15, 16, 17, 
        6, 7, 8, 9, 10, 11,
        0, 1, 2, 3, 4, 5];


                
        let start = old_to_new_cords[path[0]];
        let s1 = old_to_new_cords[path[1]];
        let s2 = old_to_new_cords[path[2]];

        let dir1: i8 = s1 as i8 - start as i8;
        let dir2 = s2 as i8 - s1 as i8;

        let index_change_1: i8 = match dir1 {
        -1 => -2,
        6 => 22,
        1 => 2,
        -6 => -22,
        _ => 0

        };

        let index_change_2: i8 = match dir2 {
        -1 => -2,
        6 => 22,
        1 => 2,
        -6 => -22,
        _ => 0

        };

        let small_to_big_cords: [i8; 36] =  [0, 2, 4, 6, 8, 10,
                22, 24, 26, 28, 30, 32, 
                44, 46, 48, 50, 52, 54, 
                66, 68, 70, 72, 74, 76, 
                88, 90, 92, 94, 96, 98, 
                110, 112, 114, 116, 118, 120];
                

        let middle_pos_1 = small_to_big_cords[start as usize] as f64 + (index_change_1 as f64 / 2.0);
        let middle_pos_2 = small_to_big_cords[s1 as usize] as f64 + (index_change_2 as f64 / 2.0);

        let final_pos_1 = ((middle_pos_1 - 1.0) / 2.0) as usize;
        let final_pos_2 = ((middle_pos_2 - 1.0) / 2.0) as usize;

        // println!("{} {} {} {}, ||| {} {} {}", start, s1, s2, end, final_pos_1, final_pos_2, final_pos_3);

        bb.replace_range(final_pos_1..final_pos_1 + 1,"1");
        bb.replace_range(final_pos_2..final_pos_2 + 1,"1");

        return bb.chars().rev().collect::<String>();

    }

    let old_to_new_cords: [i8; 36] = [30, 31, 32, 33, 34, 35,
                                        24, 25, 26, 27, 28, 29, 
                                        18, 19, 20, 21, 22, 23,
                                        12, 13, 14, 15, 16, 17, 
                                        6, 7, 8, 9, 10, 11,
                                        0, 1, 2, 3, 4, 5];


                                        
    let start = old_to_new_cords[path[0]];
    let s1 = old_to_new_cords[path[1]];
    let s2 = old_to_new_cords[path[2]];
    let end = old_to_new_cords[path[3]];

    let dir1: i8 = s1 as i8 - start as i8;
    let dir2 = s2 as i8 - s1 as i8;
    let dir3 = end as i8 - s2 as i8;

    let index_change_1: i8 = match dir1 {
        -1 => -2,
        6 => 22,
        1 => 2,
        -6 => -22,
        _ => 0

    };

    let index_change_2: i8 = match dir2 {
        -1 => -2,
        6 => 22,
        1 => 2,
        -6 => -22,
        _ => 0

    };

    let index_change_3: i8 = match dir3 {
        -1 => -2,
        6 => 22,
        1 => 2,
        -6 => -22,
        _ => 0

    };

    let small_to_big_cords: [i8; 36] =  [0, 2, 4, 6, 8, 10,
                                        22, 24, 26, 28, 30, 32, 
                                        44, 46, 48, 50, 52, 54, 
                                        66, 68, 70, 72, 74, 76, 
                                        88, 90, 92, 94, 96, 98, 
                                        110, 112, 114, 116, 118, 120];
                                        

    let middle_pos_1 = small_to_big_cords[start as usize] as f64 + (index_change_1 as f64 / 2.0);
    let middle_pos_2 = small_to_big_cords[s1 as usize] as f64 + (index_change_2 as f64 / 2.0);
    let middle_pos_3 = small_to_big_cords[s2 as usize] as f64 + (index_change_3 as f64 / 2.0);

    let final_pos_1 = ((middle_pos_1 - 1.0) / 2.0) as usize;
    let final_pos_2 = ((middle_pos_2 - 1.0) / 2.0) as usize;
    let final_pos_3 = ((middle_pos_3 - 1.0) / 2.0) as usize;

    // println!("{} {} {} {}, ||| {} {} {}", start, s1, s2, end, final_pos_1, final_pos_2, final_pos_3);

    bb.replace_range(final_pos_1..final_pos_1 + 1,"1");
    bb.replace_range(final_pos_2..final_pos_2 + 1,"1");
    bb.replace_range(final_pos_3..final_pos_3 + 1,"1");

    return bb.chars().rev().collect::<String>();

}


fn two_path_to_middle_points_bb(path: [usize; 3]) -> String {
    let mut bb: String = String::from("000000000000000000000000000000000000000000000000000000000000");

    if path[2] == PLAYER_1_GOAL || path[2] == PLAYER_2_GOAL {
        let old_to_new_cords: [i8; 36] = [30, 31, 32, 33, 34, 35,
        24, 25, 26, 27, 28, 29, 
        18, 19, 20, 21, 22, 23,
        12, 13, 14, 15, 16, 17, 
        6, 7, 8, 9, 10, 11,
        0, 1, 2, 3, 4, 5];


                
        let start = old_to_new_cords[path[0]];
        let s1 = old_to_new_cords[path[1]];

        let dir1: i8 = s1 as i8 - start as i8;

        let index_change_1: i8 = match dir1 {
        -1 => -2,
        6 => 22,
        1 => 2,
        -6 => -22,
        _ => 0

        };

        let small_to_big_cords: [i8; 36] =  [0, 2, 4, 6, 8, 10,
                22, 24, 26, 28, 30, 32, 
                44, 46, 48, 50, 52, 54, 
                66, 68, 70, 72, 74, 76, 
                88, 90, 92, 94, 96, 98, 
                110, 112, 114, 116, 118, 120];
                

        let middle_pos_1 = small_to_big_cords[start as usize] as f64 + (index_change_1 as f64 / 2.0);

        let final_pos_1 = ((middle_pos_1 - 1.0) / 2.0) as usize;

        // println!("{} {} {} {}, ||| {} {} {}", start, s1, s2, end, final_pos_1, final_pos_2, final_pos_3);

        bb.replace_range(final_pos_1..final_pos_1 + 1,"1");

        return bb.chars().rev().collect::<String>();

    }

    let old_to_new_cords: [i8; 36] = [30, 31, 32, 33, 34, 35,
                                        24, 25, 26, 27, 28, 29, 
                                        18, 19, 20, 21, 22, 23,
                                        12, 13, 14, 15, 16, 17, 
                                        6, 7, 8, 9, 10, 11,
                                        0, 1, 2, 3, 4, 5];


                                        
    let start = old_to_new_cords[path[0]];
    let s1 = old_to_new_cords[path[1]];
    let end = old_to_new_cords[path[2]];

    let dir1: i8 = s1 as i8 - start as i8;
    let dir2 = end as i8 - s1 as i8;

    let index_change_1: i8 = match dir1 {
        -1 => -2,
        6 => 22,
        1 => 2,
        -6 => -22,
        _ => 0

    };

    let index_change_2: i8 = match dir2 {
        -1 => -2,
        6 => 22,
        1 => 2,
        -6 => -22,
        _ => 0

    };

    let small_to_big_cords: [i8; 36] =  [0, 2, 4, 6, 8, 10,
                                        22, 24, 26, 28, 30, 32, 
                                        44, 46, 48, 50, 52, 54, 
                                        66, 68, 70, 72, 74, 76, 
                                        88, 90, 92, 94, 96, 98, 
                                        110, 112, 114, 116, 118, 120];
                                        

    let middle_pos_1 = small_to_big_cords[start as usize] as f64 + (index_change_1 as f64 / 2.0);
    let middle_pos_2 = small_to_big_cords[s1 as usize] as f64 + (index_change_2 as f64 / 2.0);

    let final_pos_1 = ((middle_pos_1 - 1.0) / 2.0) as usize;
    let final_pos_2 = ((middle_pos_2 - 1.0) / 2.0) as usize;

    // println!("{} {} {} {}, ||| {} {} {}", start, s1, s2, end, final_pos_1, final_pos_2, final_pos_3);

    bb.replace_range(final_pos_1..final_pos_1 + 1,"1");
    bb.replace_range(final_pos_2..final_pos_2 + 1,"1");

    return bb.chars().rev().collect::<String>();

}
