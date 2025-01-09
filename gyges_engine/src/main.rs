extern crate gyges_engine;

use gyges::moves::new_movegen::*;
use gyges::moves::movegen::*;
use gyges::*;

use gyges_engine::consts::*;
use gyges_engine::ugi::*;

fn main() {
    let mut ugi = Ugi::new();
    ugi.start();

    // unsafe {
    //     benchmark();
    // }

}

pub unsafe fn benchmark() {
    let mut mg: MoveGen = MoveGen::default();

    // let mut board = BoardState::from([0,0,2,0,0,0,1,0,3,3,0,0,0,2,1,1,0,0,0,3,2,0,0,0,0,0,0,3,2,0,0,0,0,1,0,0,0,0]);
    
    let mut board = BoardState::from(STARTING_BOARD);
    let player = Player::One;

    println!("Initial board state: \n{}", board);
    println!("Valid Move Count: ");
    let iters = 1000000;
    let batchs = 3;
    for b in 0..batchs {
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let _mc = unsafe { threat_or_movecount(&mut board, player) };
        }
        let elapsed = start.elapsed();
        let elapsed = elapsed.as_secs_f64();
        let time_per_iter = elapsed / iters as f64;
        let iters_per_sec = 1.0 / time_per_iter;
        println!("  {}: {} g/s", b, iters_per_sec);
    }
    println!("");

    println!("NEW TEST: ");
    let iters = 1000000;
    let batchs = 3;
    for b in 0..batchs {
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let _mc = mg.gen::<GenMoveCount, QuitOnThreat>(&mut board, player);
        }
        let elapsed = start.elapsed();
        let elapsed = elapsed.as_secs_f64();
        let time_per_iter = elapsed / iters as f64;
        let iters_per_sec = 1.0 / time_per_iter;
        println!("  {}: {} g/s", b, iters_per_sec);
    }
    println!("");

    println!("THREADED NEW TEST: ");
    THREAD_LOCAL_MOVEGEN.with(|movegen| {
        let mut movegen = movegen.borrow_mut();

        let iters = 1000000;
        let batchs = 3;
        for b in 0..batchs {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let _mc = movegen.gen::<GenMoveCount, QuitOnThreat>(&mut board, player);
            }
            let elapsed = start.elapsed();
            let elapsed = elapsed.as_secs_f64();
            let time_per_iter = elapsed / iters as f64;
            let iters_per_sec = 1.0 / time_per_iter;
            println!("  {}: {} g/s", b, iters_per_sec);
        }

    });
    

}

// #[inline(always)]
// pub unsafe fn compress_pext(mask: u64, val: u64) -> u16 {
//     core::arch::x86_64::_pext_u64(val, mask) as u16
// }

// pub fn generate_three_map() {
//     // Initialize map
//     let map_size = 2usize.pow(12); // 4096
//     let mut map: Vec<Vec<u16>> = Vec::new();
//     for i in 0..36 {
//         let mut row = Vec::new();
//         for j in 0..map_size {
//             row.push(0);
//         }
//         map.push(row);
//     }

//     // Generate map
//     for pos in 0..36 {
//         //////// ALL POSITIONS ////////

//         let mut intercept_bb = BitBoard(ALL_THREE_INTERCEPTS[pos as usize]);

//         let positions: Vec<usize> = intercept_bb.get_data();
//         for size in 0..=positions.len() {
//             let combinations = positions.iter().combinations(size);

//             for combo in combinations {
//                 let mut temp_intercepts: u64 = 0;
//                 for c in &combo {
//                     temp_intercepts |= 1 << *c;
//                 }

//                 //////// ALL COMBINATIONS ////////

//                 let key = unsafe {
//                     compress_pext(ALL_THREE_INTERCEPTS[pos as usize], temp_intercepts as u64)
//                 };

//                 let valid_paths_idx = unsafe {
//                     THREE_MAP
//                         .get_unchecked(pos as usize)
//                         .get_unchecked(temp_intercepts as usize % 11007)
//                 };
//                 map[pos][key as usize] = *valid_paths_idx;

//                 //////// ALL COMBINATIONS ////////
//             }
//         }

//         //////// ALL POSITIONS ////////
//     }

//     // Write map to file
//     use std::io::Write;
//     let mut file = std::fs::File::create("new_three_map.rs").unwrap();

//     writeln!(file, "pub const NEW_THREE_MAP: [[u16; 4096]; 36] = [").unwrap();
//     for i in 0..36 {
//         write!(file, "    [").unwrap();
//         for j in 0..map_size {
//             write!(file, "{}, ", map[i][j]).unwrap();
//         }
//         writeln!(file, "],").unwrap();
//     }
// }

// pub fn generate_two_map() {
//     // Initialize map
//     let map_size = 2usize.pow(4);
//     let mut map: Vec<Vec<u16>> = Vec::new();
//     for i in 0..36 {
//         let mut row = Vec::new();
//         for j in 0..map_size {
//             row.push(0);
//         }
//         map.push(row);
//     }

//     // Generate map
//     for pos in 0..36 {
//         //////// ALL POSITIONS ////////

//         let mut intercept_bb = BitBoard(ALL_TWO_INTERCEPTS[pos as usize]);

//         let positions: Vec<usize> = intercept_bb.get_data();
//         for size in 0..=positions.len() {
//             let combinations = positions.iter().combinations(size);

//             for combo in combinations {
//                 let mut temp_intercepts: u64 = 0;
//                 for c in &combo {
//                     temp_intercepts |= 1 << *c;
//                 }

//                 //////// ALL COMBINATIONS ////////

//                 let key = unsafe {
//                     compress_pext(ALL_TWO_INTERCEPTS[pos as usize], temp_intercepts as u64)
//                 };

//                 let valid_paths_idx = unsafe {
//                     TWO_MAP
//                         .get_unchecked(pos as usize)
//                         .get_unchecked(temp_intercepts as usize % 29)
//                 };
//                 map[pos][key as usize] = *valid_paths_idx;

//                 //////// ALL COMBINATIONS ////////
//             }
//         }

//         //////// ALL POSITIONS ////////
//     }

//     // Write map to file
//     use std::io::Write;
//     let mut file = std::fs::File::create("new_two_map.rs").unwrap();

//     writeln!(file, "pub const NEW_TWO_MAP: [[u16; 16]; 36] = [").unwrap();
//     for i in 0..36 {
//         write!(file, "    [").unwrap();
//         for j in 0..map_size {
//             write!(file, "{}, ", map[i][j]).unwrap();
//         }
//         writeln!(file, "],").unwrap();
//     }
// }

// pub fn generate_combined_three_lists() {
//     let mut path_lists: Vec<ThreePathList> = Vec::new();

//     for path_list in UNIQUE_THREE_PATH_LISTS {
//         let mut new = ThreePathList {
//             paths: [([0; 4], 0); 36],
//             count: 0,
//         };

//         for i in 0..path_list[35] {
//             let data = UNIQUE_THREE_PATHS[path_list[i as usize] as usize];
//             new.paths[i as usize] = data;
//         }
//         new.count = path_list[35] as u8;

//         path_lists.push(new);
//     }

//     // Write lists to file
//     use std::io::Write;
//     let mut file = std::fs::File::create("NEW_THREE_PATH_LISTS.rs").unwrap();

//     writeln!(
//         file,
//         "pub const NEW_THREE_PATH_LISTS: [ThreePathList; 8389] = ["
//     )
//     .unwrap();
//     for i in 0..path_lists.len() {
//         write!(file, "    ThreePathList {{ paths: [").unwrap();
//         for j in 0..36 {
//             write!(
//                 file,
//                 "([{}, {}, {}, {}], {}), ",
//                 path_lists[i].paths[j].0[0],
//                 path_lists[i].paths[j].0[1],
//                 path_lists[i].paths[j].0[2],
//                 path_lists[i].paths[j].0[3],
//                 path_lists[i].paths[j].1
//             )
//             .unwrap();
//         }
//         writeln!(file, "], count: {} }},", path_lists[i].count).unwrap();
//     }
// }

// pub fn generate_combined_two_lists() {
//     let mut path_lists: Vec<TwoPathList> = Vec::new();

//     for path_list in UNIQUE_TWO_PATH_LISTS {
//         let mut new = TwoPathList {
//             paths: [([0; 3], 0); 12],
//             count: 0,
//         };

//         for i in 0..path_list[12] {
//             let data = UNIQUE_TWO_PATHS[path_list[i as usize] as usize];
//             new.paths[i as usize] = data;
//         }
//         new.count = path_list[12] as u8;

//         path_lists.push(new);
//     }

//     println!("Path Lists: {}", path_lists.len());

//     // Write lists to file
//     use std::io::Write;
//     let mut file: std::fs::File = std::fs::File::create("new_two_path_lists.rs").unwrap();

//     writeln!(file, "pub const NEW_TWO_PATH_LISTS: [TwoPathList; 365] = [").unwrap();
//     for i in 0..path_lists.len() {
//         write!(file, "    TwoPathList {{ paths: [").unwrap();
//         for j in 0..12 {
//             write!(
//                 file,
//                 "([{}, {}, {}], {}), ",
//                 path_lists[i].paths[j].0[0],
//                 path_lists[i].paths[j].0[1],
//                 path_lists[i].paths[j].0[2],
//                 path_lists[i].paths[j].1
//             )
//             .unwrap();
//         }
//         writeln!(file, "], count: {} }},", path_lists[i].count).unwrap();
//     }
// }

// pub fn generate_combined_one_lists() {
//     let mut path_lists: Vec<OnePathList> = Vec::new();

//     for path_list in UNIQUE_ONE_PATH_LISTS {
//         let mut new = OnePathList {
//             paths: [([0; 2], 0); 4],
//             count: 0,
//         };

//         for i in 0..path_list[4] {
//             let data = UNIQUE_ONE_PATHS[path_list[i as usize] as usize];
//             new.paths[i as usize] = data;
//         }
//         new.count = path_list[4] as u8;

//         path_lists.push(new);
//     }

//     println!("Path Lists: {}", path_lists.len());

//     // Write lists to file
//     use std::io::Write;
//     let mut file: std::fs::File = std::fs::File::create("new_one_path_lists.rs").unwrap();

//     writeln!(file, "pub const NEW_ONE_PATH_LISTS: [OnePathList; 133] = [").unwrap();
//     for i in 0..path_lists.len() {
//         write!(file, "    OnePathList {{ paths: [").unwrap();
//         for j in 0..4 {
//             write!(
//                 file,
//                 "([{}, {}], {}), ",
//                 path_lists[i].paths[j].0[0], path_lists[i].paths[j].0[1], path_lists[i].paths[j].1
//             )
//             .unwrap();
//         }
//         writeln!(file, "], count: {} }},", path_lists[i].count).unwrap();
//     }
// }
