//! Contains the Zobrist hashing functions and data.
//! 
//! Zobrist hashing is a hashing algorithm that is used to hash a [`BoardState`]. 
//! Once a hash is generated it can be used as the key for a entry in the transpostion table.
//! 

#[doc(hidden)]
use rand::Rng;

use crate::board::*;
use crate::core::*;

/// P1 hash data
pub const PLAYER_1_HASH: u64 = 11071850447646664432;
/// P2 hash data
pub const PLAYER_2_HASH: u64 = 15525846075063937794;
/// Hash data for each player.
pub const PLAYER_HASH_DATA: [u64; 2] = [PLAYER_1_HASH, PLAYER_2_HASH];

/// Randomly generated hash data for each square and piece combination.
pub const ZOBRIST_HASH_DATA: [[u64; 3]; 38] = [
    [16019584827874668947, 10756861282518694731, 326575046895935871], 
    [10801462851853052059, 8912476932191465120, 4868635385389206586], 
    [6811037796297371929, 5452186390884417396, 6595524892338886420], 
    [9515779292363133931, 4771700775976261681, 7508435081546226485], 
    [13143799653672417594, 2977295878291973880, 12312946028297191917],
    [1170089553000533213, 6764588673318868279, 11275009673257570151], 
    [14594542294030362533, 18177455830854766254, 1901908595772958652], 
    [1159529803391868207, 3970892217229352740, 14785767409178671674], 
    [10999170248432652892, 14229154700594107303, 8307568924959719677], 
    [1666831744467359211, 15273435638547708406, 4625106060006123695], 
    [2527832789102003890, 2162706509636595181, 12274659024403278944], 
    [11462922661288872929, 705234877256492788, 15285067547267551823],
    [8685523986883344813, 5189402771566872437, 13365870371239193017], 
    [8691484281425062096, 12795630334971816194, 14801425411312727128], 
    [3736235832881329480, 963362520841036787, 2425747491837933480], 
    [1622722818849118758, 14755043213862161363, 13874178899784194194],
    [3632717305838481192, 1214071339724279273, 9751709077147155499],
    [13669617437297662519, 6110060895647374355, 3471925163867334933],
    [16158453995329915075, 16765318684890845859, 8791236189902103374],
    [4634727984228282364, 262259015920141319, 8826640880047261317],
    [1589117574797907833, 953782989870186252, 3153682763919135176],
    [17408595535725358576, 1246916769994843403, 13951244434621740654],
    [14669817827077357355, 15304820503557575621, 7862176287495905405],
    [13826258914807279454, 11209076148279190151, 26448026095026763], 
    [11034525618811536398, 13888113278806122428, 14569381278722617385],
    [12530655029540635044, 6793892545556219748, 18172314098722366665],
    [14298642491616581458, 10954788128023082162, 13727665909253622461],
    [745668369553588023, 7013162420860163092, 11006761376858914789], 
    [10414598511333318548, 17267560392544836476, 15936562580257439990],
    [7006531785942637488, 14560359703138467045, 18445347912108569150],
    [17832013397205793368, 8267460749007099312, 14715435418032827703], 
    [14833722001886878630, 12351490699278550218, 14502109663154771443],
    [1583596717951030208, 17248805708277767730, 2916269679780750626],
    [7358915110805147243, 7780761880636494827, 13012271822774676984], 
    [17822699807347824378, 2276387032319463503, 14412694685856993128], 
    [15379848971541347362, 4782424837802252869, 18257384685568861618],
    [1437437257699243109, 13354908353672560208, 2688163757136230402], 
    [14902729570033028431, 13321119784409744787, 7531814678345366155]
                     
];

/// Randomly Generates new hash data.
pub fn gen_data() {
    let mut rng = rand::thread_rng();

    let new_hash_data: [[u64; 3]; 38] = [[0; 3]; 38].map(|_: [u64; 3]| {
        [rng.gen(), rng.gen(), rng.gen()]

    });
    
    println!("{:?}", new_hash_data);

}

/// Gets the hash for a [`BoardState`] including the player to move.
pub fn get_hash(board: &mut BoardState, player: Player) -> u64 {
    let mut hash = PLAYER_HASH_DATA[player as usize];

    for (i, piece) in board.data.iter().enumerate().take(36) {
        if *piece != Piece::None {
            hash ^= ZOBRIST_HASH_DATA[i][*piece as usize];

        }

    }

    hash

}

/// Gets the hash for a board not including the player to move.
pub fn get_uni_hash(data: [Piece; 38]) -> u64 {
    let mut hash = 0;

    for (i, piece) in data.iter().enumerate().take(36) {
        if *piece != Piece::None {
            hash ^= ZOBRIST_HASH_DATA[i][*piece as usize];

        }

    }

    hash

}