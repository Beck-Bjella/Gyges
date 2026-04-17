extern crate gyges_engine;

use gyges_engine::{search::network::init_network, ugi::*};

fn main() {
    // Init neural network
    if let Err(e) = init_network("C:\\Users\\beckb\\Documents\\GitHub\\GygesRust\\weights\\hce_p3r2_1s_161k_weights.bin") {
        eprintln!("Warning: could not load network: {}", e);
    }

    // UGI
    let mut ugi = Ugi::new();
    ugi.start();
    
}
