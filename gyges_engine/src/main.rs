extern crate gyges_engine;

use gyges_engine::{search::network::init_network, ugi::*};

fn main() {
    // Init neural network
    if let Err(e) = init_network("C:\\Users\\beckb\\Documents\\GitHub\\GygesRust\\hce_p3r2_2s_56k_weights.bin") {
        eprintln!("Warning: could not load network: {}", e);
    }

    // UGI
    let mut ugi = Ugi::new();
    ugi.start();
    
}
