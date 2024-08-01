//! An incredibly strong Gygès engine.
//!
//! This crate is not intended to be used as a dependency. Run the compiled binary to run the engine. 
//! More information about the engine can be found on the [github page](https://github.com/Beck-Bjella/Gyges).
//! 
//! If you are looking for the core library, check out the gyges crate on [crates.io](https://crates.io/crates/gyges).
//! 

pub mod search;
pub mod consts;
pub mod ugi;

use crate::ugi::*;

fn main() {
    let mut ugi = Ugi::new();
    ugi.start();

}
