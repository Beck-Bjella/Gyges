//! 
//! 
//! # Usage
//! This crate is not intended to be used for any functionaly. Run the compiled binary to run the engine. 
//! 
//! 
//! 
//! 

pub mod search;
pub mod consts;
pub mod ugi;

use crate::ugi::*;

fn main() {
    let mut ugi = Ugi::new();
    ugi.start();

}
