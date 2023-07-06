#![feature(test)]

mod board;
mod helper;
mod moves;
mod search;
mod tools;
mod consts;
mod ugi;

use crate::ugi::*;

fn main() {
    let mut ugi = Ugi::new();
    ugi.start();

}
