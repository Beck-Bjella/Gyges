#![feature(test)]

mod board;
mod helper;
mod moves;
mod search;
mod consts;
mod ugi;
mod tools;

use crate::ugi::*;

fn main() {
    let mut ugi = Ugi::new();
    ugi.start();

}
