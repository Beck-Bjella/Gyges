mod search;
mod consts;
mod ugi;

use crate::ugi::*;

fn main() {
    let mut ugi = Ugi::new();
    ugi.start();

}
