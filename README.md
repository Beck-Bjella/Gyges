# Overview 
This project was created to take a deep dive into the world of Gygès.

Gygès is an abstract strategy game for two players. The object of Gygès is to move a piece to your opponent's last row. The catch is that no one owns the pieces. You can only move a piece in the row nearest you. Pieces are made up of 1, 2, or 3 rings; this number is also the number of spaces they must move. If a piece lands on another piece, it can move the number of spaces equal to that piece's number of rings. It can also displace that piece to any open space. 

Check out the game on Board Game Arena: [Gygès](https://boardgamearena.com/gamepanel?game=gyges). 

This project is made up of two main components.
The:
- [Gyges crate](https://github.com/Beck-Bjella/Gyges/tree/main/gyges). 
- [Gyges-engine crate](https://github.com/Beck-Bjella/Gyges/tree/main/gyges_engine).

The Gyges crate is a library that provides all the game's core functionality. It is intended to be used as a dependency in other projects.

The Gyges-engine crate is a fully functional engine for playing the game.

Both of these crates are written in Rust and are found in the `gyges` and `gyges_engine` directories, respectively. The respective READMEs provide more information about each crate and its usage.

# Standalone Usage
To use the standalone engine, check out the [Gyges-engine](https://github.com/Beck-Bjella/Gyges/tree/main/gyges_engine) crate README for more information.

# Library Usage
You can add this crate to your project by adding the following to your `Cargo.toml` file:
```toml
[dependencies]
gyges = "1.0.1"
```

## Examples

### Setting up a starting board position
This is one specific starting position built into the library. Other constant board positions can be loaded as well.
```rust 
use gyges::board::*;

// Load Board
let board = BoardState::from(STARTING_BOARD);

```

### Loading a specific Board
Boards can be created using this notation, as shown below. Each set of 6 numbers represents a row on the board, starting from your side of the board and going left to right. The orientation of the board is subjective and is based on how the board is inputted.
```rust
use gyges::board::*;

// Load Board
let board = BoardState::from("321123/000000/000000/000000/000000/321123");

```

### Applying and generating moves
Move generation is done in a two-step process. You must generate a `RawMoveList` and then extract the moves from that list into a `Vec<Move>`. This is done to improve efficiency and reduce unnecessary processing. When making a move, the `make_move` function will return a copy of the board with the move applied.
```rust
use gyges::board::*;
use gyges::moves::*;

// Load Board
let mut board = BoardState::from(STARTING_BOARD);

// Define a player
let player = Player::One;

// Generate moves
let mut movelist = unsafe{ valid_moves(&mut board, player) }; // Create a MoveList
let moves = movelist.moves(&mut board); // Extract the moves

// Make a move
board.make_move(&moves[0]);

```

# Contributions 
Contributions welcome! If you'd like to contribute, please open a pull request. Feedback is greatly appreciated, along with reporting issues or suggesting improvements.

# Lisence
This project is released under the [GNU General Public License v3.0](https://github.com/Beck-Bjella/Gyges/blob/main/LICENSE). Please review and comply with the terms of the license when using or distributing the project.
