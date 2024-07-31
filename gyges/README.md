# Overview
A library for the board game Gygès.

Gygès is an abstract strategy game for two players. The object of Gygès is to move a piece to your opponent's last row. The catch is that no one owns the pieces. You can only move a piece in the row nearest you. Pieces are made up of 1, 2, or 3 rings; this number is also the number of spaces it must move. 
If it lands on another piece, it can move the number of spaces equal to that piece's number of rings. 
It can also displace that piece to any open space. 

Offical rule book: [Rules](https://s3.amazonaws.com/geekdo-files.com/bgg32746?response-content-disposition=inline%3B%20filename%3D%22gyges_rules.pdf%22&response-content-type=application%2Fpdf&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJYFNCT7FKCE4O6TA%2F20231229%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231229T031405Z&X-Amz-SignedHeaders=host&X-Amz-Expires=120&X-Amz-Signature=e7c322bed070e101346483b70e896133e22967568a021c530add36ef698b99d0)

This is specifically a library for the game Gygès not the engine to play the game. You can check out the engine [here]().

# Library Features
 - Board representaion
    - Zobrist Hashing
 - BitBoards
 - Move represenation 
 - Custom lists of moves
 - Lighting quick move generation
 - Transposition Table




# Library Usage
You can add this crate to your project by adding the following to your `Cargo.toml` file:
```toml
[dependencies]
gyges = { git = "https://github.com/Beck-Bjella/Gyges/tree/version-1.0-prep/gyges", branch = "main" }
```

# Examples

### Setting up a starting board position

This is one specific starting position built into the library. There are other constant board positions that can be loaded as well.
```rust 
use gyges::board::*;

// Load Board
let board = BoardState::from(STARTING_BOARD);

```

### Loading a specific Board

Boards can be created using this notation as shown below. Each set of 6 numbers represents a row on the board starting from your side of the board going left to right. The orientation of the board is subjetive and is based on how the board is inputed.
```rust
use gyges::board::*;

// Load Board
let board = BoardState::from("321123/000000/000000/000000/000000/321123");

```

### Applying and generating moves

Move generation is done in a two step process. You first have to generate a `RawMoveList` and then extract the moves from that list into a `Vec<Move>`. This is done to improve efficiency and reduce unnessary processing. When making a move the `make_move` function will return a copy of the board with the move applied.
```rust
use gyges::board::*;
use gyges::moves::*;

// Load Board
let mut board = BoardState::from(STARTING_BOARD);

let player = Player::One;

// Generate moves
let mut movelist = unsafe{ valid_moves(&mut board, player) }; // Create a MoveList
let moves = movelist.moves(&mut board); // Extract the moves

// Make a move
board.make_move(&moves[0]);

```

# Acknowledgements
This project and its formating was inspired by the incridible rust chess program [Pleco](https://github.com/pleco-rs/Pleco).

# Contributions 

Contributions welcome! If you'd like to contribute, please open a pull request. Feedback is greatly appreciated, along with reporting issues or suggesting improvements.

# Lisence
This project is licensed under the GPL-3.0 license. Please make sure to review and comply with the terms of the license when using or distributing the engine.