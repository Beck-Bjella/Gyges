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

# Engine Usage
For the easiest and most intuitive experience, start with the [GygesUI](https://github.com/Beck-Bjella/GygesUI), which provides a graphical interface for running the engine. If you prefer to use the engine directly, check out the [Gyges-engine](https://github.com/Beck-Bjella/Gyges/tree/main/gyges_engine) crate README for more information.

# Basic Library Usage
To add this crate to your project, include the following in your `Cargo.toml` file:

```toml
[dependencies]
gyges = "1.1.0"
```

### Setting up a Starting Board Position

The library provides predefined starting board positions. Use the following code to load the default board:

```rust
use gyges::board::*;

// Load the starting board position
let board: BoardState = BoardState::from(STARTING_BOARD);
```

### Loading a Specific Board Configuration

To load a custom board configuration, provide a string where each row is represented by a series of numbers. Rows are inputted from your side of the board to the opponent's, and pieces are numbered based on their ring count:

```rust
use gyges::board::*;

// Load a custom board configuration
let board: BoardState = BoardState::from("321123/000000/000000/000000/000000/321123");
```

### Generating Moves

The `MoveGen` structure is the core for generating moves in Gygès. It provides a flexible interface for generating and calculating move-related data tailored to specific needs using generic parameters.

#### How It Works

`MoveGen` uses two key generic parameters:

- **`GenType`**: Specifies the type of data to generate:
  - `GenMoves`: Generates all possible legal moves.
  - `GenMoveCount`: Counts the total number of possible moves.
  - `GenThreatCount`: Counts the number of threats on the board.
  - `GenControlMoveCount`: Combines control analysis and move counting.
- **`QuitType`**: Controls when generation stops:
  - `NoQuit`: Completes the full generation process.
  - `QuitOnThreat`: Stops generation immediately if a threat is found. This is particularly useful for saving computation in scenarios where you need both the data and the guarantee that no threats exist. If a threat is found, you can handle it separately.

#### Examples

##### 1. Generate All Moves

```rust
use gyges::board::*;
use gyges::moves::movegen::*;

// Setup
let mut board: BoardState = BoardState::from(STARTING_BOARD);
let player: Player = Player::One;
let mut move_gen: MoveGen = MoveGen::default();

// Generate
let data: GenResult = unsafe { move_gen.gen::<GenMoves, NoQuit>(&mut board, player) };
let moves: Vec<Move> = movelist.moves(&mut board);
println!("Generated moves: {:?}", moves);
```

##### 2. Generate Move Count, Stopping if a Threat is Found

```rust
use gyges::board::*;
use gyges::moves::movegen::*;

// Setup
let mut board: BoardState = BoardState::from(STARTING_BOARD);
let player: Player = Player::One;
let mut move_gen: MoveGen = MoveGen::default();

// Generate
let data: GenResult = move_gen.gen::<GenMoveCount, QuitOnThreat>(&mut board, player);
let move_count: usize = data.move_count;
println!("Move count: {}", move_count);
```

### Making a Move

Use the `make_move` method on the `BoardState` struct to make a move. This method takes a `Move` struct as an argument and returns a new `BoardState` with the move applied:

```rust
use gyges::board::*;
use gyges::moves::movegen::*;

// Setup
let mut board: BoardState = BoardState::from(STARTING_BOARD);
let player: Player = Player::One;

let mut move_gen: MoveGen = MoveGen::default();

// Generate moves
let data: GenResult = unsafe { move_gen.gen::<GenMoves, NoQuit>(&mut board, player) };
let movelist: RawMoveList = data.move_list;

let moves: Vec<Move> = movelist.moves(&mut board);

// Make a move
println!("Original board: {}", board);
println!("Move: {:?}", moves[0]);

let mut new_board: BoardState = board.make_move(&first_move);

println!("New board: {}", board);
```

# Contributions 
Contributions welcome! If you'd like to contribute, please open a pull request. Feedback is greatly appreciated, along with reporting issues or suggesting improvements.

# Lisence
This project is released under the [GNU General Public License v3.0](https://github.com/Beck-Bjella/Gyges/blob/main/LICENSE). Please review and comply with the terms of the license when using or distributing the project.
