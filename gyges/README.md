# Overview

A library for the board game **Gygès**.

Gygès is an abstract strategy game for two players. The object of Gygès is to move a piece to your opponent's last row. The catch is that no one owns the pieces. You can only move a piece in the row nearest you. Pieces are made up of 1, 2, or 3 rings; this number is also the number of spaces they must move. If a piece lands on another piece, it can move the number of spaces equal to that piece's number of rings. It can also displace that piece to any open space.

Check out the game on Board Game Arena: [Gygès](https://boardgamearena.com/gamepanel?game=gyges).

This library is specifically for implementing **Gygès**, not a tool for playing the game directly. To see an implemented game engine using this library, visit the [gyges_engine](https://github.com/Beck-Bjella/Gyges/tree/main/gyges_engine) repository.

- [Documentation](https://docs.rs/gyges)
- [Crates.io](https://crates.io/crates/gyges)

# Library Features
- **Board representation**:
  - Efficient game board representation with Zobrist hashing for state tracking.

- **BitBoards**:
  - Fast board analysis using optimized operations and precomputed masks.

- **Move generation**:
  - Generate and calculate move-related data.
  - Optimized for speed with flexible generic configurations.

- **Move representation**:
  - Structures for representing and processing moves.
  
- **Custom move lists**:
  - Create and manage tailored lists of moves.

- **Core components**:
  - **Pieces**: Represent game pieces.
  - **Squares**: Define board locations.
  - **Player**: Track players and turns.

- **Transposition table**:
  - Efficient caching of previously computed board states.

# Platforms

This library will only run on an x86_64 architecture. It is not currently compatible with other architectures.

# Basic Library Usage

To add this crate to your project, include the following in your `Cargo.toml` file:

```toml
[dependencies]
gyges = "1.1.0"
```

### Setting up a Starting Board Position

The library provides predefined starting board positions. Use the following code to load the default board:

```rust
use gyges::*;

// Load the starting board position
let board: BoardState = BoardState::from(STARTING_BOARD);
```

### Loading a Specific Board Configuration

To load a custom board configuration, provide a string where each row is represented by a series of numbers. Rows are inputted from your side of the board to the opponent's, and pieces are numbered based on their ring count:

```rust
use gyges::*;

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
use gyges::*;

// Setup
let mut board: BoardState = BoardState::from(STARTING_BOARD);
let player: Player = Player::One;
let mut move_gen: MoveGen = MoveGen::default();

// Generate
let data: GenResult = unsafe { move_gen.gen::<GenMoves, NoQuit>(&mut board, player) };
let mut movelist: RawMoveList = data.move_list;

let moves: Vec<Move> = movelist.moves(&mut board);
println!("Generated moves: {:?}", moves);
```

##### 2. Generate Move Count, Stopping if a Threat is Found

```rust
use gyges::*;

// Setup
let mut board: BoardState = BoardState::from(STARTING_BOARD);
let player: Player = Player::One;
let mut move_gen: MoveGen = MoveGen::default();

// Generate
let data: GenResult = unsafe { move_gen.gen::<GenMoveCount, QuitOnThreat>(&mut board, player) };
let move_count: usize = data.move_count;
println!("Move count: {}", move_count);
```

### Making a Move

Use the `make_move` method on the `BoardState` struct to make a move. This method takes a `Move` struct as an argument and returns a new `BoardState` with the move applied:

```rust
use gyges::*;
 
// Setup
let mut board: BoardState = BoardState::from(STARTING_BOARD);
let player: Player = Player::One;

let mut move_gen: MoveGen = MoveGen::default();

// Generate moves
let data: GenResult = unsafe { move_gen.gen::<GenMoves, NoQuit>(&mut board, player) };
let mut movelist: RawMoveList = data.move_list;

let moves: Vec<Move> = movelist.moves(&mut board);

// Make a move
println!("Original board: {}", board);
println!("Move: {:?}", moves[0]);

let mut new_board: BoardState = board.make_move(&moves[0]);

println!("New board: {}", board);
```

# Acknowledgements

This project and its formatting were inspired by the incredible Rust chess program [Pleco](https://github.com/pleco-rs/Pleco).

# Contributions

Contributions are welcome! If you'd like to contribute:

1. Open a pull request.
2. Provide detailed feedback or suggestions.
3. Report issues or propose new features.

# License

This project is released under the [GNU General Public License v3.0](https://github.com/Beck-Bjella/Gyges/blob/main/LICENSE). Please review and comply with the terms of the license when using or distributing the project.
