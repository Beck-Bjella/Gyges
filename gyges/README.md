# Overview
A library for the board game Gygès.

Gygès is an abstract strategy game for two players. The object of Gygès is to move a piece to your opponent's last row. The catch is that no one owns the pieces. You can only move a piece in the row nearest you. Pieces are made up of 1, 2, or 3 rings; this number is also the number of spaces it must move. 
If it lands on another piece, it can move the number of spaces equal to that piece's number of rings. 
It can also displace that piece to any open space. 

Offical rule book: [Rules](https://s3.amazonaws.com/geekdo-files.com/bgg32746?response-content-disposition=inline%3B%20filename%3D%22gyges_rules.pdf%22&response-content-type=application%2Fpdf&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJYFNCT7FKCE4O6TA%2F20231229%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231229T031405Z&X-Amz-SignedHeaders=host&X-Amz-Expires=120&X-Amz-Signature=e7c322bed070e101346483b70e896133e22967568a021c530add36ef698b99d0)

# Features
 - Board representaion 
    - Zobrist Hashing
 - BitBoards
 - Move represenation 
 - Custom lists of moves
 - Lighting quick move generation
 - Transposition Table

# Usage
In the future this crate will be published to [crates.io](https://crates.io/).

For now, you can add this crate to your project by adding the following to your `Cargo.toml` file:
```toml
[dependencies]
gyges = { git = " " }
```

# Examples

Basic boards, move generation, and move making.
```rust
    use gyges::board::*;
    use gyges::moves::movegen::*;
    use gyges::core::*;

    // Create board from a constant
    let mut board = BoardState::from(STARTING_BOARD);

    // Define a player 
    let player = Player::One;

    // Create a RawMoveList
    let mut movelist = unsafe{ valid_moves(&mut board, player) };

    // Extract the moves
    let moves = movelist.moves(&mut board);

    // Make the move
    let new_board = board.make_move(&mv);

    // Display the new board
    println!("{}", new_board);
```

# Acknowledgements
This project and its formating was inspired by the incridible rust chess program [Pleco](https://github.com/pleco-rs/Pleco).

# Contributions 

Contributions welcome! If you'd like to contribute, please open a pull request. Feedback is greatly appreciated, along with reporting issues or suggesting improvements.

# Lisence
