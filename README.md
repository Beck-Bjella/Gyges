# Gyges Engine

## Overview

The object of Gygès is to move a piece to your opponent's last row. The catch is that no one owns the pieces. You can only move a piece in the row nearest you. Pieces are made up of 1, 2, or 3 rings; this number is also the number of spaces it must move. If it lands on another piece, it can move the number of spaces equal to that piece's number of rings. It can also displace that piece to another space.

The Gyges engine is developed in Rust and operates similarly to chess engines. It utilizes a command-line interface called the Universal Gyges Interface (UGI), which is analogous to the Universal Chess Interface (UCI). The engine has no external dependencies and can be used as a standalone application.

## Technical Details

### Algorithmic Choices

The engine employs a variety of algorithms to enhance gameplay. The choice of these algorithms, including unique or optimized approaches, sets Gyges apart from other game engines.

### Game State Representation

Internally, the game state is represented using [describe data structures or representation choices]. This ensures efficient access and manipulation of the game state during gameplay.

### Move Generation and Evaluation

Gyges utilizes advanced move generation techniques to explore possible moves efficiently. The evaluation function assesses the quality of moves, incorporating heuristics to make strategic decisions.

### Search Algorithms

The engine implements [describe search algorithms such as minimax, alpha-beta pruning, etc.] for decision-making. Any optimizations or variations on these algorithms are detailed to showcase the engine's efficiency.

### Heuristics

Sophisticated heuristics are employed to evaluate the strength of a given game position. These heuristics play a crucial role in the engine's ability to make intelligent decisions.

### Concurrency (if applicable)

[If applicable, describe any concurrency or parallel processing techniques used to enhance performance.]

### Testing and Validation

To ensure correctness and reliability, the Gyges engine undergoes rigorous testing. This includes unit tests, integration tests, and comparisons against known strategies to validate its strategic decision-making capabilities.

## How to Use

### Installation

- **Executable:** The most recent release includes a precompiled executable (.exe) file for the Gyges engine, which can be downloaded directly.

- **Build from Source:** For users who prefer building from source, the Rust source code is available. Instructions for building the engine from source are provided in the repository.

### Communication

The Gyges engine communicates with the user through the Universal Gyges Interface (UGI). This interface is documented separately for users who wish to control the engine themselves. Refer to the UGI documentation for details on commands and usage.

## Development

Contributions welcome! If you'd like to contribute, please open a pull request. Feedback is greatly appreciated, along with reporting issues or suggesting improvements.

## License

The Gyges board game engine is licensed under the GNU General Public License v3.0 or later. For details, refer to the [LICENSE](link-to-license-file) file in the repository.

## Support

For support or inquiries, please [create an issue](link-to-issues) on the GitHub repository.


