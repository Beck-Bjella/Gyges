# Overview

A powerful Gygès engine.

The Gygès engine is a program designed to play the game of Gygès. Like a chess engine, it utilizes advanced algorithms and strategies to make intelligent moves. The engine communicates using the UGI Protocol, which was specifically developed for this engine. This protocol closely relates to the UCI Protocol used in chess engines, incorporating similar concepts and ideas. Check out the [UGI Protocol](https://github.com/Beck-Bjella/Gyges/blob/main/gyges_engine/UGI-Protocol-Specification.md) for more information.

Please note that the Gygès engine is a standalone application and not intended to be used as a library. If you are looking for a library to integrate Gygès functionality into your projects, you can find it [here](https://github.com/Beck-Bjella/Gyges/tree/main/gyges).

[Documentation](https://docs.rs/gyges_engine) and the [Crates.io](https://crates.io/crates/gyges_engine) page.

# Features

- Iterative Deepening Search
  - Alpha-Beta Pruning
- Move Ordering
- Transposition Table with Zobrist Hashing
- Complex Evaluation Heuristics

# Installation & Usage

The easiest and recommended way to experience the Gygès engine is through the [GygesUI](https://github.com/Beck-Bjella/GygesUI), which provides an intuitive graphical interface. For setup instructions, visit the GygesUI repository.

For advanced users who want to run the engine directly, follow one of the options below:

- **Executable:**

  - Download the latest precompiled executable file (.exe) from the [releases page](https://github.com/Beck-Bjella/Gyges/releases). This is the quickest way to get the engine running without building it yourself.

- **Build from Source:**

  - Clone this repository and build the engine manually. This option is ideal for users who want more control over the build process or need compatibility with non-standard architectures.

  ```bash
  git clone https://github.com/Beck-Bjella/Gyges.git
  cd gyges_engine
  cargo build --release
  ```

Once installed, you can interact with the engine in the following ways:

- **Command Line Interface:**

  - Start the engine directly:

  ```bash
  gyges_engine.exe
  ```

  - Communicate with the engine using the UGI Protocol. See the [UGI Protocol Specification](https://github.com/Beck-Bjella/Gyges/blob/main/gyges_engine/UGI-Protocol-Specification.md) for detailed instructions.

# Contributions

Contributions to the Gygès engine are highly welcome! If you would like to contribute to the project, please open a pull request with your changes. I appreciate any feedback, bug reports, or suggestions for improvements.

# License

The Gygès engine is released under the [GNU General Public License v3.0](https://github.com/Beck-Bjella/Gyges/blob/main/LICENSE). Please review and comply with the license terms when using or distributing the engine.

