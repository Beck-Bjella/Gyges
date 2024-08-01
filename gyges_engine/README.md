# Overview
A powerful Gygès engine.

The Gygès engine is a program designed to play the game of Gygès. Like a chess engine, it utilizes advanced algorithms and strategies to make intelligent moves.

The engine communicates using the UGI Protocol, which was specifically developed for this engine. This protocol closely relates to the UCI Protocol used in chess engines, incorporating similar concepts and ideas. Check out the [UGI Protocol](https://github.com/Beck-Bjella/Gyges/blob/main/gyges_engine/UGI-Protocol-Specification.md) for more information.

Please note that the Gygès engine is a standalone application and not intended to be used as a library. If you are looking for a library to integrate Gygès functionality into your projects, you can find it [here](https://github.com/Beck-Bjella/Gyges/tree/main/gyges).

# Features
 - Iterative Deepening Search
    - Alpha-Beta Pruning
 - Move Ordering
 - Transposition Table with Zobrist Hashing
 - Complex Evaluation Heuristics

# Installation
To get started with the Gygès engine, you have two options for installation:

- **Executable:** The latest release of the Gygès engine includes a precompiled executable file (.exe) that you can download directly. This is the easiest way to get the engine running quickly.

- **Build from Source:** If you prefer to build the engine from source, you can find the source code in this repository. This option gives you more flexibility and control over the build process, mainly if you use a different architecture than x86-64.

# Usage
There are multiple ways to interact with the Gygès engine:

- **Command Line Interface:** You can communicate with the engine directly through the command line using the UGI Protocol. However, this method is not recommended for most users, as it requires manual input and lacks a user-friendly interface.

- **GygesUI:** For a more convenient, user-friendly experience, I recommend using the [GygesUI](https://github.com/Beck-Bjella/GygesUI) 
user interface. GygesUI provides a graphical interface for interacting with the Gygès engine, making it easier to use the engine and explore its features.

# Contributions
Contributions to the Gygès engine are highly welcome! If you would like to contribute to the project, please open a pull request with your changes. I appreciate any feedback, bug reports, or suggestions for improvements.

# License
The Gygès engine is released under the [GNU General Public License v3.0](https://github.com/Beck-Bjella/Gyges/blob/main/LICENSE). Please review and comply with the license terms when using or distributing the engine.
