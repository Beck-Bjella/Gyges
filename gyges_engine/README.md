# Overview

A powerful Gygès engine written in Rust. It scores positions with a neural network, searches with a parallel alpha-beta search, and runs natively or in a web browser through WebAssembly. It communicates over the UGI protocol, a text protocol modeled on the UCI protocol used by chess engines; see the [UGI Protocol Specification](https://github.com/Beck-Bjella/Gyges/blob/main/gyges_engine/UGI-Protocol-Specification.md).

The engine is a standalone program. The game rules and move generation are in the performance-oriented [gyges library](https://github.com/Beck-Bjella/Gyges/tree/main/gyges).

[Documentation](https://docs.rs/gyges_engine) · [crates.io](https://crates.io/crates/gyges_engine)

# Features

- Neural network evaluation, with a hand-crafted evaluation as a fallback
- Parallel search using the Young Brothers Wait Concept (YBWC)
- Iterative deepening with aspiration windows
- Alpha-beta search with principal variation search (PVS)
- Lockless transposition table with Zobrist hashing
- Move ordering by threat detection, mobility, and a history heuristic
- Move generation with BMI2 `pext` lookups into precomputed path lists
- Adjustable playing strength
- Native x86-64 and WebAssembly (`wasm32-wasip1`) builds

<a name="results"></a>
# Benchmarks

Strength is measured in sets: each random starting position is played twice, with each engine moving first once. A set is decisive when one engine wins both games. A split set mostly reflects the starting position, so results are reported as the share of decisive sets won.

| Benchmark | Conditions | Result |
|---|---|---|
| **2.0.0 vs 1.1.0** | 5 seconds per move, 4 threads each | 2.0.0 wins **97%** of decisive sets |
| **Network vs hand-crafted evaluation** | Same 2.0.0 search on both sides, 20k to 100k nodes per move | The neural network wins **97%** of decisive sets, steady across search depths |
| **Parallel search** | Fixed-depth search with 1 to 96 threads on a 96-core server | Up to **50.5x** speedup ([details](#scaling)) |

# How It Works

## Evaluation

### Neural Network

Game positions are evaluated by a small feedforward neural network, specifically an MLP (multi-layer perceptron). Its format is as follows:

```
5,886 sparse binary inputs  ->  1,024 hidden (ReLU)  ->  1 output (tanh)
```

The 5,886 inputs encode positional piece pairings, capturing the complex relationships that determine who is winning in the game. Positions are encoded from the perspective of the player to move, with the opponent's view rotated 180 degrees so one network serves both players. The inputs form three groups:

| Group | Description | Inputs |
|---|---|---|
| Piece on square | each of the 36 squares, for each of the 3 piece types | 108 |
| Same-type pairs | each unordered pair of squares, for each piece type | 3 × 630 |
| Cross-type pairs | each ordered pair of squares, for each pair of different types | 3 × 1,296 |

With only 12 pieces on the board, only 78 inputs are ever active (12 piece-square, 18 same-type pair, and 48 cross-type pair inputs), allowing microsecond-level inference (about 8 µs per evaluation) that makes it practical inside an alpha-beta search. With the weights stored per input, an evaluation is 78 column additions instead of a full matrix multiply.

The weights are compiled into the binary with `include_bytes!`, so the engine needs no external files on any platform. Other weights can be loaded at runtime with the `weightsPath` option.

#### Training

- **Data:** ~28 million board positions (after augmentation: left-right mirror images added, duplicates removed), from games the engine played against itself using the V1.1.0 hand-crafted evaluation, from random home-row arrangements, generated on an AWS EC2 Linux server.
- **Labels:** each position is labeled with its corresponding game's final result, scaled to ±0.75 to stay in the linear range of tanh, allowing the network to learn, from game outcomes alone, which positional patterns lead to winning.
- **Training:** PyTorch with MSE loss, Adam (learning rate 0.001), batch size 4,096, 0.5 dropout, and an 80/20 train/validation split.
- **Selection:** the shipped weights are from epoch 9, at the validation plateau.

### Hand-Crafted Evaluation

The 1.1.0 evaluation is kept as a fallback (`setoption nn false`). It scores each side's mobility plus the pieces and squares that only that side can reach, weighting 1-ring pieces most.

## Search

The engine uses negamax alpha-beta with principal variation search, driven by iterative deepening over odd depths (1, 3, 5, ...) with aspiration windows centered on the previous score. The first five moves at each position are searched with a full window, and the rest with a null window that is widened only if a move scores better than expected.

A lockless, clustered transposition table keyed by Zobrist hashes is shared across all threads. It uses 400 MB by default (about 25 MB in WebAssembly), and its design follows the [Pleco](https://github.com/pleco-rs/Pleco) chess engine. Positions that repeat along the current line are skipped, which prevents the search from looping.

## Move Ordering

Strict move ordering, which prunes losing candidates early, mitigates Gygès' branching factor of hundreds of moves (222 legal moves from the starting position). Each candidate is ranked by:

- **Losing moves:** removed outright if they let the opponent reach the goal on the next move.
- **Threats:** moves that create a winning threat of their own are searched first.
- **Opponent mobility:** moves that leave the opponent fewer legal replies rank higher.
- **History heuristic:** breaks near-ties (scores within 3%), favoring moves that caused cutoffs elsewhere in the search.

## Move Generation

Moves come from the performance-oriented [gyges library](https://github.com/Beck-Bjella/Gyges/tree/main/gyges), which generates a full legal move list in about 1 µs and a move count in about 120 ns. For 2- and 3-ring pieces, the occupancy of the squares that could block a piece's paths is compressed with the BMI2 `pext` instruction into an index into precomputed lists of unblocked paths, one per square and blocking pattern: 365 lists for 2-ring pieces and 8,389 for 3-ring pieces. Bounces are followed with an explicit stack, using bitboards to stop path segments from being reused and pieces from being bounced off twice. Generation is generic over what it collects (moves, move counts, threat counts, or reachable squares) and over when it stops, so the threat check in move ordering exits at the first goal-reaching move.

## Parallel Search

The search is parallelized with the Young Brothers Wait Concept (YBWC). At each position, one thread searches the first move alone to establish a bound. The remaining moves then form a split point that other threads can join. The first move is usually the best, so its bound lets the remaining moves be searched with a much narrower window.

- A main thread runs iterative deepening; helper threads wait in a shared pool for split points.
- Split points open only with at least 2 ply of search remaining, and at most 32 are open at once.
- A thread waiting for its own split point to finish helps with other split points instead of idling.
- When a thread finds a cutoff, the split point is marked, and every thread working on it stops at its next node.

### Scaling

Scaling was measured on a 96-core AWS server (EPYC 9R45, Zen 5, SMT off, one NUMA node), timing a fixed-depth search with 1 to 96 threads on a wide position (`start`, many moves with close scores) and a narrow one (`bench`):

![Thread scaling](../assets/scaling.svg)

On the wide position, the search scales close to linearly to 32 threads and reaches **50.5x** at 96 threads, with almost no extra work. Up to 16 threads it is slightly better than linear, because helper threads find cutoffs earlier than a single thread would.

On the narrow position, it levels off near **13x**. With fewer good moves to distribute, extra threads mostly search lines a single thread would have pruned. Most Gygès middlegames resemble the wide case.

## Playing Strength

The `skill-weakness`, `skill-reach`, and `skill-allowLosing` options lower the engine's strength, for example for beginners. After the search, part of each move's score gap to the best move is forgiven, so slightly worse moves become likely choices while much worse moves stay unlikely. The method follows Stockfish's skill levels.

## Platforms

- **x86-64:** built for the x86-64-v3 level (AVX2, FMA, BMI2). Move generation uses the BMI2 `pext` instruction to look up blocked paths.
- **WebAssembly:** `wasm32-wasip1`. The search runs on one thread, inside the `go` command, and needs no filesystem because the weights are embedded.
- **Other targets:** other architectures, such as aarch64, build with a portable fallback for `pext`. These builds are not tuned or tested.

# Installation & Usage

Play against the engine at [gyges.app](https://gyges.app), or run it directly:

- **Executable:** download a build from the [releases page](https://github.com/Beck-Bjella/Gyges/releases).
- **From source:**

  ```bash
  git clone https://github.com/Beck-Bjella/Gyges.git
  cd Gyges
  cargo build --release -p gyges_engine
  ```

  The binary is written to `target/release/gyges_engine`. For WebAssembly:

  ```bash
  rustup target add wasm32-wasip1
  cargo build --release -p gyges_engine --target wasm32-wasip1
  ```

Start the engine and send UGI commands, for example:

```
setpos start
setoption threads 8
setoption maxTime 5
go
```

All commands and options are listed in the [UGI Protocol Specification](https://github.com/Beck-Bjella/Gyges/blob/main/gyges_engine/UGI-Protocol-Specification.md).

# Contributions

Contributions are welcome. Open a pull request, or open an issue to report a bug or suggest an improvement.

By submitting a contribution, you confirm you have the right to submit it, and you grant Beck Bjella a perpetual, irrevocable license to use, modify, and relicense it under any license, including the project's current licenses.

# License

Copyright © 2022-2026 Beck Bjella. Released under the [GNU Affero General Public License v3.0](https://github.com/Beck-Bjella/Gyges/blob/main/gyges_engine/LICENSE).
