# Universal Gyges Interface (UGI) Documentation [Oct 2026]

This document describes the Universal Gyges Interface (UGI) for interacting with the Gyges engine. The interface is designed for communication over the command line and facilitates board state management, settings, and engine execution.

---

## Board Format

### Overview:

The board has 36 squares in 6 rows of 6, numbered 0 to 35. Row 0 (positions 0-5) is the home row of the side to move, and row 5 (positions 30-35) is the opponent's home row. The engine always plays as the side to move.

The two goals are positions `36` (behind row 0) and `37` (behind row 5). A move that ends on `37` wins the game.

```
            37
30 31 32 33 34 35
24 25 26 27 28 29
18 19 20 21 22 23
12 13 14 15 16 17
 6  7  8  9 10 11
 0  1  2  3  4  5
            36
```

### Layout:

A board is written as one digit per position, starting at position 0. `0` is an empty position, and `1`, `2`, and `3` are pieces with that many rings. Rows may be separated by a `/`, and a layout can be up to 38 digits long. Digits for positions 36 and 37 are accepted but ignored, and any positions left off the end are empty. A search requires all 12 pieces on the board.

**Example** (the starting layout):

```
321123/000000/000000/000000/000000/321123
```

- Positions 0-5 (row 0) hold `3 2 1 1 2 3`.
- Rows 1-4 are empty.
- Positions 30-35 (row 5) hold `3 2 1 1 2 3`.

---

## Move Format

### Overview:

Moves are specified using numbers separated by a `|`. Each number is a position on the board, as described in [Board Format](#board-format).

### Formats:

1. **Standard Move with Drop**:

   ```
   <start>|<end>|<drop>
   ```

   - `start`: Position of the piece to move.
   - `end`: Position to move the piece to.
   - `drop`: Position to drop the replaced piece.

   **Example**:

   ```
   1|2|9
   ```

   - Move the piece at position 1.
   - Replace the piece at position 2.
   - Drop the replaced piece at position 9.

2. **Standard Move without Drop**:

   ```
   <start>|<end>
   ```

   - `start`: Position of the piece to move.
   - `end`: Position to move the piece to.

   **Example**:

   ```
   1|2
   ```

   - Move the piece at position 1 to position 2.

---

## Communication Protocol

### Communication to Engine

This section describes the commands sent to the engine via the command line, how they work, and their corresponding responses returned by the engine.

#### `ugi`

- **Purpose**: Initiates communication with the engine.
- **Engine Response**: 
  ```
  id name <_>
  id author <_>
  option <_>
  ugiok
  ```

#### `isready`

- **Purpose**: Verifies that the engine is ready for new commands.
- **Engine Response**:
  ```
  readyok
  ```

#### `setoption <name> <value>`

- **Purpose**: Adjusts engine settings.
- **Parameters**:
  - `maxPly <_>`: Integer > 0 (maximum search depth).
  - `maxTime <_>`: Number > 0 (maximum search time in seconds).
  - `maxNodes <_>`: Integer > 0 (maximum nodes searched).
  - `threads <_>`: Integer > 0 (number of search threads, default 1).
  - `nn <_>`: Boolean (use the neural network evaluation, default true).
  - `weightsPath <_>`: File path (load network weights from a file).
  - `skill-weakness <_>`: Integer 0-100 (play below full strength, default 0).
  - `skill-reach <_>`: Integer 0-100 (percent of moves a weakened pick can choose from, default 25).
  - `skill-allowLosing <_>`: Boolean (let a weakened pick choose a losing move, default false).
  - `randomize <_>`: Boolean (play a random non-losing move, default false).
  - Booleans accept `true`/`false`, `1`/`0`, or `on`/`off`.
- **Engine Response**: None, or `Unknown Command: '<command>'` if the value is invalid.

#### `setpos <layout>`

- **Purpose**: Updates the board layout.
- **Parameters**:
  - `start`: Standard starting layout.
  - `bench`: Benchmark testing layout.
  - `test`: Testing layout.
  - `draw`: Drawn layout, for testing.
  - `data <_>`: Custom board layout, written as described in [Board Format](#board-format).
- **Engine Response**: None, or `Unknown Command: '<command>'` if the layout has more than 38 digits or any character other than `0`-`3`.

#### `showpos`

- **Purpose**: Prints the current board.
- **Engine Response**: The board.

#### `eval`

- **Purpose**: Prints a breakdown of the hand-crafted evaluation for the current board.
- **Engine Response**: The breakdown.

#### `go`

- **Purpose**: Begins the engine's move search. If the board does not have exactly 12 pieces, the engine replies `info string invalid board: a search needs exactly 12 pieces` and does not search.
- **Example Engine Response**:
  ```
  info ply <depth> bestmove <move> score <score> nodes <count> nps <rate> time <elapsed-time> pv <principal-variation>
  bestmove <move> score <score> time <elapsed-time>
  ```

#### `stop`

- **Purpose**: Stops the engine's search.
- **Engine Response** (only if a search is running):
  ```
  bestmove <move> score <score> time <elapsed-time>
  ```

#### `quit`

- **Purpose**: Exits the program. The engine also exits if its input is closed.
- **Engine Response**: None.

---

### Communication from Engine

This section details the messages the engine sends back via the command line, explaining the information provided in each response.

#### `ugiok`

- **Purpose**: Confirms the engine is in UGI mode.
- **Example**:
  ```
  ugiok
  ```

#### `id <_>`

- **Purpose**: Provides information about the engine.
- **Details**:
  - `name`: Engine name.
  - `author`: Author name.
- **Example**:
  ```
  id name Helios
  id author Beck-Bjella
  ```

#### `option <_>`

- **Purpose**: Describes configurable options for the engine.
- **Details**:
  - `maxPly`, `maxTime`, `maxNodes`, `randomize`, `skill-weakness`, `skill-reach`, `skill-allowLosing`, `nn`, `weightsPath`, `threads`
- **Example**:
  ```
  option maxPly
  option maxTime
  ```

#### `readyok`

- **Purpose**: Indicates the engine is ready for the next command.
- **Example**:
  ```
  readyok
  ```

#### `info string <_>`

- **Purpose**: A message for a human, such as which evaluation is being used.
- **Example**:
  ```
  info string NN evaluation using default.bin (embedded) enabled
  ```

#### `info <_>`

- **Purpose**: Provides information during a search.
- **Details**:
  - `ply`: Search depth.
  - `bestmove`: Best move found.
  - `score`: Evaluation of the best move.
  - `nodes`: Number of nodes searched.
  - `nps`: Nodes per second.
  - `time`: Time taken to search.
  - `pv`: The principal variation.

- **Example**:
  ```
  info ply 1 bestmove 0|1|9 score 1061.000 nodes 225 nps 234619 time 0.001 pv 0|1|9
  ```

#### `bestmove <_>`

- **Purpose**: Sends the best move after a completed search.
- **Details**:
  - `move`: The best move found, or `null` if the position is a draw.
  - `score`: Evaluation of the best move.
  - `time`: Time taken to search.
- **Example**:
  ```
  bestmove 0|1|14 score 3493.000 time 0.950
  ```

---

## Example Communication Session

### Scenario:

A GUI initializes the engine, sets a custom board layout, and requests a move. A '*' indicates the engine's response.

```bash
* Gyges UGI Engine v2.0.0

ugi
* id name Helios
* id author Beck-Bjella
* option maxPly
* option maxTime
* option maxNodes
* option randomize
* option skill-weakness
* option skill-reach
* option skill-allowLosing
* option nn
* option weightsPath
* option threads
* ugiok

setpos data 321123/000000/000000/000000/000000/321123

setoption maxPly 3

isready
* readyok

go
* info string NN evaluation using default.bin (embedded) enabled
* info ply 1 bestmove 0|1|9 score 1061.000 nodes 225 nps 234619 time 0.001 pv 0|1|9
* info ply 3 bestmove 0|1|14 score 3493.000 nodes 68861 nps 72432 time 0.950 pv 0|1|14 5|12|9
* bestmove 0|1|14 score 3493.000 time 0.950

quit
```

---
