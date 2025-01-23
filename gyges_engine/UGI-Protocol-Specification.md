# Universal Gyges Interface (UGI) Documentation [Jan 2025]

This document describes the Universal Gyges Interface (UGI) for interacting with the Gyges engine. The interface is designed for communication over the command line and facilitates board state management, settings, and engine execution.

---

## Move Format

### Overview:

Moves are specified using numbers separated by a `|`. Each number represents a position on the board, as defined in the `BoardState` documentation.

### Formats:

1. **Standard Move with Drop**:

   ```
   <start>|<end>|<drop>
   ```

   - : Position of the piece to move.
   - : Position to move the piece to.
   - : Position to drop the replaced piece.

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

   - : Position of the piece to move.
   - : Position to move the piece to.

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
  - `MaxPly <_>`: Integer > 0 (maximum search depth).
  - `MaxTime <_>`: Integer > 0 (maximum search time in seconds).
- **Engine Response**: None.

#### `setpos <layout>`

- **Purpose**: Updates the board layout.
- **Parameters**:
  - `start`: Standard starting layout.
  - `bench`: Benchmark testing layout.
  - `test`: Testing layout.
  - `data <_>`: Custom board layout.
- **Engine Response**: None.

#### `go`

- **Purpose**: Begins the engine's move search.
- **Example Engine Response**:
  ```
  info ply <depth> bestmove <move> score <score> nodes <count> nps <rate> time <elapsed-time> pv <principal-variation>
  bestmove <move>
  ```

#### `stop`

- **Purpose**: Stops the engine's search.
- **Engine Response**:
  ```
  bestmove <move>
  ```

#### `quit`

- **Purpose**: Exits the program.
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
  - `maxPly`
  - `maxTime`
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
  info ply 1 bestmove 0|1|9 score 1061 nodes 225 nps 234619 time 0.000959 pv 0|1|9
  ```

#### `bestmove <_>`

- **Purpose**: Sends the best move after a completed search.
- **Details**:
  - `move`: The best move found.
- **Example**:
  ```
  bestmove 0|1|14
  ```

---

## Example Communication Session

### Scenario:

A GUI initializes the engine, sets a custom board layout, and requests a move. A '*' indicates the engine's response.

```bash
* Gyges UGI Engine v1.1.0

ugi
* id name Helios
* id author Beck-Bjella
* option maxPly
* option maxTime
* ugiok

setpos data 321123/000000/000000/000000/000000/321123

setoption maxPly 7

isready
* readyok

go
* info ply 1 bestmove 0|1|9 score 1061 nodes 225 nps 234619 time 0.000959 pv 0|1|9
* info ply 3 bestmove 0|1|14 score 3493 nodes 68861 nps 72432 time 0.950 pv 0|1|14 5|12|9
* bestmove 0|1|14

stop
* bestmove 0|1|14

quit
```

---
