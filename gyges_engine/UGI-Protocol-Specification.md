# Description of the Universal Gyges Interface (UGI) [Jan 2024]

## Move Format:

Moves are formated by numbers seperated with a `|`. Every number represents a position on the board. You can find the board layout in the `BoardState` documentation.
The format is as follows:
 - The first number is the position of the piece you start with.
 - The second number is the position of the piece you want to replace.
 - The third number is the position you want to drop the replaced piece.

```
<start>|<end>|<drop>
---------------------
   1   |  2  |  9
```
	- Grab the piece at position 1 on the board.
	- Replace the piece at position 2 on the board with it.
	- Drop the replaced piece at position 9.


If a move doesnt require a drop, the format is as follows:
``` 
<start>|<end>
-------------
   1   |  2
```
	- Grab the piece at position 1 on the board.
	- Move it to position 2 on the board.



## Communation Protocol:
All communication takes place over the command line.

### Communation to Engine:
* `ugi`
	Initiates communication with the engine.

* `isready`
	Synchronizes the engine with the GUI, indicating readiness.

* `setoption <name> <value>`
	Used to modify internal parameters.
	* `MaxPly`: int > 0 
	* `MaxTime`: int > 0 (time in seconds)

* `setpos <layout>`
	Sets the board layout; no response is sent.
	* `bench`
	* `start`
	* `data <_>` followed by the board layout

* `go`
	Initiates the engine to start searching.
	Subsequently sends "info <_>" commands during the search.
	Responds with "bestmove <_>" when the search is complete.

* `stop`
	Instructs the engine to stop calculations as soon as possible.
	Response: "bestmove <_>"

* `quit`
	Terminates the program.


### Communation from Engine:
* `ugiok` 
	Sent after processing the `ugi` command to acknowledge UGI mode.

* `id <_>` 
	Information about the engine after the `ugi` command.
	* `name` (engine name)
	* `author` (author name)

* `option <_>`
	Informs the GUI about parameters that can be changed in the engine sent after the `ugi` command.
	* `maxPly`
	* `maxTime` 
	* `ttEnabled`

* `readyok`
	Responds with when ready to accept new commands after the `isready` command.
 
* `info <_>` 
	The engine sends information to the GUI during a search.
    * `ply` (depth of search)
	* `bestmove` (best move found)
	* `score` (evaluation of best move)
	* `nodes` (number of nodes searched)
	* `nps` (nodes per second)
	* `abf` (average branching factor)
	* `time` (time taken to search)

* `bestmove <_>`
	Sent after a completed search to indicate the best move found.
	* `move` (best move found)

## Example:
Communication session between GUI and the engine.

'*' indicates the engine's communication.

```bash
* Gyges UGI Engine v1.0.0

ugi
* id name Helios
* id author beck-bjella
* option maxply
* option maxtime
* option tt_enabled
* ugiok

setpos data 321123/000000/000000/000000/000000/321123 

setoption maxPly 7

isready
* readyok

go
* info ply 1 bestmove 0|1|9 score 1061 nodes 225 nps 234619 abf 225 time 0.000959
* info ply 3 bestmove 0|1|14 score 3493 nodes 68861 nps 72432 abf 40.988 time 0.950

stop
* info ply 3 bestmove 0|1|14 score 3493 nodes 68861 nps 72432 abf 40.988 time 0.950
*bestmove 0|1|14

quit
```

