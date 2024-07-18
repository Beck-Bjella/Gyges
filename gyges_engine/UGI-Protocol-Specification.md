# Description of the Universal Gyges Interface (UGI) [Jan 2024]

## Move Format:

The moves are formated by numbers seperated with `|`. 
	* e.g. 0|3|2|12|3|8
	* e.g. 0|1|3|7

	Lets look at the first example:

	It translates into:
		* Put the piece '0' at position '3'1


## GUI to Engine:
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
	* `data`

* `go`
	Initiates the engine to start searching.
	Subsequently sends "info ____" commands during the search.
	Responds with "bestmove ____" when the search is complete.

* `stop`
	Instructs the engine to stop calculations as soon as possible.
	Response: "bestmove ____" (providing the best move found)

* `quit`
	Terminates the program.


## Engine to GUI:
* `ugiok`
	Sent after processing the "ugi" command to acknowledge UGI mode.

* `id`
	* `name` (engine name)
	* `author` (author name)

* `readyok`
	Responds with when ready to accept new commands.

* `go`
	When searching, sends "info __" commands.
	Responds with "bestmove __" when the search is complete.

* `info`
	The engine sends information to the GUI.
    Example parameters: ply, score, time, nodes

* `option`
  	Command to inform the GUI about parameters that can be changed in the engine.
	* `maxply`
	* `maxtime`

## Example:
```bash





```

