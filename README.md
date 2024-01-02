# Overview 
This project was created to explore the game of Gygès.

Gygès is an abstract strategy game for two players. The object of Gygès is to move a piece to your opponent's last row. The catch is that no one owns the pieces. You can only move a piece in the row nearest you. Pieces are made up of 1, 2, or 3 rings; this number is also the number of spaces it must move. If it lands on another piece, it can move the number of spaces equal to that piece's number of rings. 
It can also displace that piece to any open space. 

Offical rule book: [Rules](https://s3.amazonaws.com/geekdo-files.com/bgg32746?response-content-disposition=inline%3B%20filename%3D%22gyges_rules.pdf%22&response-content-type=application%2Fpdf&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJYFNCT7FKCE4O6TA%2F20231229%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231229T031405Z&X-Amz-SignedHeaders=host&X-Amz-Expires=120&X-Amz-Signature=e7c322bed070e101346483b70e896133e22967568a021c530add36ef698b99d0)

This project is made up of two main components.
The:
- [Gyges crate](). 
- [Gyges-engine crate]().

The Gyges crate is a library that provides all core funcinaly for the game. It is intended to be used as a dependency in other projects.

The Gyges-engine crate is a fully functional engine to play the game.

Both of these crates are written in Rust, and found in the `gyges` and `gyges_engine` directories respectively. You can find more information about each crate in their respective READMEs.

# Contributions 

Contributions welcome! If you'd like to contribute, please open a pull request. Feedback is greatly appreciated, along with reporting issues or suggesting improvements.

# Lisence
