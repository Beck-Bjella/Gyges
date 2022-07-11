# fancy board print

printable_board = [[0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               0, 0]

            for y in range(6):
                for x in range(6):
                    if self.board[y][x] == 0:
                        printable_board[y][x] = " "
                    else:
                        printable_board[y][x] = self.board[y][x]

            if self.board[6] == 1:
                printable_board[6] = "!"
            if self.board[7] == 1:
                printable_board[7] = "!"

            print(f"")
            print(f"                              PLAYER 1")
            print(f"                            + ------- +")
            print(f"                            |         |")
            print(f"                            |    {printable_board[6]}    |")
            print(f"                            |         |")
            print(f"                            + ------- +")
            print(f"")

            print(f"        0         1         2         3         4         5     ")
            print(f"   + ------- + ------- + ------- + ------- + ------- + ------- +")

            for y in range(6):
                print(f"   |         |         |         |         |         |         |")
                print(f"{y}  |    {printable_board[y][0]}    |    {printable_board[y][1]}    |    {printable_board[y][2]}    |    {printable_board[y][3]}    |    {printable_board[y][4]}    |    {printable_board[y][5]}    |  {y}")
                print(f"   |         |         |         |         |         |         |")
                print(f"   + ------- + ------- + ------- + ------- + ------- + ------- +")
            print(f"        0         1         2         3         4         5     ")

            print(f"")
            print(f"                            + ------- +")
            print(f"                            |         |")
            print(f"                            |    {printable_board[7]}    |")
            print(f"                            |         |")
            print(f"                            + ------- +")
            print(f"                              PLAYER 2")
            print(f"")