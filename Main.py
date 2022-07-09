import Multiprocessor

board = [[0, 2, 0, 3, 2, 0],
         [0, 0, 0, 0, 0, 1],
         [0, 1, 0, 0, 0, 0],
         [3, 0, 1, 0, 0, 0],
         [0, 2, 0, 0, 0, 0],
         [0, 3, 0, 2, 3, 1],
         0, 0]


# ---------- Broken ----------

def print_board(board, player=1):
    printable_board = [[0, 2, 0, 3, 2, 0],
                       [0, 0, 0, 0, 0, 1],
                       [0, 1, 0, 0, 0, 0],
                       [3, 0, 1, 0, 0, 0],
                       [0, 2, 0, 0, 0, 0],
                       [0, 3, 0, 2, 3, 1],
                       [0, 9, 9, 9, 9, 9],
                       [0, 9, 9, 9, 9, 9]]
    for y in range(6):
        for x in range(6):
            if board[y][x] == "0":
                new_board[y][x] = " "
            else:
                new_board[y][x] = board[y][x]
    if board[6] == "!":
        new_board[6] = "!"
    if board[7] == "!":
        new_board[7] = "!"

    if player == 1:
        print(f"")
        print(f"                              PLAYER 2")
        print(f"                            + ------- +")
        print(f"                            |         |")
        print(f"                            |    {new_board[7]}    |")
        print(f"                            |         |")
        print(f"                            + ------- +")
        print(f"")

        print(f"        5         4         3         2         1         0     ")
        print(f"   + ------- + ------- + ------- + ------- + ------- + ------- +")

        for y in range(5, -1, -1):
            print(f"   |         |         |         |         |         |         |")
            print(f"{y}  |    {printable_board[y][5]}    |    {printable_board[y][4]}    |    {printable_board[y][3]}    |    {printable_board[y][2]}    |    {printable_board[y][1]}    |    {new_board[y][0]}    |  {y}")
            print(f"   |         |         |         |         |         |         |")
            print(f"   + ------- + ------- + ------- + ------- + ------- + ------- +")
        print(f"        5         4         3         2         1         0     ")

        print(f"")
        print(f"                            + ------- +")
        print(f"                            |         |")
        print(f"                            |    {new_board[6]}    |")
        print(f"                            |         |")
        print(f"                            + ------- +")
        print(f"                              PLAYER 1")
        print(f"")
    else:

        print(f"")
        print(f"                              PLAYER 1")
        print(f"                            + ------- +")
        print(f"                            |         |")
        print(f"                            |    {new_board[6]}    |")
        print(f"                            |         |")
        print(f"                            + ------- +")
        print(f"")

        print(f"        0         1         2         3         4         5     ")
        print(f"   + ------- + ------- + ------- + ------- + ------- + ------- +")

        for y in range(6):
            print(f"   |         |         |         |         |         |         |")
            print(f"{y}  |    {new_board[y][0]}    |    {new_board[y][1]}    |    {new_board[y][2]}    |    {new_board[y][3]}    |    {new_board[y][4]}    |    {new_board[y][5]}    |  {y}")
            print(f"   |         |         |         |         |         |         |")
            print(f"   + ------- + ------- + ------- + ------- + ------- + ------- +")
        print(f"        0         1         2         3         4         5     ")

        print(f"")
        print(f"                            + ------- +")
        print(f"                            |         |")
        print(f"                            |    {new_board[7]}    |")
        print(f"                            |         |")
        print(f"                            + ------- +")
        print(f"                              PLAYER 2")
        print(f"")


def player_turn():
    action = input("Bounce or Replace:")

    if action == "B":
        starting_xy_input = input("Starting Cords: ")
        starting_xy = (int(starting_xy_input[0]), int(starting_xy_input[1]))
        starting_piece = current_board[starting_xy[1]][starting_xy[0]]

        final_xy_input = input("Final Cords: ")
        if final_xy_input == "GOAL":
            current_board[7] = "!"
        else:
            final_xy = (int(final_xy_input[0]), int(final_xy_input[1]))
            current_board[final_xy[1]][final_xy[0]] = starting_piece
            current_board[starting_xy[1]][starting_xy[0]] = "0"
    else:
        starting_xy_input = input("Starting Cords: ")
        starting_xy = (int(starting_xy_input[0]), int(starting_xy_input[1]))
        starting_piece = current_board[starting_xy[1]][starting_xy[0]]

        replacement_xy_input = input("Replacement Cords: ")
        replacement_xy = (int(replacement_xy_input[0]), int(replacement_xy_input[1]))
        replacement_piece = current_board[replacement_xy[1]][replacement_xy[0]]

        drop_xy_input = input("Drop Cords: ")
        drop_xy = (int(drop_xy_input[0]), int(drop_xy_input[1]))

        current_board[replacement_xy[1]][replacement_xy[0]] = starting_piece
        current_board[drop_xy[1]][drop_xy[0]] = replacement_piece
        current_board[starting_xy[1]][starting_xy[0]] = "0"


def computer_turn():
    print("Waiting on computer...")
    best_move, best_score = Multiprocessor.get_best_move(current_board, depth=1)
    print(best_move, best_score)

    changed_pieces = []
    for i in range(len(best_move)):
        change = best_move[i]

        if change[1] == "G1":
            current_board[6] = "!"
            changed_pieces.append("G1")
        elif change[1] == "G2":
            current_board[7] = "!"
            changed_pieces.append("G2")
        else:
            changed_pieces.append(current_board[change[1][1]][change[1][0]])

            change_x = change[1][0]
            change_y = change[1][1]
            change_piece = change[0]
            current_board[change_y][change_x] = change_piece


def play_against_computer():
    player_starting = input("Do you want to start: ")
    player_starting_line = input("Enter your starting line: ")

    for x in range(len(player_starting_line)):
        current_board[0][x] = str(player_starting_line[x])

    computer_starting_pieces = ["1", "1", "2", "2", "3", "3"]
    for x in range(6):
        index = random.randrange(len(computer_starting_pieces))
        current_board[5][x] = computer_starting_pieces[index]
        computer_starting_pieces.pop(index)

    if player_starting == "Y":
        while True:
            print_board(current_board)
            player_turn()
            if Calculations.game_over(current_board):
                print_board(current_board)
                print("PLAYER WINS")
                break

            print_board(current_board)
            computer_turn()
            if Calculations.game_over(current_board):
                print_board(current_board)
                print("COMPUTER WINS")
                break
    else:
        while True:
            print_board(current_board)
            computer_turn()
            if Calculations.game_over(current_board):
                print_board(current_board)
                print("COMPUTER WINS")
                break

            print_board(current_board)
            player_turn()
            if Calculations.game_over(current_board):
                print_board(current_board)
                print("PLAYER WINS")
                break


# --------------------

if __name__ == '__main__':
    moves = Multiprocessor.get_best_move(board, depth=1)
    for item_idx, item in enumerate(moves):
        print(f"{item_idx}. {item}")
