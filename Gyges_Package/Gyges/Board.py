class Board:
    piece_movements = {"1": [[(-1, 0)], [(0, -1)], [(1, 0)], [(0, 1)]],

                       "2": [[(-1, 0), (-1, 0)], [(-1, 0), (0, -1)], [(0, -1), (-1, 0)], [(0, -1), (0, -1)], [(0, -1), (1, 0)], [(1, 0), (0, -1)], [(1, 0), (1, 0)], [(1, 0), (0, 1)], [(0, 1), (1, 0)], [(0, 1), (0, 1)], [(0, 1), (-1, 0)], [(-1, 0), (0, 1)]],

                       "3": [[(-1, 0), (-1, 0), (-1, 0)], [(0, -1), (0, -1), (0, -1)], [(1, 0), (1, 0), (1, 0)], [(0, 1), (0, 1), (0, 1)], [(-1, 0), (-1, 0), (0, -1)], [(-1, 0), (0, -1), (-1, 0)], [(0, -1), (-1, 0), (-1, 0)], [(-1, 0), (0, -1), (0, -1)], [(0, -1), (-1, 0), (0, -1)], [(0, -1), (0, -1), (-1, 0)], [(0, -1), (0, -1), (1, 0)], [(0, -1), (1, 0), (0, -1)], [(1, 0), (0, -1), (0, -1)],
                             [(0, -1), (1, 0), (1, 0)], [(1, 0), (0, -1), (1, 0)], [(1, 0), (1, 0), (0, -1)], [(1, 0), (1, 0), (0, 1)], [(1, 0), (0, 1), (1, 0)], [(0, 1), (1, 0), (1, 0)], [(1, 0), (0, 1), (0, 1)], [(0, 1), (1, 0), (0, 1)], [(0, 1), (0, 1), (1, 0)], [(0, 1), (0, 1), (-1, 0)], [(0, 1), (-1, 0), (0, 1)], [(-1, 0), (0, 1), (0, 1)], [(0, 1), (-1, 0), (-1, 0)],
                             [(-1, 0), (0, 1), (-1, 0)],
                             [(-1, 0), (-1, 0), (0, 1)], [(0, 1), (-1, 0), (0, -1)], [(0, -1), (-1, 0), (0, 1)], [(-1, 0), (0, -1), (1, 0)], [(1, 0), (0, -1), (-1, 0)], [(0, -1), (1, 0), (0, 1)], [(0, 1), (1, 0), (0, -1)], [(1, 0), (0, 1), (-1, 0)], [(-1, 0), (0, 1), (1, 0)]]}

    def __init__(self, board=None):
        if board is None:
            self.board = [[0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          0, 0]
        else:
            self.board = board

        self.completed_moves = []

    def print(self):
        print("")
        if self.board[6] == 0:
            print(f"       .")
        else:
            print(f"       !")
        print("")

        for y in range(6):
            for x in range(6):
                if self.board[y][x] == 0:
                    print(".", end="  ")
                else:
                    print(self.board[y][x], end="  ")
            print("")

        print("")
        if self.board[7] == 0:
            print(f"       .")
        else:
            print(f"       !")
        print("")

    def fancy_print(self):
        printable_board = [[" ", " ", " ", " ", " ", " "],
                           [" ", " ", " ", " ", " ", " "],
                           [" ", " ", " ", " ", " ", " "],
                           [" ", " ", " ", " ", " ", " "],
                           [" ", " ", " ", " ", " ", " "],
                           [" ", " ", " ", " ", " ", " "],
                           " ", " "]

        for y in range(6):
            for x in range(6):
                if self.board[y][x] != 0:
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

    def push(self, move):
        changed_pieces = []

        for step_idx, step in enumerate(move):
            if step[1] == "G1":
                self.board[6] = 1
                changed_pieces.append("G1")
            elif step[1] == "G2":
                self.board[7] = 1
                changed_pieces.append("G2")
            else:
                changed_pieces.append(self.board[step[1][1]][step[1][0]])

                change_x = step[1][0]
                change_y = step[1][1]
                change_piece = step[0]
                self.board[change_y][change_x] = change_piece

        self.completed_moves.append([move, changed_pieces])

    def pop(self):
        pop_data = self.completed_moves.pop()

        move = pop_data[0]
        changed_pieces = pop_data[1]

        for i in range(len(changed_pieces)):
            if changed_pieces[i] == "G1":
                self.board[6] = 0
            elif changed_pieces[i] == "G2":
                self.board[7] = 0
            else:
                change = move[i]
                change_x = change[1][0]
                change_y = change[1][1]

                self.board[change_y][change_x] = changed_pieces[i]

    def active_lines(self):
        player_1_set = False

        player_1_active_line = -1
        player_2_active_line = -1

        for y in range(6):
            for x in range(6):
                if self.board[y][x] != 0:
                    if not player_1_set:
                        player_1_active_line = y
                        player_1_set = True
                    player_2_active_line = y

        return player_1_active_line, player_2_active_line

    def game_over(self):
        game_over = False

        if self.board[6] == 1 or self.board[7] == 1:
            game_over = True

        return game_over

    def evaluate(self):
        if self.board[6] == 1:
            return float("inf")
        elif self.board[7] == 1:
            return float("-inf")

        evaluation = 0

        # ---------- GET DATA ----------
        player_1_moves = self.valid_moves(1)
        # --------------------

        evaluation -= len(player_1_moves)

        return evaluation

    def valid_moves(self, player):
        player_1_active_line, player_2_active_line = self.active_lines()

        if player == 1:
            player_1_drops = []
            for y in range(player_1_active_line, player_2_active_line + 1):
                for x in range(6):
                    if self.board[y][x] == 0:
                        player_1_drops.append((x, y))

            player_1_moves = []
            for x in range(6):
                if self.board[player_1_active_line][x] != 0:
                    moves = self.__get_piece_moves((x, player_1_active_line), (x, player_1_active_line), [], [], 1, player_1_drops)
                    player_1_moves.extend(moves)
            return player_1_moves

        elif player == 2:
            player_2_drops = []
            for y in range(player_2_active_line, player_1_active_line - 1, -1):
                for x in range(6):
                    if self.board[y][x] == 0:
                        player_2_drops.append((x, y))

            player_2_moves = []
            for x in range(6):
                if self.board[player_2_active_line][x] != 0:
                    moves = self.__get_piece_moves((x, player_2_active_line), (x, player_2_active_line), [], [], 2, player_2_drops)
                    player_2_moves.extend(moves)
            return player_2_moves

    def __get_piece_moves(self, current_piece, starting_piece, previous_path, previous_banned_bounces, player, current_player_drops):
        final_moves = []

        piece = self.board[current_piece[1]][current_piece[0]]
        if piece == 1 or piece == 2 or piece == 3:
            for path_index, path in enumerate(self.piece_movements[str(piece)]):
                current_x = current_piece[0]
                current_y = current_piece[1]

                current_path = [(current_x, current_y)]

                for step_index, step in enumerate(path):
                    xy = (current_x, current_y)
                    if xy in previous_path:
                        xy_index = previous_path.index(xy)
                        if previous_path[xy_index - 1] == (current_x + step[0], current_y + step[1]):
                            break

                    current_x += step[0]
                    current_y += step[1]

                    if not 0 <= current_x <= 5:
                        break

                    if step_index == len(path) - 1:
                        if current_y == -1:
                            if player == 2:
                                final_moves.append([[self.board[starting_piece[1]][starting_piece[0]], "G1"], [0, starting_piece]])
                                break
                        if current_y == 6:
                            if player == 1:
                                final_moves.append([[self.board[starting_piece[1]][starting_piece[0]], "G2"], [0, starting_piece]])
                                break

                    if not 0 <= current_y <= 5:
                        break

                    if self.board[current_y][current_x] != 0:
                        if step_index != len(path) - 1:
                            break

                    current_path.append((current_x, current_y))

                    if step_index == len(path) - 1:
                        if self.board[current_y][current_x] != 0:
                            total_path = previous_path + current_path
                            total_banned_bounces = []
                            total_banned_bounces.extend(previous_banned_bounces)

                            if not (current_x, current_y) in total_banned_bounces:
                                total_banned_bounces.append((current_x, current_y))

                                for drop in current_player_drops:
                                    piece_to_drop = self.board[current_y][current_x]
                                    drop_location = drop
                                    replacement_piece = self.board[starting_piece[1]][starting_piece[0]]
                                    replacement_location = (current_x, current_y)

                                    final_moves.append([[piece_to_drop, drop_location], [replacement_piece, replacement_location], [0, starting_piece]])

                                final_moves.extend(self.__get_piece_moves((current_x, current_y), starting_piece, total_path, total_banned_bounces, player, current_player_drops))
                        else:
                            final_moves.append([[self.board[starting_piece[1]][starting_piece[0]], current_path[-1]], [0, starting_piece]])

        return final_moves
