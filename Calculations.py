import cython

piece_moves = []
piece_movements = {"1": [[(-1, 0)], [(0, -1)], [(1, 0)], [(0, 1)]],

                   "2": [[(-1, 0), (-1, 0)], [(-1, 0), (0, -1)], [(0, -1), (-1, 0)], [(0, -1), (0, -1)], [(0, -1), (1, 0)], [(1, 0), (0, -1)], [(1, 0), (1, 0)], [(1, 0), (0, 1)], [(0, 1), (1, 0)], [(0, 1), (0, 1)], [(0, 1), (-1, 0)], [(-1, 0), (0, 1)]],

                   "3": [[(-1, 0), (-1, 0), (-1, 0)], [(0, -1), (0, -1), (0, -1)], [(1, 0), (1, 0), (1, 0)], [(0, 1), (0, 1), (0, 1)], [(-1, 0), (-1, 0), (0, -1)], [(-1, 0), (0, -1), (-1, 0)], [(0, -1), (-1, 0), (-1, 0)], [(-1, 0), (0, -1), (0, -1)], [(0, -1), (-1, 0), (0, -1)], [(0, -1), (0, -1), (-1, 0)], [(0, -1), (0, -1), (1, 0)], [(0, -1), (1, 0), (0, -1)], [(1, 0), (0, -1), (0, -1)],
                         [(0, -1), (1, 0), (1, 0)], [(1, 0), (0, -1), (1, 0)], [(1, 0), (1, 0), (0, -1)], [(1, 0), (1, 0), (0, 1)], [(1, 0), (0, 1), (1, 0)], [(0, 1), (1, 0), (1, 0)], [(1, 0), (0, 1), (0, 1)], [(0, 1), (1, 0), (0, 1)], [(0, 1), (0, 1), (1, 0)], [(0, 1), (0, 1), (-1, 0)], [(0, 1), (-1, 0), (0, 1)], [(-1, 0), (0, 1), (0, 1)], [(0, 1), (-1, 0), (-1, 0)], [(-1, 0), (0, 1), (-1, 0)],
                         [(-1, 0), (-1, 0), (0, 1)], [(0, 1), (-1, 0), (0, -1)], [(0, -1), (-1, 0), (0, 1)], [(-1, 0), (0, -1), (1, 0)], [(1, 0), (0, -1), (-1, 0)], [(0, -1), (1, 0), (0, 1)], [(0, 1), (1, 0), (0, -1)], [(1, 0), (0, 1), (-1, 0)], [(-1, 0), (0, 1), (1, 0)]]}
piece_heat_maps = {"1": [["0", "0", "0", "0", "0", "0"],
                         ["0", "0", "0", "0", "0", "0"],
                         ["0", "0", "1", "1", "0", "0"],
                         ["0", "1", "2", "2", "1", "0"],
                         ["1", "2", "3", "3", "2", "1"],
                         ["0", "1", "2", "2", "1", "0"]],

                   "2": [["0", "0", "1", "1", "0", "0"],
                         ["0", "1", "2", "2", "1", "0"],
                         ["1", "2", "3", "3", "2", "1"],
                         ["0", "1", "2", "2", "1", "0"],
                         ["0", "0", "1", "1", "0", "0"],
                         ["0", "0", "0", "0", "0", "0"]],

                   "3": [["0", "0", "0", "0", "0", "0"],
                         ["0", "0", "1", "1", "0", "0"],
                         ["0", "1", "2", "2", "1", "0"],
                         ["1", "2", "3", "3", "2", "1"],
                         ["0", "1", "2", "2", "1", "0"],
                         ["0", "0", "1", "1", "0", "0"]]}


@cython.cfunc
def get_piece_moves(current_piece, starting_piece, previous_path, previous_banned_bounces, player, current_player_drops, board):
    piece = board[current_piece[1]][current_piece[0]]
    try:
        for path_index, path in enumerate(piece_movements[piece]):
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

                if 0 <= current_x <= 5:
                    if ((len(path) - 1) - step_index) == 0:
                        if current_y == -1:
                            if player == 2:
                                piece_moves.append([[board[starting_piece[1]][starting_piece[0]], "G1"], ["0", starting_piece]])
                                break
                        if current_y == 6:
                            if player == 1:
                                piece_moves.append([[board[starting_piece[1]][starting_piece[0]], "G2"], ["0", starting_piece]])
                                break

                if not 0 <= current_x <= 5 or not 0 <= current_y <= 5:
                    break

                if board[current_y][current_x] != "0":
                    if step_index != len(path) - 1:
                        break

                current_path.append((current_x, current_y))

                if step_index == len(path) - 1:
                    if board[current_y][current_x] != "0":
                        total_path = previous_path + current_path
                        total_banned_bounces = previous_banned_bounces

                        if not (current_x, current_y) in total_banned_bounces:
                            total_banned_bounces.append((current_x, current_y))

                            for drop in current_player_drops:
                                piece_to_drop = board[current_y][current_x]
                                drop_location = drop
                                replacement_piece = board[starting_piece[1]][starting_piece[0]]
                                replacement_location = (current_x, current_y)

                                piece_moves.append([[piece_to_drop, drop_location], [replacement_piece, replacement_location], ["0", starting_piece]])

                            get_piece_moves((current_x, current_y), starting_piece, total_path, total_banned_bounces, player, current_player_drops, board)
                    else:
                        piece_moves.append([[board[starting_piece[1]][starting_piece[0]], current_path[-1]], ["0", starting_piece]])
    except KeyError:
        pass


def get_all_moves(board, player):
    player_1_active_line, player_2_active_line = get_active_lines(board)

    if player == 1:
        player_1_drops = get_all_drops(board, 1)

        player_1_moves = []
        for x in range(6):
            if board[player_1_active_line][x] != "0":
                get_piece_moves((x, player_1_active_line), (x, player_1_active_line), [], [(x, player_2_active_line)], 1, player_1_drops, board)
                player_1_moves.extend(piece_moves)
                piece_moves.clear()
        return player_1_moves

    if player == 2:
        player_2_drops = get_all_drops(board, 2)

        player_2_moves = []
        for x in range(6):
            if board[player_2_active_line][x] != "0":
                get_piece_moves((x, player_2_active_line), (x, player_2_active_line), [], [(x, player_2_active_line)], 2, player_2_drops, board)
                player_2_moves.extend(piece_moves)
                piece_moves.clear()
        return player_2_moves


@cython.cfunc
def get_all_drops(board, player):
    player_1_active_line, player_2_active_line = get_active_lines(board)

    if player == 1:
        player_1_drops = []
        for y in range(player_1_active_line, player_2_active_line + 1):
            for x in range(6):
                if board[y][x] == "0":
                    player_1_drops.append((x, y))

        return player_1_drops

    if player == 2:
        player_2_drops = []
        for y in range(player_2_active_line, player_1_active_line - 1, -1):
            for x in range(6):
                if board[y][x] == "0":
                    player_2_drops.append((x, y))

        return player_2_drops


@cython.cfunc
def get_active_lines(board):
    player_1_set = False

    player_1_active_line = -1
    player_2_active_line = -1

    for y in range(6):
        for x in range(6):
            if board[y][x] != "0":
                if not player_1_set:
                    player_1_active_line = y
                    player_1_set = True
                player_2_active_line = y

    return player_1_active_line, player_2_active_line


def game_over(board):
    game_over = False

    if board[6] == "!" or board[7] == "!":
        game_over = True

    return game_over


@cython.cfunc
def static_evaluation(board):
    evaluation = 0

    # ---------- Wins/Losses ----------

    if board[6] == "!":
        evaluation += 1000000
    elif board[7] == "!":
        evaluation -= 1000000

    # ---------- Move Counts ----------

    player_1_moves = get_all_moves(board, 1)
    player_2_moves = get_all_moves(board, 2)
    evaluation += len(player_2_moves) - len(player_1_moves)

    # ---------- Heat Maps ----------

    for x in range(6):
        for y in range(6):
            if board[y][x] != "0":
                heat_map = piece_heat_maps[board[y][x]]
                evaluation += (int(heat_map[y][x]) * 100)

    # --------------------

    return evaluation


def mini_max(depth, is_maximizing, board):
    if depth == 0 or game_over(board):
        return static_evaluation(board)

    if is_maximizing:
        current_moves = get_all_moves(board, 2)

        max_eval = float('-inf')
        for position in current_moves:
            changed_pieces = []

            for i in range(len(position)):
                change = position[i]

                if change[1] == "G1":
                    board[6] = "!"
                    changed_pieces.append("G1")
                elif change[1] == "G2":
                    board[7] = "!"
                    changed_pieces.append("G2")
                else:
                    changed_pieces.append(board[change[1][1]][change[1][0]])

                    change_x = change[1][0]
                    change_y = change[1][1]
                    change_piece = change[0]
                    board[change_y][change_x] = change_piece

            score = mini_max(depth - 1, False, board)
            max_eval = max(score, max_eval)

            for i in range(len(changed_pieces)):
                if changed_pieces[i] == "G1":
                    board[6] = "0"
                elif changed_pieces[i] == "G2":
                    board[7] = "0"
                else:
                    change = position[i]
                    change_x = change[1][0]
                    change_y = change[1][1]

                    board[change_y][change_x] = changed_pieces[i]

        return max_eval

    else:
        current_moves = get_all_moves(board, 1)

        min_eval = float('inf')
        for position in current_moves:
            changed_pieces = []

            for i in range(len(position)):
                change = position[i]

                if change[1] == "G1":
                    board[6] = "!"
                    changed_pieces.append("G1")
                elif change[1] == "G2":
                    board[7] = "!"
                    changed_pieces.append("G2")
                else:
                    changed_pieces.append(board[change[1][1]][change[1][0]])

                    change_x = change[1][0]
                    change_y = change[1][1]
                    change_piece = change[0]
                    board[change_y][change_x] = change_piece

            score = mini_max(depth - 1, True, board)
            min_eval = min(score, min_eval)

            for i in range(len(changed_pieces)):
                if changed_pieces[i] == "G1":
                    board[6] = "0"
                elif changed_pieces[i] == "G2":
                    board[7] = "0"
                else:
                    change = position[i]
                    change_x = change[1][0]
                    change_y = change[1][1]

                    board[change_y][change_x] = changed_pieces[i]

        return min_eval
