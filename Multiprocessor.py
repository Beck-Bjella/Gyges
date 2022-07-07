import time
import Calculations
import multiprocessing


def get_position_score(board, depth, position, process_index, processes_complete, total_processes, process_data, start_time):
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

    # ---------- Positional Score ----------

    positional_score = Calculations.mini_max(depth, float("-inf"), float("inf"), False, board)

    # ---------- Depth of 0 Score -----------

    player_2_moves = Calculations.get_all_moves(board, 2)

    threats = 0
    unique_threats = []
    for position2 in player_2_moves:
        if position2[0][1] == "G1":
            threats += 1

            if position2[-1] not in unique_threats:
                unique_threats.append(position2[-1])

    # ----------------------

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

    process_data.append([positional_score, threats, len(unique_threats), process_index])

    processes_complete.append(process_index)
    print(f"{len(processes_complete)} / {total_processes} completed in {round(time.time() - start_time, 3)} seconds.")


def get_best_move(board, depth):
    start_time = time.time()

    processes = []
    process_data = multiprocessing.Manager().list()
    processes_complete = multiprocessing.Manager().list()

    player_2_moves = Calculations.get_all_moves(board, 2)
    for i in range(len(player_2_moves)):
        position = player_2_moves[i]

        process = multiprocessing.Process(target=get_position_score, args=(board, depth, position, i, processes_complete, len(player_2_moves), process_data, start_time))
        processes.append(process)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    end_time = time.time()
    print(f"Finished depth of {depth} calculation in {round(end_time - start_time, 3)} seconds")

    positional_score_list = []
    threat_list = []
    unique_threat_list = []

    for data in process_data:
        positional_score = data[0]
        threats = data[1]
        unique_threats = data[2]

        move = player_2_moves[data[3]]

        in_list = False
        for item in positional_score_list:
            if item[3] == move:
                in_list = True

        if not in_list:
            positional_score_list.append([positional_score, threats, unique_threats, move])
            threat_list.append([positional_score, threats, unique_threats, move])
            unique_threat_list.append([positional_score, threats, unique_threats, move])

    positional_score_list.sort(key=lambda x: x[0], reverse=True)
    threat_list.sort(key=lambda x: x[1], reverse=True)
    unique_threat_list.sort(key=lambda x: x[2], reverse=True)

    final_data = []
    for positional_score_index, data in enumerate(positional_score_list):

        for threat_index, temp_data in enumerate(threat_list):
            if temp_data == data:
                break

        for unique_threat_index, temp_data in enumerate(unique_threat_list):
            if temp_data == data:
                break

        # positional_score_bias = 35/2676
        # threat_bias = 33/2676
        # unique_threat_bias = 40/2676

        positional_score_bias = 1
        threat_bias = 0
        unique_threat_bias = 0

        if data[0] == float("inf"):
            final_score = float("inf")
        elif data[0] == float("-inf"):
            final_score = float("-inf")
        else:
            final_score = (positional_score_index * positional_score_bias) + (threat_index * threat_bias) + (unique_threat_index * unique_threat_bias)

        final_data.append([final_score, data[3]])

    final_data.sort(key=lambda x: x[0], reverse=False)

    return final_data
