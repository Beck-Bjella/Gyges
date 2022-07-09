import multiprocessing
import time
import Calculations


def get_position_score(board, depth, position, process_index, processes_complete, total_processes, process_data, start_time):
    changed_pieces = []
    for i in range(len(position)):
        change = position[i]

        if change[1] == "G1":
            board[6] = 1
            changed_pieces.append("G1")
        elif change[1] == "G2":
            board[7] = 1
            changed_pieces.append("G2")
        else:
            changed_pieces.append(board[change[1][1]][change[1][0]])

            change_x = change[1][0]
            change_y = change[1][1]
            change_piece = change[0]
            board[change_y][change_x] = change_piece

    score = Calculations.mini_max(depth, float("-inf"), float("inf"), False, board)

    for i in range(len(changed_pieces)):
        if changed_pieces[i] == "G1":
            board[6] = 0
        elif changed_pieces[i] == "G2":
            board[7] = 0
        else:
            change = position[i]
            change_x = change[1][0]
            change_y = change[1][1]

            board[change_y][change_x] = changed_pieces[i]

    process_data.append([score, process_index])

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

    final_data = []
    for data in process_data:
        score = data[0]
        move = player_2_moves[data[1]]

        final_data.append([score, move])

    final_data.sort(key=lambda x: x[0], reverse=True)

    return final_data
