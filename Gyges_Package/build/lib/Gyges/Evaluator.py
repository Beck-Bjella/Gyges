import time
import multiprocessing


def remove_dupes(seq):
    return [x for y, x in enumerate(seq) if x not in seq[:y]]


def mini_max(depth, alpha, beta, is_maximizing, board):
    if depth == 0 or board.game_over():
        return board.evaluate()

    if is_maximizing:
        current_moves = remove_dupes(board.valid_moves(2))

        max_eval = float('-inf')
        for move_idx, move in enumerate(current_moves):
            board.push(move)

            cur_eval = mini_max(depth - 1, alpha, beta, False, board)
            max_eval = max(cur_eval, max_eval)

            board.pop()

            alpha = max(alpha, cur_eval)
            if alpha >= beta:
                break

        return max_eval

    else:
        current_moves = remove_dupes(board.valid_moves(1))

        min_eval = float('inf')
        for move_idx, move in enumerate(current_moves):
            board.push(move)

            cur_eval = mini_max(depth - 1, alpha, beta, True, board)
            min_eval = min(cur_eval, min_eval)

            board.pop()

            beta = min(beta, cur_eval)
            if beta <= alpha:
                break

        return min_eval


def get_move_score(board, depth, move, process_index, processes_complete, total_processes, process_data, start_time):
    board.push(move)

    score = mini_max(depth, float("-inf"), float("inf"), False, board)

    board.pop()

    process_data.append([score, process_index])
    processes_complete.append(process_index)

    print(f"{len(processes_complete)} / {total_processes} completed in {round(time.perf_counter() - start_time, 3)} seconds.")


def best_move(board, depth):
    start_time = time.perf_counter()

    processes = []
    process_data = multiprocessing.Manager().list()
    processes_complete = multiprocessing.Manager().list()

    player_2_moves = remove_dupes(board.valid_moves(2))
    for move_idx, move in enumerate(player_2_moves):
        process = multiprocessing.Process(target=get_move_score, args=(board, depth, move, move_idx, processes_complete, len(player_2_moves), process_data, start_time))
        processes.append(process)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    end_time = time.perf_counter()
    print(f"Finished depth of {depth} calculation in {round(end_time - start_time, 3)} seconds")

    final_data = []
    for data in process_data:
        score = data[0]
        move = player_2_moves[data[1]]

        final_data.append([score, move])

    final_data.sort(key=lambda x: x[0], reverse=True)

    return final_data
