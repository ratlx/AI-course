import mindspore as ms
import numpy as np

def is_safe(board, row, col):
    # 检查当前列是否有皇后
    for i in range(row):
        if board[i, col] == 1:
            return False

    # 检查左上对角线是否有皇后
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i, j] == 1:
            return False

    # 检查右上对角线是否有皇后
    for i, j in zip(range(row, -1, -1), range(col, len(board), 1)):
        if board[i, j] == 1:
            return False

    return True

def solve_n_queens(board, row):
    if row >= len(board):
        return True

    for col in range(len(board)):
        if is_safe(board, row, col):
            board[row, col] = 1
            if solve_n_queens(board, row + 1):
                return True
            board[row, col] = 0

    return False

def print_board(board):
    for row in board:
        print(" ".join("Q" if col == 1 else "." for col in row))

def main():
    n = 8
    board = np.zeros((n, n), dtype=int)
    if solve_n_queens(board, 0):
        print_board(board)
    else:
        print("No solution found")

if __name__ == '__main__':
    main()
