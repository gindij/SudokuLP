import cvxpy as cp
import numpy as np
import sys

def parse_board(path):
    with open(path, "r") as board_file:
        return [
            [int(x) if x != "." else -1 for x in row[:-1]]
            for row in board_file.readlines()
        ]

def to_idx(r, c, i):
    return r * (D ** 2) + c * D + i

def get_row_constr(i, v):
    constr = [0] * N
    for j in range(D):
        idx = to_idx(i, j, v)
        constr[idx] = 1
    return constr

def get_col_constr(j, v):
    constr = [0] * N
    for i in range(D):
        idx = to_idx(i, j, v)
        constr[idx] = 1
    return constr

def get_box_constr(i, j, v):
    srow, scol = i * 3, j * 3
    constr = [0] * N
    for di in range(3):
        for dj in range(3):
            i = srow + di
            j = scol + dj
            idx = to_idx(i, j, v)
            constr[idx] = 1
    return constr


if __name__ == "__main__":
    filename = sys.argv[1]
    board = parse_board(filename)

    D = len(board)
    N = D ** 3

    print("ORIGINAL BOARD:")
    print(np.array(board))

    constrs = set()

    # the values that have already been set must be respected
    for i in range(D):
        for j in range(D):
            if board[i][j] > 0:
                constr = [0] * N
                constr[to_idx(i, j, board[i][j] - 1)] = 1
                constrs.add(tuple(constr))

    # each cell must have exactly one value set
    for i in range(0, N, D):
        constr = np.zeros(N)
        constr[i:i+D] = 1
        constrs.add(tuple(constr))

    ineq_constrs = set()

    # row, column, and box constraints. these will be input as inequality
    # constraints, but will combine with the "each cell has exactly one value set"
    # constraints to effectively behave like equality constraints
    for i in range(D):
        for v in range(D):
            constrs.add(tuple(get_row_constr(i, v)))
            constrs.add(tuple(get_col_constr(i, v)))

    for i in range(3):
        for j in range(3):
            for v in range(D):
                constrs.add(tuple(get_box_constr(i, j, v)))

    x = cp.Variable(N, boolean=True)

    A = np.array(list(constrs))
    b = np.ones(len(constrs))

    # because the solution is unique, this is just a feasibility problem
    objective = cp.Minimize(0)
    constraints = [A @ x == b]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS_BB)

    assert problem.status == cp.OPTIMAL, problem.status

    # show solution
    sol = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            idx = to_idx(i, j, 0)
            sol[i, j] = np.argmax(x.value[idx:idx+D]) + 1
    print()
    print("SOLUTION:")
    print(sol)
