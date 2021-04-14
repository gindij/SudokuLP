import cvxpy as cp
import numpy as np
import sys
import time

D = 9
N = D**3

def parse_board(path, empty_char="0"):
    boards = []
    with open(path, "r") as board_file:
        lines = board_file.readlines()
        i = 0
        for line in lines[::10]:
            boards.append([
                [int(x) if x != empty_char else -1 for x in row[:-1]]
                for row in lines[i+1:i+10]
            ])
            i += 10
    return boards

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


def solve_one(board, verbose=False, feasibility=False):

    if verbose:
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
    obj_fn = 0 if feasibility else x.T @ np.ones(N)
    objective = cp.Minimize(obj_fn)
    constraints = [A @ x == b]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS_BB)

    solved = problem.status == cp.OPTIMAL, problem.status

    # show solution
    sol = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            idx = to_idx(i, j, 0)
            sol[i, j] = np.argmax(x.value[idx:idx+D]) + 1
    if verbose:
        print()
        print("SOLUTION:")
        print(sol)

    return solved

if __name__ == "__main__":
    filename = sys.argv[1]
    boards = parse_board(filename)
    board_idx = sys.argv[2]
    feasibility = sys.argv[3] == "feas"
    if board_idx != "all":
        solve_one(boards[int(board_idx)], verbose=True, feasibility=feasibility)
    else:
        solve_times = []
        for i, board in enumerate(boards):
            start = time.time()
            solved = solve_one(board, verbose=False, feasibility=feasibility)
            end = time.time()
            ms = (end - start) * 1000
            solve_times.append(ms)
            print(f"solved problem {i+1} in {ms}ms")
        print(f"mean solve time: {np.mean(solve_times)}")
        print(f"median solve time: {np.median(solve_times)}")
        print(f"std solve time: {np.std(solve_times, ddof=1)}")
