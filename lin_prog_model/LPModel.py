from lin_prog_task.LPTask import LPTask
from coptpy import Envr, Model, tupledict, tuplelist, quicksum, COPT
from numpy import array


def create_vars(M: Model, L1: array, L2: array, r: int, N: int, n: int):
    indices = tuplelist()
    lb = {}
    ub = {}
    for i in range(r):
        for k in range(N):
            indices.append((i, k))
            lb[i, k] = L1[i]
            ub[i, k] = L2[i]
    U = M.addVars(indices, vtype=COPT.CONTINUOUS, lb=lb, ub=ub, nameprefix='U')
    z = M.addVars(tuplelist((l, j) for l in range(2) for j in range(n)), vtype=COPT.CONTINUOUS, nameprefix='z')

    return U, z


def add_terminal_constraints(M: Model, U: tupledict, z: tupledict, A: array, b: array, n: int):
    M.addConstrs((U.prod(A[j]) == b[j] + z[0, j] - z[1, j] for j in range(n)), nameprefix='terminal_constr')


def add_strong_constraint(M: Model, z: tupledict):
    strong = M.addConstr(quicksum(z) == 0, name='strong')
    return strong


def add_objective(M: Model, U: tupledict, Q: array, D: array, r: int, N: int):
    M.setObjective(quicksum(Q[i1, i2, k] * U[i1, k] * U[i2, k]
                            for i1 in range(r) for i2 in range(r) for k in range(N))
                   + quicksum(D[i, k] * U[i, k] for i in range(r) for k in range(N)))


def add_deviation_objective(M: Model, z: tupledict):
    M.setObjective(quicksum(z))


def search_optimal_control(task: LPTask):
    env = Envr()
    M: Model = env.createModel('Optimal Piecewise Constant Control')

    U, z = create_vars(M, task.L1, task.L2, task.r, task.N, task.n)
    add_terminal_constraints(M, U, z, task.A, task.b, task.n)
    strong = add_strong_constraint(M, z)
    add_objective(M, U, task.Q, task.D, task.r, task.N)

    M.write('lp_model.lp')
    M.solve()

    if M.getAttr(COPT.Attr.LpStatus) == COPT.INFEASIBLE:
        print('***INFEASIBLE MODEL***')
        M.remove(strong)
        add_deviation_objective(M, z)
        M.write('lp_model.lp')
        M.solve()

    M.write('lp_model.sol')
    return array([[U[i, k].X for i in range(task.r)] for k in range(task.N)]), M.getObjective().getValue()
