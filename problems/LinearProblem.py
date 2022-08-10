from task.Task import Task
from lin_prog_task.LPTask import LPTask
from lin_prog_model.LPModel import search_optimal_control
from common.common import time_dict_to_func, calculate_trajectory
from scipy.integrate import odeint, quad
from numpy import array, zeros, linspace, transpose
from typing import Callable


class LinearProblem:

    def __init__(self, task: Task):
        self.task = task
        self.FM: Callable[[float], array] = lambda t: None
        self.lp_task: LPTask = LPTask()
        self.opt_u: Callable[[float], array] = lambda t: None
        self.opt_x: Callable[[float], array] = lambda t: None
        self.obj_value = float('inf')

    def rhs(self, t: float, x: array, u: array) -> array:
        return self.task.A(t).dot(x) + self.task.B(t).dot(u) + self.task.C(t)

    def homogeneous_rhs(self, t: float, x: array) -> array:
        return self.task.A(t).dot(x)

    def rhs_zeros_control(self, t: float, x: array) -> array:
        return self.task.A(t).dot(x) + self.task.C(t)

    def conjugate_rhs(self, t: float, x: array):
        return -transpose(self.task.A(t)).dot(x)

    def closed_rhs(self, t: float, x: array) -> array:
        return self.rhs(t, x, self.opt_u(t))

    def init_opt_u(self, u: dict) -> Callable[[float], array]:
        def opt_u(t: float) -> array:
            if t < self.task.t0:
                return u[0, :]
            elif t > self.task.T:
                return u[-1, :]

            for k in range(self.task.N):
                if t <= self.task.t0 + (k + 1) * self.task.h:
                    return u[k, :]

        return opt_u

    def calculate_FM(self):
        t_grid = linspace(start=self.task.T, stop=self.task.t0, num=self.task.steps)
        _FM = {round(t_i, 4): zeros((self.task.n, self.task.n)) for t_i in t_grid}
        for j in range(self.task.n):
            start = zeros(self.task.n)
            start[j] = 1
            x_grid = odeint(self.conjugate_rhs, start, t_grid, tfirst=True)
            for t_i, x_i in zip(t_grid, x_grid):
                _FM[round(t_i, 4)][j, :] = x_i
        self.FM = time_dict_to_func(_FM, self.task.t0, self.task.T, self.task.precision)

    def define_lp_params(self):
        lp_A = []
        for j in range(self.task.n):
            lp_A.append({})
            for i in range(self.task.r):
                FM_b = lambda t: self.FM(t)[j, :].dot(self.task.B(t)[:, i])
                for k in range(self.task.N):
                    bI = quad(FM_b, self.task.t0 + k * self.task.h, self.task.t0 + (k + 1) * self.task.h)
                    lp_A[j][i, k] = bI[0]

        t_grid = linspace(start=self.task.t0, stop=self.task.T, num=self.task.steps)
        x_grid = odeint(self.rhs_zeros_control, self.task.x0, t_grid, tfirst=True)
        lp_b = self.task.xT - x_grid[-1]

        lp_Q = array([[[self.task.Q[i1, i2] * self.task.h for _ in range(self.task.N)]
                       for i2 in range(self.task.r)] for i1 in range(self.task.r)])

        lp_D = array([[self.task.d[i] * self.task.h for _ in range(self.task.N)] for i in range(self.task.r)])

        self.lp_task = LPTask(A=lp_A, b=lp_b, L1=self.task.L1, L2=self.task.L2, Q=lp_Q, D=lp_D)

    def find_optimal_control(self):
        if self.FM(0) is None:
            self.calculate_FM()

        if not self.lp_task:
            self.define_lp_params()

        _opt_u, self.obj_value = search_optimal_control(self.lp_task)

        self.opt_u = self.init_opt_u(_opt_u)

    def calculate_optimal_trajectory(self):
        if self.opt_u(0) is None:
            self.find_optimal_control()

        self.opt_x = calculate_trajectory(self.closed_rhs, self.task)
