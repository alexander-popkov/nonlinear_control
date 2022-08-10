from problems.LinearProblem import LinearProblem
from task.Task import Task
from common.common import calculate_trajectory
from numpy import array
from typing import Callable, Tuple


class NonLinearProblem:

    def __init__(self, task: Task):
        self.task = task
        self.u: Callable[[float], array] = lambda t: None
        self.x: Callable[[float], array] = lambda t: None
        self.xi: Callable[[float], array] = lambda t: None
        self.obj_value = float('inf')

    def rhs(self, t: float, x: array, u: array) -> array:
        return self.task.f(t, x) + self.task.B(t).dot(u)

    def closed_rhs(self, t: float, x: array) -> array:
        return self.rhs(t, x, self.u(t))

    def generate_lin_params(self) -> \
            Tuple[Callable[[float], array], Callable[[float], array]]:

        def A(t: float) -> array:
            return self.task.fx(t, self.x(t))

        def C(t: float) -> array:
            return self.task.f(t, self.x(t)) - self.task.fx(t, self.x(t)).dot(self.x(t))

        return A, C

    def generate_lin_params_by_start_point(self) -> \
            Tuple[Callable[[float], array], Callable[[float], array]]:

        def A(t: float) -> array:
            return self.task.fx(t, self.task.x0)

        def C(t: float) -> array:
            return self.task.f(t, self.task.x0) - self.task.fx(t, self.task.x0).dot(self.task.x0)

        return A, C

    def get_control_by_linearization(self, method: str = 'prev_solution'):
        lp_task = Task()
        lp_task.copy_task(self.task)

        if method == 'prev_solution':
            lp_task.A, lp_task.C = self.generate_lin_params()
        elif method == 'start_point':
            lp_task.A, lp_task.C = self.generate_lin_params_by_start_point()

        lp = LinearProblem(lp_task)
        lp.calculate_optimal_trajectory()
        self.u = lp.opt_u
        self.xi = lp.opt_x
        self.obj_value = lp.obj_value

    def calculate_closed_trajectory(self):
        self.x = calculate_trajectory(self.closed_rhs, self.task)
