from task.Task import Task
from scipy.integrate import odeint
from numpy import array, linspace, floor, ceil
from typing import Dict, Callable, List


def time_dict_to_func(d: Dict[float, array], t0: float, t1: float, precision: int) \
        -> Callable[[float], array]:
    def f(t):
        if t < t0:
            return d[t0]
        elif t > t1:
            return d[t1]
        t_min = round(floor(t * (10 ** precision)) / (10 ** precision), precision)
        t_max = round(ceil(t * (10 ** precision)) / (10 ** precision), precision)
        if t_min == t_max:
            return d[t_min]

        coef = (t_max - t) / (t_max - t_min)
        return coef * d[t_min] + (1 - coef) * d[t_max]

    return f


def calculate_trajectory(rhs: Callable[[float, array], array], task: Task) -> Callable[[float], array]:
    t_grid = linspace(start=task.t0, stop=task.T, num=task.steps)
    x_grid = odeint(rhs, task.x0, t_grid, tfirst=True)

    trajectory = {}
    for t_i, x_i in zip(t_grid, x_grid):
        trajectory[round(t_i, 4)] = x_i

    return time_dict_to_func(trajectory, task.t0, task.T, task.precision)


def build_positional_function(sequence: List[Callable[[float, array], array]], task: Task) -> Callable[[float], array]:
    def f(t: float) -> array:
        if t < task.t0:
            return sequence[0](t)
        elif t >= task.T:
            return sequence[-1](t)
        return sequence[int(t / task.h)](t)

    return f


def calculate_objective_value(u: Callable[[float, array], array], task: Task) -> float:
    val = 0
    for k in range(task.N):
        u_val = u((k+0.5) * task.h)
        val += task.h * sum(sum(task.Q[i1, i2] * u_val[i1] * u_val[i2] for i2 in range(task.r))
                            + task.d[i1] * u_val[i1] for i1 in range(task.r))

    return val
