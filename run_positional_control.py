from problems.NonLinearProblem import NonLinearProblem
from task.pendulum_control import pendulum_control
from common.common import build_positional_function, calculate_objective_value
from plotting.plotting import plot_realtime_portrait
from numpy.linalg import norm
from numpy import linspace
from time import time

timer = time()

task = pendulum_control()
u_sequence = []
xi_sequence = []
x_sequence = []
stats = []

Sl = 1

while task.N > 0:
    nlp = NonLinearProblem(task)
    for i in range(Sl):
        if not x_sequence and not i:
            nlp.get_control_by_linearization(method='start_point')
        else:
            if not i:
                nlp.x = x_sequence[-1]
            nlp.get_control_by_linearization()
        nlp.calculate_closed_trajectory()
    u_sequence.append(nlp.u)
    xi_sequence.append(nlp.xi)
    x_sequence.append(nlp.x)
    stats.append((nlp.obj_value, norm(nlp.task.xT - nlp.x(nlp.task.T))))
    task.N -= 1
    task.t0 = round((task.t0 + task.h) * (10 ** task.precision)) / (10 ** task.precision)
    task.x0 = nlp.x(task.t0)
    task.steps = task.get_steps()

print(time() - timer)

task = pendulum_control()

X = build_positional_function(x_sequence, task)
U = build_positional_function(u_sequence, task)

for stat in stats:
    print(stat)

print(calculate_objective_value(U, task))

plot_realtime_portrait(task, x_sequence, X)
