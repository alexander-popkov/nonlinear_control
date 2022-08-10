from problems.NonLinearProblem import NonLinearProblem
from task.nonlinear_scalar_task import nonlinear_scalar_task
from task.pendulum_control import pendulum_control
from plotting.plotting import plot_results, plot_trajectories, plot_controls, plot_phase_portrait
from numpy.linalg import norm
from time import time

timer = time()

nlp = NonLinearProblem(pendulum_control())
u_sequence = []
xi_sequence = []
x_sequence = []
stats = []

Sl = 10

for i in range(Sl):
    if i == 0:
        nlp.get_control_by_linearization(method='start_point')
    else:
        nlp.get_control_by_linearization()
    nlp.calculate_closed_trajectory()
    u_sequence.append(nlp.u)
    xi_sequence.append(nlp.xi)
    x_sequence.append(nlp.x)
    stats.append((nlp.obj_value, norm(nlp.task.xT - nlp.x(nlp.task.T))))

print(time() - timer)

for stat in stats:
    print(stat)

plot_results(stats)
# plot_controls(nlp, u_sequence)
# plot_trajectories(nlp, xi_sequence, x_sequence)
plot_phase_portrait(nlp, xi_sequence, x_sequence)
