from problems.NonLinearProblem import NonLinearProblem
from task.Task import Task
from matplotlib import pyplot as plt
from numpy import linspace


def plot_control(nlp: NonLinearProblem):
    plt.rcParams.update({
        "font.family": "Times New Roman",
        'mathtext.fontset': 'cm'
    })
    ax = plt.gca()

    ax.annotate('$t$', xy=(1., 0.), ha='left', va='top', xycoords='axes fraction')
    ax.annotate('$u(t)$', xy=(0., 1.), ha='left', va='top', xycoords='axes fraction',
                textcoords='offset points')
    plt.title('Управление')
    plt.grid(True)
    t = linspace(start=nlp.task.t0, stop=nlp.task.T, num=nlp.task.steps)
    u = [nlp.u(t_i) for t_i in t]
    plt.plot(t, u, label='$u(t)$')
    plt.legend()

    plt.show()


def plot_trajectory(nlp: NonLinearProblem):
    plt.rcParams.update({
        "font.family": "Times New Roman",
        'mathtext.fontset': 'cm'
    })
    ax = plt.gca()

    ax.annotate('$t$', xy=(1., 0.), ha='left', va='top', xycoords='axes fraction')
    ax.annotate(r'$x(t), \xi(t)$', xy=(0., 1.), ha='left', va='top', xycoords='axes fraction',
                textcoords='offset points')
    plt.title('Положение объекта')
    plt.grid(True)
    t = linspace(start=nlp.task.t0, stop=nlp.task.T, num=nlp.task.steps)
    plt.plot(t, [nlp.x(t_i) for t_i in t], label='$x(t)$')
    plt.plot(t, [nlp.xi(t_i) for t_i in t], label=r'$\xi(t)$')
    plt.legend()

    plt.show()


def plot_controls(nlp: NonLinearProblem, u_sequence: list):
    plt.rcParams.update({
        "font.family": "Times New Roman",
        'mathtext.fontset': 'cm'
    })
    ax = plt.gca()

    ax.annotate('$t$', xy=(1.02, 0.015), ha='left', va='top', xycoords='axes fraction')
    ax.annotate('$u(t)$', xy=(-0.02, 1.05), ha='left', va='top', xycoords='axes fraction',
                textcoords='offset points')
    plt.grid(True)
    plt.xlim(nlp.task.t0, nlp.task.T)
    plt.ylim(nlp.task.L1[0] - 0.02, nlp.task.L2[0] + 0.02)
    t = linspace(start=nlp.task.t0, stop=nlp.task.T, num=nlp.task.steps)
    for i in range(5):
        plt.plot(t, [u_sequence[i](t_i)[0] for t_i in t], label=r'$u^{(' + str(i+1) + r')}(t)$')
    plt.legend(loc='right')

    plt.show()


def plot_trajectories(nlp: NonLinearProblem, xi_sequence: list, x_sequence: list):
    plt.rcParams.update({
        "font.family": "Times New Roman",
        'mathtext.fontset': 'cm'
    })
    y_max = 9
    for i in range(4):

        plt.subplot(2, 2, i+1)
        ax = plt.gca()
        ax.annotate('$t$', xy=(1.03, 0.04), ha='left', va='top', xycoords='axes fraction')
        ax.annotate(r'$\xi, x$', xy=(-0.03, 1.1), ha='left', va='top', xycoords='axes fraction',
                    textcoords='offset points')
        plt.title('$s = {}$'.format(i+1))
        plt.grid(True)
        plt.xlim(nlp.task.t0 - 0.1, nlp.task.T + 0.1)
        plt.ylim(-0.1, y_max + 0.1)
        plt.xticks(list(range(int(nlp.task.t0), int(nlp.task.T) + 1, 1)))
        plt.yticks(list(range(0, y_max+1, 2)))
        t = linspace(start=nlp.task.t0, stop=nlp.task.T, num=nlp.task.steps)
        plt.plot(t, [xi_sequence[i](t_i)[0] for t_i in t], label=r'$\xi^{(' + str(i+1) + ')}(t)$')
        plt.plot(t, [x_sequence[i](t_i)[0] for t_i in t], label='$x^{(' + str(i+1) + ')}(t)$')
        plt.plot(nlp.task.t0, nlp.task.x0[0], 'b.', nlp.task.T, nlp.task.xT[0], 'b.')
        plt.legend()

    plt.subplots_adjust(hspace=0.33)
    plt.show()


def plot_results(stats: list):
    plt.rcParams.update({
        "font.family": "Times New Roman",
        'mathtext.fontset': 'cm'
    })
    s_grid = range(1, len(stats) + 1, 1)
    plt.subplot(2, 1, 1)
    ax = plt.gca()

    ax.annotate('$s$', xy=(1.02, 0.02), ha='left', va='top', xycoords='axes fraction')
    ax.annotate(r'$J_s - J_{s-1}$', xy=(-0.03, 1.15), ha='left', va='top', xycoords='axes fraction',
                textcoords='offset points')
    plt.title('Сходимость целевой функции')
    plt.grid(True)
    plt.xlim(1 - 0.1, len(s_grid) + 0.1)
    plt.xticks(list(range(1, len(s_grid)+1, 1)))
    ax.set_yscale('log')
    plt.plot(s_grid[1:], [abs(stats[s - 1][0] - stats[s - 2][0]) for s in s_grid[1:]], 'm',
             s_grid[1:], [abs(stats[s - 1][0] - stats[s - 2][0]) for s in s_grid[1:]], 'm.')

    plt.subplot(2, 1, 2)
    ax = plt.gca()

    ax.annotate('$s$', xy=(1.02, 0.02), ha='left', va='top', xycoords='axes fraction')
    ax.annotate(r'$|| Hx^{(s)}(T) - g^0 ||$', xy=(-0.08, 1.15), ha='left', va='top',
                xycoords='axes fraction', textcoords='offset points')
    plt.title('Сходимость финального положения')
    plt.grid(True)
    plt.xlim(1 - 0.1, len(s_grid) + 0.1)
    plt.xticks(list(range(1, len(s_grid)+1, 1)))
    ax.set_yscale('log')
    s_grid = range(1, len(stats) + 1, 1)
    plt.plot(s_grid, [stats[s - 1][1] for s in s_grid], 'g', s_grid, [stats[s - 1][1] for s in s_grid], 'g.')

    plt.subplots_adjust(hspace=0.5)
    plt.show()


def plot_phase_portrait(nlp: NonLinearProblem, xi_sequence: list, x_sequence: list):
    plt.rcParams.update({
        "font.family": "Times New Roman",
        'mathtext.fontset': 'cm'
    })
    for i in range(4):

        plt.subplot(2, 2, i+1)
        ax = plt.gca()
        ax.annotate('$x_1$', xy=(1.03, 0.04), ha='left', va='top', xycoords='axes fraction')
        ax.annotate(r'$x_2$', xy=(-0.03, 1.1), ha='left', va='top', xycoords='axes fraction',
                    textcoords='offset points')
        plt.title('$s = {}$'.format(i+1))
        plt.grid(True)
        plt.xlim(-0.1, 4)
        plt.ylim(-2, 4)
        t = linspace(start=nlp.task.t0, stop=nlp.task.T, num=nlp.task.steps)
        plt.plot([xi_sequence[i](t_i)[0] for t_i in t], [xi_sequence[i](t_i)[1] for t_i in t],
                 label=r'$\xi^{(' + str(i+1) + ')}(t)$')
        plt.plot([x_sequence[i](t_i)[0] for t_i in t], [x_sequence[i](t_i)[1] for t_i in t],
                 label='$x^{(' + str(i+1) + ')}(t)$')
        plt.plot(nlp.task.x0[0], nlp.task.x0[1], 'b.', nlp.task.xT[0], nlp.task.xT[1], 'b.')
        plt.legend(loc='upper left')

    plt.subplots_adjust(hspace=0.33)
    plt.subplots_adjust(wspace=0.25)
    plt.show()


def plot_realtime_portrait(task: Task, x_sequence: list, X):
    plt.rcParams.update({
        "font.family": "Times New Roman",
        'mathtext.fontset': 'cm'
    })

    plt.subplot(1, 2, 1)
    task1 = Task()
    task1.copy_task(task)
    ax = plt.gca()
    ax.annotate('$x_1$', xy=(1.03, 0.04), ha='left', va='top', xycoords='axes fraction')
    ax.annotate(r'$x_2$', xy=(-0.03, 1.1), ha='left', va='top', xycoords='axes fraction',
                textcoords='offset points')
    plt.grid(True)
    plt.ylim(-2, 4)
    plt.yticks(list(range(-2, 5, 2)))
    k = 1
    for x in x_sequence:
        t = linspace(start=task1.t0, stop=task1.T, num=task1.steps)
        plt.plot([x(t_i)[0] for t_i in t], [x(t_i)[1] for t_i in t], label=r'$\bar x^{(' + str(k) + ')}(t)$')

        task1.N -= 1
        task1.t0 += task1.h
        task1.steps = task1.get_steps()
        k += 1
    plt.plot(task1.x0[0], task1.x0[1], 'b.', task1.xT[0], task1.xT[1], 'b.')
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)
    ax = plt.gca()
    ax.annotate('$x_1$', xy=(1.03, 0.04), ha='left', va='top', xycoords='axes fraction')
    ax.annotate(r'$x_2$', xy=(-0.03, 1.1), ha='left', va='top', xycoords='axes fraction',
                textcoords='offset points')
    plt.grid(True)
    plt.ylim(-2, 4)
    plt.yticks(list(range(-2, 5, 2)))
    t = linspace(start=task.t0, stop=task.T, num=task.steps)
    print(task.t0, task.T, task.steps)
    plt.plot([X(t_i)[0] for t_i in t], [X(t_i)[1] for t_i in t], 'r', label=r'$\bar x(t)$')
    plt.plot(task.x0[0], task.x0[1], 'b.', task.xT[0], task.xT[1], 'b.')
    plt.legend(loc='upper left')

    plt.subplots_adjust(wspace=0.25)
    plt.show()


def plot_positional_portrait(task: Task, X):
    plt.rcParams.update({
        "font.family": "Times New Roman",
        'mathtext.fontset': 'cm'
    })

    ax = plt.gca()
    ax.annotate('$x_1$', xy=(1.03, 0.04), ha='left', va='top', xycoords='axes fraction')
    ax.annotate(r'$x_2$', xy=(-0.03, 1.1), ha='left', va='top', xycoords='axes fraction',
                textcoords='offset points')
    plt.grid(True)

    t = linspace(start=task.t0, stop=task.T, num=task.steps)
    plt.plot([X(t_i)[0] for t_i in t], [X(t_i)[1] for t_i in t], label=r'$\bar x(t)$')
    plt.plot(task.x0[0], task.x0[1], 'b.', task.xT[0], task.xT[1], 'b.')
    plt.legend()

    plt.show()
