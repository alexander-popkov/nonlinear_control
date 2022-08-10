from task.Task import Task
from numpy import array, sin, cos, pi


def pendulum_control() -> Task:
    # Задание задачи управления маятником

    # Задание уравнения d^2x/dt^2 + sin(x) = u
    def f(t: float, x: array) -> array:
        return array([x[1], -sin(x[0])])

    def B(t: float) -> array:
        return array([[0, 0], [-1, 1]])

    # Задание матрицы Якоби df(t, x)/dx
    def fx(t: float, x: array) -> array:
        return array([[0, 1], [-cos(x[0]), 0.]])

    # Прямые ограничения на управление L1 <= u <= L2
    L1 = array([0., 0.])
    L2 = array([4., 4.])

    # Задание терминальных условий x(0) = x0, x(T) = xT
    x0 = array([pi, 1.])
    xT = array([0., 0.])

    # Целевая функция int_0^T [u1(t) + u2(t)] dt --> min
    Q: array = array([[0., 0.], [0., 0.]])
    d: array = array([1., 1.])
    c: array = array([0., 0.])

    # Время начала и окончания управления
    t0: float = 0
    T: float = 6

    # Количество разбиений отрезка управления
    N: int = 4

    # Количество точек на временной шкале
    precision: int = 2

    task = Task(f=f, B=B, fx=fx, L1=L1, L2=L2, x0=x0, xT=xT, Q=Q, d=d, c=c, t0=t0, T=T, N=N, precision=precision)

    return task
