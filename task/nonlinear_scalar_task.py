from task.Task import Task
from numpy import array


def nonlinear_scalar_task() -> Task:
    # Задание нелинейной скалярной задачи

    # Задание уравнения dx/dt = 1/(x^2 + 1) + u
    def f(t: float, x: array) -> array:
        return array([1 / (x[0] ** 2 + 1)])

    def B(t: float) -> array:
        return array([[1]])

    # Задание матрицы Якоби df(t, x)/dx
    def fx(t: float, x: array) -> array:
        return array([[-2 * x[0] / (x[0] ** 2 + 1) ** 2]])

    # Прямые ограничения на управление L1 <= u <= L2
    L1 = array([0.])
    L2 = array([2])

    # Задание терминальных условий x(0) = x0, x(T) = xT
    x0 = array([0.])
    xT = array([8.])

    # Целевая функция int_0^T u^2(t) dt --> min
    Q: array = array([[1.]])
    d: array = array([0.])
    c: array = array([0.])

    # Время окончания управления
    t0: float = 0
    T: float = 5

    # Количество разбиений отрезка управления
    N: int = 10

    # Точность решения дифференциальных уравнений (в числе знаков после запятой)
    precision: int = 2

    task = Task(f=f, B=B, fx=fx, L1=L1, L2=L2, x0=x0, xT=xT, Q=Q, d=d, c=c, t0=t0, T=T, N=N, precision=precision)

    return task
