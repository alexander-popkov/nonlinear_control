from numpy import array
from typing import Callable, Tuple


class Task:

    def __init__(
            self,
            f: Callable[[float, array], array] = None,
            fx: Callable[[float, array], array] = None,
            A: Callable[[float, array], array] = None,
            B: Callable[[float, array], array] = None,
            C: Callable[[float, array], array] = None,
            L1: array = None,
            L2: array = None,
            x0: array = None,
            xT: array = None,
            Q: array = None,
            d: array = None,
            c: array = None,
            t0: float = None,
            T: float = None,
            N: int = None,
            precision: int = None
    ):
        # Задание нелинейной части в уравнении dx/dt = f(t, x) + B(t)u
        self.f = f

        # Задание матрицы Якоби df(t, x)/dx
        self.fx = fx

        # Задание уравнения dx/dt = A(t)x + B(t)u + C(t)
        self.A = A
        self.B = B
        self.C = C

        # Прямые ограничения на управление L1 <= u <= L2
        self.L1 = L1
        self.L2 = L2

        # Задание терминальных условий x(0) = x0, x(T) = xT
        self.x0 = x0
        self.xT = xT

        # Целевая функция в виде int_0^T [u(t)'*Q*u(t) + d*u(t)] dt + c*x(T) --> min
        self.Q = Q
        self.d = d
        self.c = c

        # Время начала и окончания управления
        self.t0 = t0
        self.T = T

        # Количество разбиений отрезка управления
        self.N = N

        # Точность решения дифференциальных уравнений (в числе знаков после запятой)
        self.precision = precision

        # Количество точек на временной шкале
        self.steps = self.get_steps()

        # Определение размерностей задачи
        self.n, self.r = self.get_dimensions()

        # Период квантования задачи
        self.h = self.get_quantization()

    def get_dimensions(self) -> Tuple[int, int]:
        """
        Определение размерностей задачи
        :return: 1) n --- размерность фазового вектора,
                 2) r --- размерность вектора управлений
        """
        n = None if self.x0 is None else len(self.x0)
        r = None if self.L1 is None else len(self.L1)

        return n, r

    def get_quantization(self) -> float:
        """
        Определение периода квантования задачи T/N
        :return: период квантования
        """
        return (self.T - self.t0) / self.N if self.T else 0

    def get_steps(self) -> int:
        """
        Определение временной шкалы исходя из заданной точности
        :return: количество точек на временной шкале
        """
        return int(round((self.T - self.t0) * (10 ** self.precision) + 1)) if self.T else 0

    def copy_task(self, task: 'Task'):
        self.f = task.f
        self.fx = task.fx
        self.A = task.A
        self.B = task.B
        self.C = task.C
        self.L1 = task.L1
        self.L2 = task.L2
        self.x0 = task.x0
        self.xT = task.xT
        self.Q = task.Q
        self.d = task.d
        self.c = task.c
        self.t0 = task.t0
        self.T = task.T
        self.N = task.N
        self.precision = task.precision
        self.steps = self.get_steps()
        self.n, self.r = self.get_dimensions()
        self.h = self.get_quantization()
