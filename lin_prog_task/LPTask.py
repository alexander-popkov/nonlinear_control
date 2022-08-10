from numpy import array, shape


class LPTask:
    """
    Задание параметров интервальной задачи линейного программирования --- сведенной к ЛП линейной задачи управления
    """
    def __init__(self, A: list=None, b: array=None, L1: array=None, L2: array=None, Q: array=None, D: array=None):
        # Терминальные условия AU=b
        self.A = A
        self.b = b

        # Прямые ограничения L1 <= U <= L2
        self.L1 = L1
        self.L2 = L2

        # Квадратичная ЦФ: U'QU + D'U --> min
        self.Q = Q
        self.D = D

        # Размерности параметров
        if A is not None:
            self.n, self.r, self.N = len(A), shape(Q)[0], shape(Q)[2]

    def __bool__(self):
        return bool(self.A)
