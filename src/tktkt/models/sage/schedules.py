from abc import abstractmethod, ABC

from math import exp
from math import log as ln


class Schedule(ABC):
    """
    Function defined between 0 and 1.
    """
    @abstractmethod
    def get(self, t: float) -> float:
        pass


class LinearSchedule(Schedule):
    def __init__(self, start: int, end: int):
        self.a = end-start
        self.b = start

    def get(self, t: float) -> float:
        return self.a*t + self.b


class DoubleLinearSchedule(Schedule):

    def __init__(self, start: int, mid: int, end: int, t_mid: float=0.5):
        self.line1 = LinearSchedule(start, mid)
        self.line2 = LinearSchedule(mid, end)
        self.t_mid = t_mid

    def get(self, t: float) -> float:
        if t < self.t_mid:
            return self.line1.get(t/self.t_mid)
        else:
            return self.line2.get((t-self.t_mid)/(1-self.t_mid))


class Dilation(Schedule):
    """
    Functions defined between 0 and 1 that return 0 at 0 and return 1 at 1.
    Technically called a "companding curve". Time dilation is a known physical phenomenon, however.
    """
    pass


class IdentityDilation(Dilation):
    def get(self, t: float) -> float:
        return t


class PowerDilation(Dilation):
    """
    f(t) = t^a
    """

    def __init__(self, a: float):
        assert a > 0
        self.a = a

    def get(self, t: float) -> float:
        return pow(t, self.a)


class ExponentialDilation(Dilation):
    """
    f(t) = a*(1-e^{b*t}) where b = ln(1 - 1/a), which is equivalent to
    f(t) = a*(1-(1-1/a)^t)
    """

    def __init__(self, a: float):
        assert a < 0 or a > 1
        self.a = a
        self.b = ln(1 - 1/a)

    def get(self, t: float) -> float:
        return self.a*(1-exp(self.b*t))


class LogDilation(Dilation):
    """
    f(t) = a*ln(1 + b*x) where b = e^{1/a} - 1
    """

    def __init__(self, a: float):
        assert a != 0
        self.a = a
        self.b = exp(1/a) - 1

    def get(self, t: float) -> float:
        return self.a * ln(1 + self.b*t)
