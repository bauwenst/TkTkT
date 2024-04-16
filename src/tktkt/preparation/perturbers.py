from abc import abstractmethod
import numpy.random as npr

from .mappers import TextMapper

RNG = npr.default_rng(0)


class Perturber(TextMapper):

    def __init__(self, p: float):  # Chance of perturbing the input at all.
        self.p = p

    def convert(self, text: str) -> str:
        if RNG.random() < self.p:
            return self.perturb(text)
        else:
            return text

    @abstractmethod
    def perturb(self, text: str) -> str:
        pass


class Gobble(Perturber):

    def perturb(self, text: str) -> str:
        return ""


class FixedTruncate(Perturber):
    """
    Truncate a fixed amount of characters from the start and/or end.
    """

    def __init__(self, p: float, from_start: int=0, from_end: int=1):
        super().__init__(p)
        self.start = from_start
        self.end   = from_end

    def perturb(self, text: str) -> str:
        return text[self.start:len(text)-self.end]


class RandomPop(Perturber):

    def __init__(self, p: float, min_n: int=1, max_n: int=1):
        super().__init__(p)
        assert min_n <= max_n

        self.min = min_n
        self.max = max_n

    def perturb(self, text: str) -> str:
        N = len(text)
        n = RNG.choice(range(min(N,self.min), min(N,self.max)+1))
        indices = set(RNG.choice(range(0,N), size=n, replace=False))
        return "".join(map(lambda i: text[i], filter(lambda i: i not in indices, range(0,N))))


class GeometricPop(Perturber):
    """
    The amount of pops is geometrically distributed, although capped between the minimum and maximum.
    """

    def __init__(self, p: float, min_n: int, max_n: int, probability_to_stop: float, start_at_max=False):
        super().__init__(p)
        assert min_n <= max_n

        self.min = min_n
        self.max = max_n
        self.q = probability_to_stop
        self.reversed = start_at_max

    def perturb(self, text: str) -> str:
        N = len(text)
        if not self.reversed:
            n = self.min
            while n < self.max and RNG.random() > self.q:  # Sanity check: if q == 1, you always stop.
                n += 1
        else:
            n = self.max
            while n > self.min and RNG.random() > self.q:
                n -= 1

        indices = set(RNG.choice(range(0, N), size=n, replace=False))
        return "".join(map(lambda i: text[i], filter(lambda i: i not in indices, range(0, N))))


class ProportionalPop(Perturber):

    def __init__(self, p: float, probability_to_pop: float):
        super().__init__(p)
        self.local_p = probability_to_pop

    def perturb(self, text: str) -> str:
        result = ""
        i = 0
        while i < len(text):
            if RNG.random() > self.local_p:
                result += text[i]
            i += 1
        return result
