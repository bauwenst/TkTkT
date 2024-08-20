from abc import abstractmethod, ABC
from typing import Tuple, Dict, List, Iterable

from functools import reduce
import numpy as np
import numpy.random as npr

from ..util.functions import softmax, relu
from .mappers import TextMapper

RNG = npr.default_rng(0)


class Perturber(TextMapper):
    """
    Non-invertible text transformation that applies a perturbation function to
    a random subset of inputs, and leaves the others unchanged.
    """

    def __init__(self, p: float):
        """
        :param p: Chance of perturbing the input at all.
        """
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
    """
    Delete the entire word.
    """

    def perturb(self, text: str) -> str:
        return ""


class FixedTruncate(Perturber):
    """
    Truncate a fixed amount of characters from the start and/or end.
    TODO: A probabilistic version would be interesting. Have e.g. a capped geometric distribution across the head characters and the tail characters.
    """

    def __init__(self, p: float, from_start: int=0, from_end: int=1):
        super().__init__(p)
        self.start = from_start
        self.end   = from_end

    def perturb(self, text: str) -> str:
        return text[self.start:relu(len(text)-self.end)]


class PerturbationPointSampler(ABC):
    """
    Given a string, generates the points where we want to cause a perturbation.
    """

    @abstractmethod
    def sample(self, characters: str) -> Iterable[int]:
        pass


class FixedUniformSampler(PerturbationPointSampler):
    """
    Choose a random amount of characters between a given minimum and maximum,
    then randomly chooses exactly that amount of character positions.
    """

    def __init__(self, min_n: int, max_n: int):
        self.min = max(min_n,0)
        self.max = max(max_n,0)

        assert self.min <= self.max

    def sample(self, characters: str) -> Iterable[int]:
        N = len(characters)
        n = RNG.integers(range(min(N,self.min), min(N,self.max)+1))
        return RNG.choice(N, size=n, replace=False)


class ConstantSampler(FixedUniformSampler):
    """
    Always samples the same amount of points.
    """

    def __init__(self, n: int):
        super().__init__(n, n)


class ProportionalSampler(PerturbationPointSampler):
    """
    Every character has the same probability of being chosen.

    The amount of characters isn't actually sampled, unlike in the FixedUniformSampler. If you wanted to sample it here,
    it would not come from a uniform distribution, but from a binomial distribution (the amount of positive outcomes
    across N independent experiments each with probability p).
    """

    def __init__(self, probability_per_char: float):
        self.local_p = probability_per_char

    def sample(self, characters: str) -> Iterable[int]:
        return (i for i in range(len(characters)) if RNG.random() < self.local_p)


class GeometricSampler(PerturbationPointSampler):
    """
    Instead of choosing the amount of perturbations uniformly, it is now geometrically distributed, although capped
    between the minimum and maximum. After that, they are chosen uniformly.
    """

    def __init__(self, min_n: int, max_n: int, probability_to_stop: float, start_at_max=False):
        self.min = max(min_n,0)
        self.max = max(max_n,0)

        assert self.min <= self.max

        self.q = probability_to_stop
        self.reversed = start_at_max

    def sample(self, characters: str) -> Iterable[int]:
        N = len(characters)
        if not self.reversed:  # Start at min, approach max.
            n = min(self.min,N)
            while n < N and n < self.max and RNG.random() > self.q:  # Sanity check: if q == 1, you always stop.
                n += 1
        else:  # Start at Start at max, approach min.
            n = min(self.max,N)
            while n > self.min and RNG.random() > self.q:
                n -= 1

        return RNG.choice(N, size=n, replace=False)


class Pop(Perturber):

    def __init__(self, p: float, sampler: PerturbationPointSampler):
        super().__init__(p)
        self.sampler = sampler

    def perturb(self, text: str) -> str:
        pop_indices = set(self.sampler.sample(text))
        return "".join(
            map(lambda i: text[i],
                filter(lambda i: i not in pop_indices,
                       range(0, len(text)))))  # join is faster than building a string with += https://stackoverflow.com/a/1350289/9352077


import clavier
class SubstituteKeyboardTypo(Perturber):
    """
    Uses QWERTY keyboard layout to perturb characters in the word. Characters closer on the keyboard have a higher probability.

    Oriented around the Latin alphabet (using str.isalpha(), for example) because `clavier` has no other support anyway.
    When a sampled character falls outside of this set, it is treated *as if it wasn't sampled*.

    FIXME: A perturber should really ensure that when the sampler says "do n perturbations" that n perturbations happen
           if n perturbable characters exist. That means you should sample from those characters, rather than ignoring bad samples without resampling.
    """

    PMF = Tuple[np.ndarray,List[str]]

    def __init__(self, p: float, sampler: PerturbationPointSampler, temperature: float=1.0):
        """
        :param temperature: Should be any real number greater than 0. Closer to 0 (colder) makes keys very close to the
                            given key much more likely than keys further away. Closer to infinity (hotter) makes distance
                            between keys matter less.
        """
        super().__init__(p)
        self.sampler = sampler

        self.keyboard = clavier.load_qwerty()
        self.supported_keys = {k for k in self.keyboard.keys() if k.isalpha()}
        self.probability_mass_cache: Dict[str, SubstituteKeyboardTypo.PMF] = dict()
        self.temperature = temperature

    def _pmfFromDistances(self, distances: Iterable[float]) -> np.ndarray:
        """
        Simple radial probability densities are e^{-r} and 1/r. I prefer the first because it looks like the
        1s orbital of a hydrogen atom. I'm kind of an electrical engineer myself, you know.
        """
        return softmax(-np.array(distances)/self.temperature)

    def _getPMF(self, char: str) -> "SubstituteKeyboardTypo.PMF":
        if not char in self.supported_keys:
            raise ValueError(f"The given chararacter '{char}' has no probability mass.")

        # If you don't have the PMF yet, impute it now.
        if char not in self.probability_mass_cache:
            neighbours = []
            distances  = []

            for neighbour, distance in self.keyboard.nearest_neighbors(char=char, cache=False, metric="l2"):  # We don't cache because we have our own cache and because caching drops the distances the second time.
                if neighbour in self.supported_keys:
                    neighbours.append(neighbour)
                    distances.append(distance)

            assert neighbours  # Gotta have more than 0 neighbours.

            probabilities = self._pmfFromDistances(distances)
            self.probability_mass_cache[char] = (probabilities, neighbours)

        return self.probability_mass_cache[char]

    def _perturbCharacter(self, char: str) -> str:
        char_lower = char.lower()
        if char_lower not in self.supported_keys:  # Not really how you should perturb :/
            return char

        p, neighbours = self._getPMF(char_lower)
        typo = neighbours[RNG.choice(len(neighbours), p=p)]
        if char.isupper():
            return typo.upper()
        else:
            return typo

    def perturb(self, text: str) -> str:
        sub_indices = set(self.sampler.sample(text))
        return "".join(
            map(lambda i: self._perturbCharacter(text[i]) if i in sub_indices else text[i],
                range(0, len(text))))


class SeriesPerturber(Perturber):
    """
    A word that is selected, has all the given perturbations applied to it in series.
    """

    def __init__(self, p: float, perturbations: Iterable[Perturber]):
        super().__init__(p)
        self.chain = list(perturbations)

    def perturb(self, text: str) -> str:
        return reduce(lambda s,p: p.perturb(s), self.chain, text)


class ParallelPerturber(Perturber):
    """
    A word that is selected, has exactly one of the given perturbations applied to it at random.
    """

    def __init__(self, p: float, perturbations: Iterable[Perturber]):
        super().__init__(p)
        self.pool = list(perturbations)

    def perturb(self, text: str) -> str:
        return self.pool[RNG.integers(len(self.pool))].perturb(text)
