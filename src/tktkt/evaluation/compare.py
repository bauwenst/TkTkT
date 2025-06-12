"""
Evaluations that involve comparing the outputs of two tokenisers per input before aggregating, rather than
aggergating the outputs and then comparing per dataset (which is all other evaluations).
"""
from typing import Tuple, Iterable, List, TypeVar, Generic
from collections import Counter
from abc import ABC, abstractmethod

from ..interfaces.tokeniser import Tokeniser, Preprocessor
from ..preparation.instances import IdentityPreprocessor
from ..util.iterables import streamProgress
from ..util.printing import warn

R = TypeVar("R")  # Result from computing the metric.

class ComparisonMetric(ABC, Generic[R]):

    def __init__(self, texts: Iterable[str], n_repeats: int, global_preprocessor: Preprocessor=None):
        self._iterable = texts
        self._repeats = n_repeats
        if global_preprocessor is None:
            global_preprocessor = IdentityPreprocessor()
        self._preprocessor = global_preprocessor

    @abstractmethod
    def _reset(self):
        """Clear the metric for a new run."""
        pass

    @abstractmethod
    def _add(self, global_pretoken: str, tk1: Tokeniser, tk2: Tokeniser):
        """Add a sample to this object by comparing the tokenisation of both tokenisers on the given pretoken."""
        pass

    @abstractmethod
    def _compute(self) -> R:
        """Using all the collected samples, produce the output of the metric."""
        pass

    def compare(self, tk1: Tokeniser, tk2: Tokeniser) -> R:
        self._reset()
        for text in streamProgress(self._iterable):
            pretokens = self._preprocessor.do(text)
            for pretoken in pretokens:
                for _ in range(self._repeats):
                    self._add(pretoken, tk1, tk2)

        return self._compute()


class ExactMatches(ComparisonMetric[Tuple[float,int,int]]):
    """
    Count how many out of a given amount of texts is tokenised into exactly the same tokens by the two tokenisers.
    """

    def _reset(self):
        self._n_matches = 0
        self._n_total = 0

    def _add(self, global_pretoken: str, tk1: Tokeniser, tk2: Tokeniser):
        self._n_matches += tk1.prepareAndTokenise(global_pretoken) == tk2.prepareAndTokenise(global_pretoken)
        self._n_total   += 1

    def _compute(self) -> R:
        return self._n_matches / self._n_total if self._n_total else 1, self._n_matches, self._n_total


class MicroMacro:

    def __init__(self):
        self._micro_num = 0
        self._micro_den = 0
        self._macro_num = 0
        self._macro_den = 0

    def add(self, num: float, den: float):
        self._micro_num += num
        self._micro_den += den
        self._macro_num += num / den
        self._macro_den += 1

    def compute(self) -> Tuple[float, float]:
        return self._micro_num/self._micro_den if self._micro_den else float("inf") if self._micro_num else 1.0,  \
               self._macro_num/self._macro_den if self._macro_den else float("inf") if self._macro_num else 1.0


class MicroMacroTokenJaccard(ComparisonMetric[Tuple[float, float]]):
    """
    Gives the micro-average and macro-average of Jaccard similarity of the token sequences produced by the given tokenisers.
    """

    def _reset(self):
        self._sum_intersection = 0
        self._sum_union        = 0
        self._sum_jaccard = 0
        self._n_total     = 0

    def _add(self, global_pretoken: str, tk1: Tokeniser, tk2: Tokeniser):
        J, num, denom = jaccard(tk1.prepareAndTokenise(global_pretoken), tk2.prepareAndTokenise(global_pretoken))
        self._sum_jaccard += J
        self._n_total     += 1

        self._sum_intersection += num
        self._sum_union        += denom

    def _compute(self) -> R:
        return self._sum_intersection/self._sum_union if self._sum_union else 1, \
               self._sum_jaccard/self._n_total if self._n_total else 1


def jaccard(tokens1: List[str], tokens2: List[str]) -> Tuple[float,int,int]:
    """
    Generalised Jaccard similarity, which is based on multi-sets.
    https://en.wikipedia.org/wiki/Jaccard_index#Weighted_Jaccard_similarity_and_distance
    If a sentence is tokenised as A, B, C, D, A and A, A, B, C, E, A then the Jaccard similarity is:

    |{A: 2, B: 1, C: 1}| / |{A: 3, B: 1, C: 1, D: 1, E: 1}|

    The numerator obviously contains the overlap in the two sets, accounting for multiplicities.

    Note that rather than containing the sum count of each element in the sets, the denominator contains the larger of the two.
    If one has 2 As and the other has 3 As, you could say that either there are 4 As that match out of 5 As total, but
    by this logic, the traditional Jaccard similarity would count every element twice when taking the intersection and
    the union. That's not how it works.
    """
    c1 = Counter(tokens1)
    c2 = Counter(tokens2)
    domain = set(c1.keys()) | set(c2.keys())

    numerator   = 0
    denominator = 0
    for t in domain:
        numerator   += min(c1[t], c2[t])
        denominator += max(c1[t], c2[t])

    return numerator/denominator if denominator else 1, numerator, denominator


class MicroMacroMultiplicityRatio(ComparisonMetric[Tuple[float, float, float, float]]):
    """
    Rather than computing similarity of the tokens, computes similarity of the amount of tokens produced by the given tokenisers
    by computing their ratio. This ratio has also been called "parity" in literature.

    Reports results for both tokeniser 1 over tokeniser 2 and tokeniser 2 over tokeniser 1.
    """

    def _reset(self):
        self._sum_m1 = 0
        self._sum_m2 = 0
        self._sum_m1_on_m2 = 0
        self._sum_m2_on_m1 = 0
        self._n_total = 0

    def _add(self, global_pretoken: str, tk1: Tokeniser, tk2: Tokeniser):
        m1 = len(tk1.prepareAndTokenise(global_pretoken))
        m2 = len(tk2.prepareAndTokenise(global_pretoken))
        if m1 == 0 or m2 == 0:
            case = ("1" if m1 == 0 else "") + (" and " if m1 == 0 and m2 == 0 else "") + ("2" if m2 == 0 else "")
            warn(f"Tokeniser {case} produced no tokens for pretoken '{global_pretoken}'. Skipping it.")
            return

        self._n_total += 1
        self._sum_m1 += m1
        self._sum_m2 += m2
        self._sum_m1_on_m2 += m1/m2
        self._sum_m2_on_m1 += m2/m1

    def _compute(self):
        return (self._sum_m1/self._sum_m2        if self._sum_m2 else float("inf") if self._sum_m1 else 1.0,
                self._sum_m1_on_m2/self._n_total if self._n_total else 1.0,
                self._sum_m2/self._sum_m1        if self._sum_m1 else float("inf") if self._sum_m2 else 1.0,
                self._sum_m2_on_m1/self._n_total if self._n_total else 1.0)
