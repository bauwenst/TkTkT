from typing import Tuple, Iterable, List
from collections import Counter

from ..interfaces.tokeniser import Tokeniser


def exactMatches(texts: Iterable[str], tk1: Tokeniser, tk2: Tokeniser) -> Tuple[float,int,int]:
    """
    Count how many out of a given amount of texts is tokenised into exactly the same strings by the two tokenisers.
    """
    n_matches = 0
    n_total   = 0
    for text in texts:
        n_matches += tk1.prepareAndTokenise(text) == tk2.prepareAndTokenise(text)
        n_total   += 1

    return n_matches/n_total if n_total else 1, n_matches, n_total


def microMacroTokenJaccard(texts: Iterable[str], tk1: Tokeniser, tk2: Tokeniser) -> Tuple[float, float]:
    sum_intersection = 0
    sum_union        = 0
    sum_jaccard = 0
    n_total     = 0
    for text in texts:
        J, num, denom = jaccard(tk1.prepareAndTokenise(text), tk2.prepareAndTokenise(text))
        sum_jaccard += J
        n_total += 1

        sum_intersection += num
        sum_union        += denom

    return sum_intersection/sum_union if sum_union else 1, \
           sum_jaccard/n_total if n_total else 1


def jaccard(tokens1: List[str], tokens2: List[str]) -> Tuple[float,int,int]:
    """
    Generalised Jaccard similarity, which is based on multi-sets.
    https://en.wikipedia.org/wiki/Jaccard_index#Weighted_Jaccard_similarity_and_distance
    If a sentence is tokenised as A, B, C, D, A and A, A, B, C, E, A then the Jaccard similarity is:

    |{A: 2, B: 1, C: 1}| / |{A: 3, B: 1, C: 1, D: 1, E: 1}|

    The numerator obviously contains the overlap in the two sets,

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
