from math import ceil
from typing import Iterable, Tuple, Dict
from collections import Counter

import numpy as np
from scipy.stats import entropy as _scipy_entropy

from ..interfaces.tokeniser import TokeniserWithFiniteTypeDomain
from ..util.iterables import streamProgress

DEFAULT_RENYI_ALPHA = 2.5


def shannonEntropy(probabilities: Iterable[float]):
    # return -np.sum(probabilities * np.log2(probabilities))
    return _scipy_entropy(pk=list(probabilities), base=2)  # Deals with p_i == 0.


def renyiEntropy(probabilities: Iterable[float], alpha: float=DEFAULT_RENYI_ALPHA):
    """
    Computes Rényi entropy, a generalisation of Shannon entropy.
    https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy

    Shannon entropy appears for alpha = 1.0, which causes L'Hôpital's rule to be applied and turns log(sum) into sum(log).

    Sometimes the case alpha = 2.0 is called "the" Rényi entropy, equivalent to the negative log probability of two
    tokens being equal if both are sampled at random. However, to be fully correct, you should always use "Rényi at alpha = ...".
    For evaluating tokenisers, alpha = 2.5 has been found to be best (when used for efficiency, not entropy).
    https://aclanthology.org/2023.acl-long.284v2.pdf
    """
    probabilities = np.array(list(probabilities))
    if alpha == 1.0:
        return shannonEntropy(probabilities)

    return 1/(1-alpha)*np.log2(np.sum(probabilities ** alpha))


def renyiEfficiency(probabilities: Iterable[float], alpha: float=DEFAULT_RENYI_ALPHA) -> Tuple[float,float,float]:
    """
    Rényi efficiency of a token distribution equals the fraction which its Rényi entropy is of the Shannon entropy of a
    uniform distribution of the same size. At alpha = 2.5, this fraction correlates highly with downstream performance.

    This function computes three quantities according to https://aclanthology.org/2023.acl-long.284v2.pdf:
        - Simplified lower bound:         H_alpha / ceil(H_0)   (by analogy of theorem 4.5 to theorem 3.9)
        - Simplified fraction in between: H_alpha / H_0
        - Simplified upper bound:   ceil(H_alpha) / H_0   (simplified equation 21)

    There are two approximations at play here:
        - There is no explicit formula for Rényi efficiency (it's based on a ratio of two limits, see equation 12), so
          instead, bounds are used, which can be expressed explicitly based on Rényi entropy.
        - The full expressions for these bounds contain a covariance term in the numerator. This is said to be small and
          negative by the authors, so the upper bound is slightly higher than it needs to be.

    The middle fraction doesn't mean anything, but it is between the lower and upper bound.
    """
    probabilities = np.array(list(probabilities))  # You need this list() because numpy has weird behaviour for e.g. dict.values().
    V = probabilities.size
    H_a = renyiEntropy(probabilities, alpha=alpha)
    H_0 = np.log2(V)
    return H_a/ceil(H_0), H_a/H_0, ceil(H_a)/H_0


########################################################################################################################


def tokenDistributionFromSentences(tokeniser: TokeniserWithFiniteTypeDomain, corpus: Iterable[str]) -> Dict[str,float]:
    """
    For each type that a tokeniser can produce, compute the fraction of produced tokens that belong to that type.
    (A much more elaborate and caching version of this function can be found in the bpe_inversion.eda.computation package.)
    """
    type_frequencies = Counter()
    for t in tokeniser.types():
        type_frequencies[t] = 0

    for sentence in streamProgress(corpus, "Tokenising"):
        tokens = tokeniser.prepareAndTokenise(sentence)
        for t in tokens:
            type_frequencies[t] += 1

    return normaliseCounter(type_frequencies)


def normaliseCounter(counts: Counter) -> dict:
    total = counts.total()
    return {t: c/total for t,c in counts.items()}
