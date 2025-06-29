"""
Evaluations that generate distributions, either across the vocabulary or the domain of segmentations.
"""
from typing import Iterable, Tuple, Dict, Optional, List, TypeVar, Union
from dataclasses import dataclass
from collections import Counter

from math import ceil
import numpy as np
from scipy.stats import entropy as _scipy_entropy

from ..interfaces.tokeniser import TokeniserWithFiniteTypeDomain, Tokeniser
from ..util.iterables import streamProgress
from ..util.combinatorics import getBitKey
from ..util.dicts import argmax, normaliseCounter
from ..util.types import NamedIterable

SHANNON_RENYI_ALPHA = 1.0
DEFAULT_RENYI_ALPHA = 2.5
T = TypeVar("T")


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

    normalisation_constant = probabilities.sum()
    if normalisation_constant != 0:
        probabilities = probabilities / normalisation_constant

    if alpha == 1.0:
        return shannonEntropy(probabilities)

    return 1/(1-alpha)*np.log2(np.sum(probabilities ** alpha))


def renyiEfficiency(probabilities: Iterable[float], alpha: float=DEFAULT_RENYI_ALPHA,
                    domain_size: int=None, sample_size: int=None) -> Tuple[float,float,float]:
    """
    Rényi efficiency of a probability distribution equals the fraction which its Rényi entropy is of the Shannon entropy
    of a uniform distribution of the same size.

    At alpha = 1.0, you have Shannon efficiency.
    At alpha = 2.5, if the domain of the distribution is a vocabulary of token types, then Rényi efficiency correlates
    highly with downstream performance according to https://aclanthology.org/2023.acl-long.284v2.pdf.

    This function computes three quantities from that paper:
        - Simplified lower bound:         H_alpha / ceil(H_0)   (by analogy of theorem 4.5 to theorem 3.9; see below)
        - Simplified fraction in between: H_alpha / H_0
        - Simplified upper bound:   ceil(H_alpha) / H_0   (simplified equation 21)

    There are two approximations at play here:
        - There is no explicit formula for Rényi efficiency (it's based on a ratio of two limits, see equation 12), so
          instead, bounds are used, which can be expressed explicitly based on Rényi entropy.
        - The full expressions for these bounds contain a covariance term in the numerator. This is said to be small and
          negative by the authors, so the upper bound is slightly higher than it needs to be.
          For the lower bound, analogising the theorems is correct according to Vilém Zouhar, but leaving out the
          covariance term is not (it is unknown in size and can be negative). Hence, beware that the lower bound
          should not be treated as the true lower bound but rather as a new, arbitrary, related metric.

    The middle fraction is how efficiency is defined traditionally, and lies between the lower and upper bound.

    :param domain_size: The amount of values that could theoretically be emitted by the stochastic variable whose
                        distribution is being tested. If not given, the amount of probabilities given will be used.
                        Note: if the given probabilities are a distribution over segmentations rather than over tokens,
                        you definitely want to set this argument, lest a deterministic tokeniser get an efficiency of 1.
    :param sample_size: If the probabilities are sample proportions from a finite amount of samples, the maximal
                           entropy achievable is not just log(domain size) but rather log(min(domain size, samples)).
                           If the domain has 10000 values but you only took 10 samples, the maximal entropy you could
                           get is not the uniform distribution [1/10000]*10000 but rather [1/10, ..., 1/10, 0, 0, ..., 0].
    """
    probabilities = np.array(list(probabilities))  # You need this list() because numpy has weird behaviour for e.g. dict.values().
    if domain_size is None:
        domain_size = probabilities.size  # Default is as small as possible, i.e. the given distribution and no more.
    assert domain_size >= probabilities.size, f"Domain size ({domain_size}) can't be manually set to be lower than the amount of probabilities given ({probabilities.size})."
    if sample_size is None:
        sample_size = domain_size  # Default is as large as possible, i.e. you assume we've had enough samples to theoretically spread uniformly over the entire domain.
    assert sample_size >= 0  # TODO: Technically should be "the amount of non-zero probabilities", not zero.

    if domain_size == 0:    # A variable with an empty domain produces values that are trivially as entropic as they can be. (How you sample it is a mystery.)
        return 1.0, 1.0, 1.0
        # return 0.0, 0.0, 0.0
    elif sample_size == 0:  # A variable that doesn't generate anything is perfectly predictable and hence has no entropy.
        return 0.0, 0.0, 0.0
    elif domain_size == 1:  # A variable with a domain of 1 possible value which we know has been sampled more than zero times, is trivially uniform.
        return 1.0, 1.0, 1.0
        # return 0.0, 0.0, 0.0
    elif sample_size == 1:  # A variable with a domain larger than 1 isn't just trivially uniform when only one sample exists for it. Adding one sample with the same value makes it the least entropic possible. Adding one sample with a different value makes it uniform. So, what is it now? You can't really say that 1 sample lacks uniformity, but you also can't say it lacks determinism. It has both. I argue that the result should stay the same if you multiply the sample size by any number, so this has Shannon entropy 0 in a domain with non-zero maximal entropy H_0.
        return 0.0, 0.0, 0.0

    H_a = renyiEntropy(probabilities, alpha=alpha)
    H_0 = np.log2(min(domain_size,sample_size))  # Maximal entropy possible given both the domain size and samples.
    return H_a/ceil(H_0), H_a/H_0, ceil(H_a)/H_0


########################################################################################################################


@dataclass
class TokenDiversity:
    V: int    # Amount of types in the corpus. In other words, the effective vocabulary size.
    ctc: int  # Amount of tokens in the corpus.

    @property
    def ttr(self):
        """Type-token ratio."""
        return self.V/self.ctc

    mattr: int
    mattr_window_size: int
    mattr_stride: int

    renyi_entropy: float
    renyi_efficiency: float
    corpus_name: str


def getTokenDistributionFromSentences_and_analyse(
        tokeniser: TokeniserWithFiniteTypeDomain, corpus: NamedIterable[str],
        # RE arguments
        renyi_alpha: float=DEFAULT_RENYI_ALPHA, renyi_efficiency_with_effective_vocab: bool=True,
        # MATTR arguments
        mattr_window_size: int=2000, mattr_stride: int=1000, mattr_flush_every_example: bool=False
    ) -> Tuple[Counter[str], TokenDiversity]:
    """
    Computes, for each type that a tokeniser can produce, the fraction of produced tokens that belong to that type,
    a.k.a. the token unigram distribution.
    (A much more elaborate and caching version of this function can be found in the bpe_hell.eda.computation package.)

    While doing so, it also computes several summary statistics about this distribution, among which the Rényi efficiency
    of the vocabulary and a generalised version of the moving-average type-token ratio (MATTR).
    Covington & McFall (2010, https://www.tandfonline.com/doi/full/10.1080/09296171003643098) discuss MATTR for a window
    stride of 1. This function implements it for any stride.

    We cannot use a ChainedCounter like we otherwise would to average something across counters produced for fixed-size
    windows. The reason is that in the BEST case, each subcounter will have only 1 entry with count window_size, and the
    amount of subcounters you will have is corpus_size/window_size. Say you have 1 billion tokens (5 GiB of English data,
    say), then a window size of 500 tokens will produce 2 million subcounters. That's very steep scaling.

    :param flush_window_every_example: If False, examples in the corpus will be treated as one large document and windows
                                       can spill from one into the other. The TTR inside a window will be higher if it
                                       contains tokens from more than one document, especially if they are on different topics.
    """
    assert 0 < mattr_stride <= mattr_window_size
    n_TTR   = 0
    sum_TTR = 0
    global_type_frequencies = Counter()  # Will be returned.

    window_type_frequencies = Counter()
    n_tokens_in_window      = 0
    stride_history: List[Counter] = [Counter()]
    n_tokens_in_last_stride       = 0
    for sentence in corpus:
        # Type distribution takes two lines of code:
        tokens = tokeniser.prepareAndTokenise(sentence)
        global_type_frequencies.update(tokens)

        # MATTR is a lot more:
        #  1. Try to fill up the last stride.
        deficit = mattr_stride - n_tokens_in_last_stride
        tokens_remaining = tokens
        n_remaining_tokens = len(tokens_remaining)
        while n_remaining_tokens >= deficit:  # While you can create new strides.
            tokens_added, tokens_remaining = tokens_remaining[:deficit], tokens_remaining[deficit:]
            n_remaining_tokens -= deficit

            stride_history[-1].update(tokens_added)
            stride_history.append(Counter())

            n_tokens_in_last_stride = 0
            deficit = mattr_stride

        stride_history[-1].update(tokens_remaining)
        n_tokens_in_last_stride += n_remaining_tokens

        #  2. Separately, try to fill up the window. If enough tokens have accumulated, get TTR and subtract the oldest stride.
        deficit = mattr_window_size - n_tokens_in_window
        tokens_remaining = tokens
        n_remaining_tokens = len(tokens_remaining)
        while n_remaining_tokens >= deficit:  # While you can fill the window. This never happens when there isn't at least one stride available.
            tokens_added, tokens_remaining = tokens_remaining[:deficit], tokens_remaining[deficit:]
            n_remaining_tokens -= deficit

            # Fill the window, compute stats on the full window
            window_type_frequencies.update(tokens_added)
            n_tokens_in_window += deficit
            sum_TTR += len(window_type_frequencies) / n_tokens_in_window  # Here n_tokens_in_window is always window_size.
            n_TTR   += 1

            # Shrink the window
            window_type_frequencies -= stride_history.pop(0)  # In the extreme case, this stride has JUST been formed in step 1.
            # for t in subtracted_counts.keys():
            #     if window_type_frequencies[t] == 0:
            #         window_type_frequencies.pop(t)  # As it turns out, this filtering happens automatically in a Counter -= Counter, but NOT after Counter[key] -= value.

            n_tokens_in_window -= mattr_stride
            deficit = mattr_stride

        window_type_frequencies.update(tokens_remaining)
        n_tokens_in_window += n_remaining_tokens

        #  3. Optionally: if you can't move windows across examples, commit now.
        if mattr_flush_every_example and n_remaining_tokens > 0:
            sum_TTR += len(window_type_frequencies) / n_tokens_in_window
            n_TTR   += n_tokens_in_window / mattr_window_size

            window_type_frequencies = Counter()
            n_tokens_in_window      = 0
            stride_history: List[Counter] = [Counter()]
            n_tokens_in_last_stride       = 0

    # The last half-filled window should be counted too unless it already was. You know it already was if the deficit is exactly 1 stride AND the window has already moved.
    if not mattr_flush_every_example and not(mattr_window_size - n_tokens_in_window == mattr_stride and n_TTR > 0):
        sum_TTR += len(window_type_frequencies) / n_tokens_in_window
        n_TTR   += n_tokens_in_window / mattr_window_size

    # Part 2: Analysis
    effective_vocabulary_size = len(global_type_frequencies)
    renyi_entropy          = renyiEntropy(global_type_frequencies.values(), alpha=renyi_alpha)
    _, renyi_efficiency, _ = renyiEfficiency(global_type_frequencies.values(), alpha=renyi_alpha,
                                             domain_size=effective_vocabulary_size if renyi_efficiency_with_effective_vocab else tokeniser.getVocabSize(),
                                             sample_size=global_type_frequencies.total())

    # Include the rest in the distribution
    for t in tokeniser.types():
        global_type_frequencies[t] += 0

    return global_type_frequencies, TokenDiversity(
        V=effective_vocabulary_size,
        ctc=global_type_frequencies.total(),

        mattr=sum_TTR / n_TTR,
        mattr_window_size=mattr_window_size,
        mattr_stride=mattr_stride,

        renyi_entropy=renyi_entropy,
        renyi_efficiency=renyi_efficiency,

        corpus_name=corpus.name
    )


def bitKeyFromTokens(tokens: List[str]) -> int:
    return getBitKey(list(map(len, tokens)))


SegmentationCounts = Counter[int]  # Maps segmentation ID to its count.

def segmentationDistributionFromWord(tokeniser: Tokeniser, word: str, n_samples: int) -> SegmentationCounts:
    """Repeatedly segments a word to get a (unnormalised) distribution across its segmentations."""
    segmentations = Counter()
    for _ in streamProgress(range(n_samples), f"Tokenising: {word}", known_size=n_samples):
        segmentations[bitKeyFromTokens(tokeniser.prepareAndTokenise(word))] += 1
    return segmentations


@dataclass
class SegmentationDiversity:
    uniqueness: float  # Fraction of segmentations that remain when you filter out duplicates. ~precision
    coverage: float  # Fraction of possible segmentations that have been produced. ~recall
    max_coverage_uniqueness: float  # Equals uniqueness if you take less than the possible segmentations in samples, otherwise equals coverage. Equivalent to max(U,C).

    regularisation_rate_argmax: float  # Fraction of segmentations that AREN'T the most common one.
    regularisation_rate_deterministic: Optional[float]  # Fraction of segmentations that AREN'T the given one.

    efficiency_all: float  # Fraction that the actual Shannon entropy of the segmentation distribution is of the highest possible Shannon entropy it could be.
    efficiency_no_argmax: float  # Same, but for the distribution that doesn't include the most common segmentation.
    efficiency_no_deterministic: Optional[float]  # Same, but for the given segmentation.


def analyseSegmentationDistribution(segmentations_counts: SegmentationCounts,
                                    sample_size: int, domain_size: int, renyi_alpha: float=1.0,
                                    deterministic_segmentation: Optional[List[str]]=None) -> SegmentationDiversity:
    segmentation_probabilities = normaliseCounter(segmentations_counts)

    uniqueness = len(segmentation_probabilities) / sample_size
    coverage   = len(segmentation_probabilities) / domain_size
    max_coverage_uniqueness = len(segmentation_probabilities) / min(sample_size,domain_size)  # Equals max(coverage,uniqueness).
    _, entropic_efficiency_all, _ = renyiEfficiency(segmentation_probabilities.values(), alpha=renyi_alpha, domain_size=domain_size, sample_size=sample_size)

    # Compute statistics for the distribution without its mode.
    argmax_index = argmax(segmentation_probabilities)[0]
    max_probability = segmentation_probabilities[argmax_index]
    regularisation_rate_argmax = 1 - max_probability

    segmentation_probabilities.pop(argmax_index)  # The renyiEfficiency() call will re-normalise automatically. It is not an issue that the count distribution was already normalised before, because normalise-pop-renormalise is mathematically equivalent to pop-normalise.
    _, entropic_efficiency_no_argmax, _ = renyiEfficiency(segmentation_probabilities.values(), alpha=renyi_alpha, domain_size=domain_size-1, sample_size=round(sample_size*regularisation_rate_argmax))
    segmentation_probabilities[argmax_index] = max_probability  # Since we normalised a copy, segmentation_probabilities is already normalised again.

    # Compute statistics for the distribution without the given deterministic segmentation.
    if deterministic_segmentation is not None:
        det_index = bitKeyFromTokens(deterministic_segmentation)
        det_probability = segmentation_probabilities[det_index]
        regularisation_rate_deterministic = 1 - det_probability

        segmentation_probabilities.pop(det_index)
        _, entropic_efficiency_no_deterministic, _ = renyiEfficiency(segmentation_probabilities.values(), alpha=renyi_alpha, domain_size=domain_size-1, sample_size=round(sample_size*regularisation_rate_deterministic))
        segmentation_probabilities[det_index] = det_probability
    else:
        regularisation_rate_deterministic    = None
        entropic_efficiency_no_deterministic = None

    return SegmentationDiversity(
        coverage=coverage,
        uniqueness=uniqueness,
        max_coverage_uniqueness=max_coverage_uniqueness,

        regularisation_rate_argmax=regularisation_rate_argmax,
        regularisation_rate_deterministic=regularisation_rate_deterministic,

        efficiency_all=entropic_efficiency_all,
        efficiency_no_argmax=entropic_efficiency_no_argmax,
        efficiency_no_deterministic=entropic_efficiency_no_deterministic
    )
