"""
Evaluation of the context around tokens.
"""
from typing import Iterable, Dict, Set, Union, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import numpy as np

from ..factories.preprocessing import IdentityPreprocessor
from ..interfaces.tokeniser import Tokeniser, Preprocessor
from ..util.iterables import streamProgress
from .entropy import renyiEfficiency


VocabRef = int  # To avoid storing token strings over and over, we construct a vocab on-the-fly (even with tokenisers that have no vocab).
WordOrSentenceIterable = Union[Iterable[str], Iterable[Tuple[str,int]]]
def getIterableWithCounts(iterable: WordOrSentenceIterable) -> Iterable[Tuple[str,int]]:
    for thing in iterable:
        if isinstance(thing, tuple):
            word, frequency = thing
        else:
            word = thing
            frequency = 1
        yield word, frequency


@dataclass
class AccessorDistributions:
    vocab: Dict[str,VocabRef]
    left_of:  Dict[VocabRef, Counter[VocabRef]]
    right_of: Dict[VocabRef, Counter[VocabRef]]


@dataclass
class TypeAccessorSummary:  # Looks a lot like the SegmentationDiversity dataclass in TkTkT's entropy module.
    total_accessors: int=0  # Amount of non-unique accessors.
    av: int=0               # Amount of unique accessors.
    coverage: float=0       # Fraction which AV makes up of all possible unique accessors.
    uniqueness: float=0     # Fraction of accessors that are unique. Basically, TTR for the set of accessors of this type.
    entropic_efficiency: float=0


@dataclass
class DistributionAccessorSummaries:
    per_type: Dict[str, TypeAccessorSummary]
    averages: TypeAccessorSummary
    weighted_averages: TypeAccessorSummary


@dataclass
class AllAccessorSummaries:
    left:  DistributionAccessorSummaries
    right: DistributionAccessorSummaries
    both:  DistributionAccessorSummaries
    min:   DistributionAccessorSummaries  # For each type separately, picks the accessor distribution with the fewest types (i.e. the most predictable side) and copies its metrics.


def getAccessors(tokeniser: Tokeniser, texts: WordOrSentenceIterable, split_into_disjunct_examples: Preprocessor=None) -> AccessorDistributions:
    if split_into_disjunct_examples is None:
        split_into_disjunct_examples = IdentityPreprocessor()

    max_id: VocabRef = 0
    vocab: Dict[str,VocabRef] = dict()

    # Everything you have seen to the left and right of a given type.
    left_of:  Dict[VocabRef, Counter[VocabRef]] = defaultdict(Counter)
    right_of: Dict[VocabRef, Counter[VocabRef]] = defaultdict(Counter)

    for text,frequency in getIterableWithCounts(texts):
        for bounded_text in split_into_disjunct_examples.do(text):
            tokens = tokeniser.prepareAndTokenise(bounded_text)
            ids = []
            for token in tokens:
                try:
                    ids.append(vocab[token])
                except:
                    vocab[token] = max_id
                    max_id += 1
                    ids.append(vocab[token])

            # Edge tokens
            # - When a type appears at the start/end of an example, it probably still has an accessor to its left/right
            #   in reality, but we can't see it because the example is only an excerpt.
            # - An upper estimate on accessor variety is to always consider these unknown edges to be unique accessors.
            # - If you are studying morphology, you probably want to have an edge around every word, because it matters
            #   much less in such cases what the exact type was that came before (there is no connection between the
            #   characters of the previous word and of the current word, only the meanings).
            # - The ID for this start/end is -1 and doesn't appear in the vocabulary.
            left_of[ids[0]][-1]   += frequency
            right_of[ids[-1]][-1] += frequency

            if len(ids) > 1:
                right_of[ids[0]][ids[1]]  += frequency  # Has no token to the left
                left_of[ids[-1]][ids[-2]] += frequency  # Has no token to the right

            # Middle tokens
            for i in range(1,len(ids)-1):
                center = ids[i]
                left_of[center][ids[i-1]]  += frequency
                right_of[center][ids[i+1]] += frequency

    # Make sure all IDs have a -1 entry.
    for i in vocab.values():
        left_of[i][-1]  += 0
        right_of[i][-1] += 0

    return AccessorDistributions(vocab, left_of, right_of)


def analyseAccessors(accessors: AccessorDistributions, do_count_ends: bool=True, predefined_vocab_size: Optional[int]=None) -> AllAccessorSummaries:
    """
    :param do_count_ends: Whether to pretend that every start/end of an example should've been counted as a unique type.
                          The longer your examples were, the less this matters.
    """
    vocab, left_of, right_of = accessors.vocab, accessors.left_of, accessors.right_of
    # reverse_vocab = {i: t for t,i in vocab.items()}  # Not really needed when you loop over the vocab items.
    # all_ids = set(left_of) | set(right_of)  # Should be equivalent to reverse_vocab.keys()

    summaries = AllAccessorSummaries(
        left=DistributionAccessorSummaries(
            per_type=defaultdict(TypeAccessorSummary),
            averages         =TypeAccessorSummary(),
            weighted_averages=TypeAccessorSummary()
        ),
        right=DistributionAccessorSummaries(
            per_type=defaultdict(TypeAccessorSummary),
            averages         =TypeAccessorSummary(),
            weighted_averages=TypeAccessorSummary()
        ),
        both=DistributionAccessorSummaries(
            per_type=defaultdict(TypeAccessorSummary),
            averages         =TypeAccessorSummary(),
            weighted_averages=TypeAccessorSummary()
        ),
        min=DistributionAccessorSummaries(
            per_type=defaultdict(TypeAccessorSummary),
            averages         =TypeAccessorSummary(),
            weighted_averages=TypeAccessorSummary()
        )
    )

    def fillTypeSummary(summary: TypeAccessorSummary, accessor_counts: Counter, end_count: int,
                        possible_accessors: int):
        nonend_count = accessor_counts.total()
        summary.total_accessors           = nonend_count + end_count
        summary.av                        = len(accessor_counts) + do_count_ends * end_count
        summary.coverage                  = len(accessor_counts) / possible_accessors
        summary.uniqueness                = len(accessor_counts) / nonend_count
        _, summary.entropic_efficiency, _ = renyiEfficiency(accessor_counts.values(), domain_size=possible_accessors, sample_size=accessor_counts.total(), alpha=1.0)

    for t,i in streamProgress(vocab.items(), known_size=len(vocab), show_as="Computing type statistics"):
        # Per-type statistics
        #   - First, remove the special -1 entry from the type's distribution.
        left_ends  = left_of[i].pop(-1)
        right_ends = right_of[i].pop(-1)
        both_ends  = left_ends + right_ends

        fillTypeSummary(summaries.left.per_type[t],  left_of[i],               left_ends,  predefined_vocab_size or len(right_of))  # About the last argument: we want to know how many possible types could appear LEFT OF type i, which is equivalent to asking how many possible types have anything RIGHT OF themselves.
        fillTypeSummary(summaries.right.per_type[t], right_of[i],              right_ends, predefined_vocab_size or len(left_of))
        fillTypeSummary(summaries.both.per_type[t],  left_of[i] + right_of[i], both_ends,  predefined_vocab_size or len(set(left_of) | set(right_of)))

        if summaries.left.per_type[t].av < summaries.right.per_type[t].av:
            summaries.min.per_type[t] = summaries.left.per_type[t]
        else:
            summaries.min.per_type[t] = summaries.right.per_type[t]

    # Averages
    def fillAverages(distribution_summaries: DistributionAccessorSummaries):
        summaries_to_average = list(distribution_summaries.per_type.values())
        weights = [s.total_accessors for s in summaries_to_average]

        distribution_summaries.averages.total_accessors, distribution_summaries.weighted_averages.total_accessors = \
            getMeanAndWeightedMean([s.total_accessors     for s in summaries_to_average],weights)
        distribution_summaries.averages.av, distribution_summaries.weighted_averages.av = \
            getMeanAndWeightedMean([s.av                  for s in summaries_to_average],weights)
        distribution_summaries.averages.coverage, distribution_summaries.weighted_averages.coverage = \
            getMeanAndWeightedMean([s.coverage            for s in summaries_to_average], weights)
        distribution_summaries.averages.uniqueness, distribution_summaries.weighted_averages.uniqueness = \
            getMeanAndWeightedMean([s.uniqueness          for s in summaries_to_average], weights)
        distribution_summaries.averages.entropic_efficiency, distribution_summaries.weighted_averages.entropic_efficiency = \
            getMeanAndWeightedMean([s.entropic_efficiency for s in summaries_to_average], weights)

    fillAverages(summaries.left)
    fillAverages(summaries.right)
    fillAverages(summaries.both)
    fillAverages(summaries.min)

    return summaries


def getMeanAndWeightedMean(values: list, weights: list) -> Tuple[float,float]:
    values  = np.array(values)
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    return float(np.mean(values)), float(np.sum(weights * values))  # np.mean is equivalent to np.sum(1/n * values).
