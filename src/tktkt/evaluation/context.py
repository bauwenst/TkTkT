"""
Evaluation of the context around tokens.
"""
from typing import Iterable, Dict, Set, Union, Tuple, Optional, Callable, List
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, Counter

import scipy as sp
import numpy as np
import json
import csv
import re

from ..paths import TkTkTPaths
from ..factories.preprocessing import IdentityPreprocessor
from ..interfaces.tokeniser import Tokeniser, Preprocessor
from ..util.iterables import streamProgress, at
from ..util.timing import datetimeDashed
from ..util.dicts import ChainedCounter
from ..util.types import NamedIterable
from .entropy import renyiEfficiency


VocabRef = int  # To avoid storing token strings over and over, we construct a vocab on-the-fly (even with tokenisers that have no vocab).

def getIterableWithCounts(iterable: Union[Iterable[str], Iterable[Tuple[str,int]]]) -> Iterable[Tuple[str,int]]:
    for thing in iterable:
        if isinstance(thing, tuple):
            word, frequency = thing
        else:
            word = thing
            frequency = 1
        yield word, frequency


@dataclass
class AccessorDistribution:
    accessors:  Dict[VocabRef, ChainedCounter[VocabRef]]
    boundaries: Dict[VocabRef, int]

    def countOccurrences(self, accessor_id: VocabRef) -> int:
        return self.accessors.get(accessor_id, Counter()).total() + self.boundaries.get(accessor_id, 0)

    def remove(self, accessor_id: VocabRef):
        # Remove as thing that (possibly) has neighbours and (possibly) has boundaries
        if accessor_id in self.accessors:
            self.accessors.pop(accessor_id)
        if accessor_id in self.boundaries:
            self.boundaries.pop(accessor_id)

        # Remove as thing that is (possibly) a neighbour
        for _, neighbours in self.accessors.items():
            if accessor_id in neighbours:
                neighbours.pop(accessor_id)

    def defragment(self):
        # Make all the counters nice again.
        for _, neighbours in streamProgress(self.accessors.items(), show_as="Defragmenting types", known_size=len(self.accessors)):
            neighbours.defragment()


@dataclass
class AccessorDistributions:
    vocab: Dict[str,VocabRef]
    left_of:  AccessorDistribution
    right_of: AccessorDistribution

    # Serialisation logic below.
    corpus_name: str

    def save(self) -> Path:
        def serialiseDistribution(distribution: AccessorDistribution):
            accessors = []
            for t1, counter in distribution.accessors.items():
                subcounters, max_size = counter.serialise()
                subcounters = [[(t2,count) for t2, count in subcounter.items()] for subcounter in subcounters]
                accessors.append([t1, max_size, subcounters])

            return {
                "boundaries": [(t,c) for t,c in distribution.boundaries.items()],
                "accessors": accessors
            }

        folder = TkTkTPaths.append(TkTkTPaths.pathToEvaluations(), "av")
        file = folder / f"{self.corpus_name}_{datetimeDashed()}.json"
        data = {
            "source": self.corpus_name,
            "vocab": self.vocab,
            "left": serialiseDistribution(self.left_of),
            "right": serialiseDistribution(self.right_of)
        }
        serialised = json.dumps(data, indent=2, ensure_ascii=False)
        serialised = re.compile(r"\[\s+([0-9]+),\s+([0-9]+)\s+\]").sub(r"[\1,\2]", serialised)
        with open(file, "w", encoding="utf-8") as handle:
            handle.write(serialised)
        return file

    @classmethod
    def load(cls, file: Path) -> "AccessorDistributions":
        with open(file, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        distributions = AccessorDistributions(
            corpus_name=data["source"],
            vocab=data["vocab"],
            left_of=AccessorDistribution(
                accessors=dict(),
                boundaries=dict()
            ),
            right_of=AccessorDistribution(
                accessors=dict(),
                boundaries=dict()
            )
        )
        for t1, max_size, subcounters in data["left"]["accessors"]:
            distributions.left_of.accessors[t1] = ChainedCounter(max_size, seed=0)
            for subcounter in subcounters:
                for t2, count in subcounter:
                    distributions.left_of.accessors[t1][t2] += count

        for t1, max_size, subcounters in data["right"]["accessors"]:
            distributions.right_of.accessors[t1] = ChainedCounter(max_size, seed=0)
            for subcounter in subcounters:
                for t2, count in subcounter:
                    distributions.right_of.accessors[t1][t2] += count

        for t, count in data["left"]["boundaries"]:
            distributions.left_of.boundaries[t] = count

        for t, count in data["right"]["boundaries"]:
            distributions.right_of.boundaries[t] = count

        return distributions


    def filter(self, remove_if: Callable[[str, "AccessorDistributions"], bool]) -> List[str]:
        # Remove
        removed = [t for t in self.vocab if remove_if(t, self)]
        for t in streamProgress(removed, show_as="Removing types"):
            self.remove(t)

        # Restore
        self.defragment()
        return removed

    def remove(self, accessor: str):
        vocab_ref = self.vocab.pop(accessor)  # Raises error if accessor doesn't exist.
        self.left_of.remove(vocab_ref)
        self.right_of.remove(vocab_ref)

    def defragment(self):
        self.left_of.defragment()
        self.right_of.defragment()


@dataclass
class TypeAccessorSummary:  # Looks a lot like the SegmentationDiversity dataclass in TkTkT's entropy module.
    """
    Summarises the accessor distribution of one type.
    """
    total_accessors: int=0  # Amount of non-unique accessors. This is equivalent to the frequency of the type.
    boundary_ratio: float=0.0   # Fraction of accessors that are pretoken boundaries.

    av: int=0               # Amount of unique accessors. Note: this is slightly different from how it is defined in the AV paper. If you want to reproduce that AV, you'll need an infinite bucket size and then add total_accessors*boundary_ratio to this number here.

    coverage: float=0       # Fraction which AV makes up of all possible unique accessors.
    uniqueness: float=0     # Fraction of accessors that are unique. Basically, TTR for the set of accessors of this type.
    mcu: float=0

    entropic_efficiency: float=0


@dataclass
class MetaSummaries:
    """
    Summarises the summaries of the accessor distributions of all types in the vocabulary.
    """
    mean:            TypeAccessorSummary = field(default_factory=TypeAccessorSummary)
    mean_weighted:   TypeAccessorSummary = field(default_factory=TypeAccessorSummary)

    median:          TypeAccessorSummary = field(default_factory=TypeAccessorSummary)
    median_weighted: TypeAccessorSummary = field(default_factory=TypeAccessorSummary)

    mad:             TypeAccessorSummary = field(default_factory=TypeAccessorSummary)
    mad_weighted:    TypeAccessorSummary = field(default_factory=TypeAccessorSummary)

    iqr:             TypeAccessorSummary = field(default_factory=TypeAccessorSummary)
    iqr_weighted:    TypeAccessorSummary = field(default_factory=TypeAccessorSummary)


@dataclass
class DistributionAccessorSummaries:
    per_type: Dict[str, TypeAccessorSummary]
    aggregates: MetaSummaries

    def save(self, csv_stem: str) -> Path:
        """
        Saves as a CSV.
        """
        folder = TkTkTPaths.append(TkTkTPaths.pathToEvaluations(), "av")
        file = folder / f"{csv_stem}.csv"
        with open(file, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["type"] + list(TypeAccessorSummary().__dict__.keys()))
            writer.writeheader()
            for f, summary in self.aggregates.__dict__.items():
                writer.writerow({"type": "$$$ " + f} | summary.__dict__)
            for t, summary in self.per_type.items():
                writer.writerow({"type": t} | summary.__dict__)
        return file


@dataclass
class AllAccessorSummaries:
    left:  DistributionAccessorSummaries
    right: DistributionAccessorSummaries
    both:  DistributionAccessorSummaries
    min:   DistributionAccessorSummaries  # For each type separately, picks the accessor distribution with the fewest types (i.e. the most predictable side) and copies its metrics.

    corpus_name: str

    def save(self):
        timestamp = datetimeDashed()
        self.left .save(self.corpus_name + "_" + timestamp + "_" + "left")
        self.right.save(self.corpus_name + "_" + timestamp + "_" + "right")
        self.both .save(self.corpus_name + "_" + timestamp + "_" + "both")
        self.min  .save(self.corpus_name + "_" + timestamp + "_" + "min")


def getAccessorsFromCorpus(
        tokeniser: Tokeniser, texts: Union[NamedIterable[str],NamedIterable[Tuple[str,int]]], bucket_samples_every: int,
        split_into_disjunct_examples: Preprocessor=None, print_contexts_for_tokens: Set[str]=None
    ) -> AccessorDistributions:
    """
    :param bucket_samples_every: Every type in the vocabulary has a left and right counter associated with it that counts
                                 how many tokens of each type are left resp. right of it. Those counts are bucketed in
                                 the order they come in with, so that later on, you can average over fixed-size windows of samples.
    """
    if split_into_disjunct_examples is None:
        split_into_disjunct_examples = IdentityPreprocessor()
    if print_contexts_for_tokens is None:
        print_contexts_for_tokens = set()

    max_id: VocabRef = 0
    vocab: Dict[str,VocabRef] = dict()

    # Everything you have seen to the left and right of a given type.
    left_of:  Dict[VocabRef, ChainedCounter[VocabRef]] = defaultdict(lambda: ChainedCounter(bucket_samples_every))
    right_of: Dict[VocabRef, ChainedCounter[VocabRef]] = defaultdict(lambda: ChainedCounter(bucket_samples_every))
    left_bounds:  Dict[VocabRef, int] = defaultdict(int)
    right_bounds: Dict[VocabRef, int] = defaultdict(int)

    for text,frequency in getIterableWithCounts(texts):
        for bounded_text in split_into_disjunct_examples.do(text):
            tokens = tokeniser.prepareAndTokenise(bounded_text)
            if not tokens:
                continue

            ids = []
            for token in tokens:
                if token in print_contexts_for_tokens:
                    print(f"Found token '{token}' in pretoken '{bounded_text}'")
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
            left_bounds[ids[0]]   += frequency
            right_bounds[ids[-1]] += frequency

            if len(ids) > 1:
                right_of[ids[0]][ids[1]]  += frequency  # Has no token to the left
                left_of[ids[-1]][ids[-2]] += frequency  # Has no token to the right

            # Middle tokens
            for i in range(1,len(ids)-1):
                center = ids[i]
                left_of[center][ids[i-1]]  += frequency
                right_of[center][ids[i+1]] += frequency

    return AccessorDistributions(
        vocab,
        AccessorDistribution(left_of, left_bounds),
        AccessorDistribution(right_of, right_bounds),
        corpus_name=texts.name
    )


def analyseAccessors(accessors: AccessorDistributions, predefined_vocab_size: Optional[int]=None) -> AllAccessorSummaries:
    # Extract some information from distributions
    vocab, left_of, right_of = accessors.vocab, accessors.left_of, accessors.right_of
    default_subcounter_size = at(0, left_of.accessors.values())._max_size

    # Step 1: Initialise empty summaries
    summaries = AllAccessorSummaries(
        left=DistributionAccessorSummaries(
            per_type=defaultdict(TypeAccessorSummary),
            aggregates=MetaSummaries()
        ),
        right=DistributionAccessorSummaries(
            per_type=defaultdict(TypeAccessorSummary),
            aggregates=MetaSummaries()
        ),
        both=DistributionAccessorSummaries(
            per_type=defaultdict(TypeAccessorSummary),
            aggregates=MetaSummaries()
        ),
        min=DistributionAccessorSummaries(
            per_type=defaultdict(TypeAccessorSummary),
            aggregates=MetaSummaries()
        ),
        corpus_name=accessors.corpus_name
    )

    # Step 2: Compute per-type summary statistics
    def fillTypeSummary(summary: TypeAccessorSummary, accessor_counts: ChainedCounter, end_count: int, possible_accessors: int):
        """Function that computes all the summary statistics for one type, and outputs the results in-place."""
        nonend_count = accessor_counts.total()

        # Quantities that are either explicitly dependent on corpus size, or which stabilise with corpus size.
        summary.total_accessors     = nonend_count + end_count
        summary.boundary_ratio      = end_count/summary.total_accessors if summary.total_accessors else 0.0  # This 0.0 is not technically correct. Let's hope this never happens.
        summary.entropic_efficiency = renyiEfficiency(accessor_counts.values(), domain_size=possible_accessors, sample_size=nonend_count, alpha=1.0)[1] if nonend_count else 0.0  # idk what to do with this default

        # Quantities that are monotonous in corpus size, and hence need to be averaged over a fixed window.
        summary.av                  = accessor_counts.averageOverCounters(lambda c: len(c))
        summary.coverage            = accessor_counts.averageOverCounters(lambda c: len(c) / possible_accessors) if nonend_count else 0.0
        summary.uniqueness          = accessor_counts.averageOverCounters(lambda c: len(c) / c.total())          if nonend_count else 1.0
        summary.mcu                 = max(summary.coverage, summary.uniqueness)

    # For each distribution we have (left/right/both/minimum), generate the per-type statistics.
    # You should pretend that all the code between here and step 3 is distribution-agnostic and run four times. That is:
    # ```
    #     for distribution in (left,right,both,min): [[[ for type in vocab: fill(distribution[type]) ]]]
    # ```
    # where [[[ ]]] is distribution-agnostic code. In practice, to save some double work, I have implemented it as
    # ```
    #     for type in vocab: for distribution in (left,right,both,min): fill(distribution[type])
    # ```
    vocabulary_size_leftward  = len(right_of.accessors)  # "How many possible types could appear LEFT OF a type?" is equivalent to asking "How many possible types have anything RIGHT OF themselves?"
    vocabulary_size_rightward = len(left_of.accessors)
    vocabulary_size_both      = len(set(left_of.accessors) | set(right_of.accessors))
    for t,i in streamProgress(vocab.items(), known_size=len(vocab), show_as="Computing type statistics"):
        left_ends  = left_of.boundaries .get(i, 0)
        right_ends = right_of.boundaries.get(i, 0)

        left_accessors  = left_of.accessors .get(i, ChainedCounter(default_subcounter_size))
        right_accessors = right_of.accessors.get(i, ChainedCounter(default_subcounter_size))

        # TODO: For types in the vocab that have 0 accessors, what should you do? They will have default values for the metrics, and those will meaninglessly skew the summary.
        fillTypeSummary(summaries.left .per_type[t], left_accessors,                   left_ends,              predefined_vocab_size or vocabulary_size_leftward)
        fillTypeSummary(summaries.right.per_type[t], right_accessors,                  right_ends,             predefined_vocab_size or vocabulary_size_rightward)
        fillTypeSummary(summaries.both .per_type[t], left_accessors + right_accessors, left_ends + right_ends, predefined_vocab_size or vocabulary_size_both)
        if summaries.left.per_type[t].av < summaries.right.per_type[t].av:
            summaries.min.per_type[t] = summaries.left.per_type[t]
        else:
            summaries.min.per_type[t] = summaries.right.per_type[t]

    # Step 3: Compute aggregates of type statistics across the vocabulary
    def fillAggregates(distribution_summaries: DistributionAccessorSummaries):
        """Function that takes filled per-type statistics and outputs aggregates for them in-place in the same object."""

        # First we define the reduction operators.
        class SummaryReduction(ABC):
            @abstractmethod
            def reduceField(self, type_values: List[float], type_weights: List[float]) -> float:
                pass

            def reduce(self, type_summaries: List[TypeAccessorSummary], type_weights: List[float]) -> TypeAccessorSummary:
                """We reduce all fields of the summaries with the same operator."""
                return TypeAccessorSummary(
                    **{
                        field_name: self.reduceField([getattr(s, field_name) for s in type_summaries], type_weights)
                        for field_name in TypeAccessorSummary().__dict__.keys()
                    }
                )

        class Mean(SummaryReduction):
            def reduceField(self, type_values: List[float], type_weights: List[float]) -> float:
                return float(np.mean(type_values))

        class Median(SummaryReduction):
            def reduceField(self, type_values: List[float], type_weights: List[float]) -> float:
                return float(np.median(type_values))

        class MAD(SummaryReduction):
            def reduceField(self, type_values: List[float], type_weights: List[float]) -> float:
                return float(sp.stats.median_abs_deviation(type_values))

        class IQR(SummaryReduction):
            def reduceField(self, type_values: List[float], type_weights: List[float]) -> float:
                return float(sp.stats.iqr(type_values))

        class MeanWeighted(SummaryReduction):
            def reduceField(self, type_values: List[float], type_weights: List[float]) -> float:
                values  = np.array(type_values)
                weights = np.array(type_weights)
                weights = weights / np.sum(weights)  # Because the weights are type frequencies, they are likely very large and hence it should be more stable to use small floats.
                return float(np.sum(weights * values))

        class MedianWeighted(SummaryReduction):
            def reduceField(self, type_values: List[float], type_weights: List[float]) -> float:
                return float(weighted_quantiles(type_values, type_weights, p=0.5))

        class MADWeighted(SummaryReduction):
            def reduceField(self, type_values: List[float], type_weights: List[float]) -> float:
                median = float(weighted_quantiles(type_values, type_weights, p=0.5))
                value_deviations = np.abs(np.array(type_values) - median)  # Every value's deviation from the median. Weights for these values and deviations should be equal.
                return float(weighted_quantiles(value_deviations, type_weights, p=0.5))

        class IQRWeighted(SummaryReduction):
            def reduceField(self, type_values: List[float], type_weights: List[float]) -> float:
                q1, q3 = weighted_quantiles(type_values, type_weights, p=[0.25, 0.75])
                return float(q3 - q1)

        # Now we get the actual instances to aggregate in this particular case.
        summaries_to_aggregate = list(distribution_summaries.per_type.values())
        weights_to_use         = [s.total_accessors for s in summaries_to_aggregate]

        distribution_summaries.aggregates.mean            = Mean()          .reduce(summaries_to_aggregate, weights_to_use)
        distribution_summaries.aggregates.mean_weighted   = MeanWeighted()  .reduce(summaries_to_aggregate, weights_to_use)
        distribution_summaries.aggregates.median          = Median()        .reduce(summaries_to_aggregate, weights_to_use)
        distribution_summaries.aggregates.median_weighted = MedianWeighted().reduce(summaries_to_aggregate, weights_to_use)
        distribution_summaries.aggregates.mad             = MAD()           .reduce(summaries_to_aggregate, weights_to_use)
        distribution_summaries.aggregates.mad_weighted    = MADWeighted()   .reduce(summaries_to_aggregate, weights_to_use)
        distribution_summaries.aggregates.iqr             = IQR()           .reduce(summaries_to_aggregate, weights_to_use)
        distribution_summaries.aggregates.iqr_weighted    = IQRWeighted()   .reduce(summaries_to_aggregate, weights_to_use)

    # As for the per-type summaries, this happens once for each distribution.
    fillAggregates(summaries.left)
    fillAggregates(summaries.right)
    fillAggregates(summaries.both)
    fillAggregates(summaries.min)

    return summaries


def weighted_quantiles(values: List[float], weights: List[float], p: Union[Iterable[float], float] = 0.5):
    """
    Given a weighted sample, computes the quantiles at the given .
    Quantile

    Taken from https://stackoverflow.com/a/73905572/9352077
    """
    values  = np.array(values)
    weights = np.array(weights)
    sorted_indices    = np.argsort(values)  # This order exists regardless of weight.
    cumsums_and_total = np.cumsum(weights[sorted_indices])  # This is essentially the unnormalised CDF.
    return values[sorted_indices[np.searchsorted(cumsums_and_total, v=np.array(p) * cumsums_and_total[-1], side="left")]]  # np.searchsorted computes the index where insertion sort would put each element given to it.
