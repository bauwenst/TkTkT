"""
Evaluation of the context around tokens.
"""
from typing import Iterable, Dict, Union, Tuple, Optional, Callable, List, Self
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field, fields
from collections import defaultdict, Counter

import scipy as sp
import numpy as np
import dacite
import json
import csv
import re

from .entropy import renyiEfficiency, DEFAULT_RENYI_ALPHA
from .observing import FinallyObservableObserver, Observer
from ..util.arrays import weighted_quantiles
from ..util.interfaces import Cacheable
from ..util.iterables import streamProgress, first
from ..util.dicts import ChainedCounter, invertdict
from ..util.types import Tokens


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

    def unigramFrequencyDistribution(self) -> Counter[int]:
        return Counter({id: self.countOccurrences(id) for id in set(self.accessors.keys()) | set(self.boundaries.keys())})

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
class AccessorDistributions(Cacheable):
    vocab: Dict[str,VocabRef]
    left_of:  AccessorDistribution
    right_of: AccessorDistribution

    _FILENAME = "distributions.json"

    @classmethod
    def exists(cls, cache_path: Path) -> bool:
        return (cache_path / AccessorDistributions._FILENAME).exists()

    def store(self, cache_path: Path):
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

        data = {
            "vocab": self.vocab,
            "left": serialiseDistribution(self.left_of),
            "right": serialiseDistribution(self.right_of)
        }
        serialised = json.dumps(data, indent=2, ensure_ascii=False)
        serialised = re.compile(r"\[\s+([0-9]+),\s+([0-9]+)\s+\]").sub(r"[\1,\2]", serialised)
        with open(cache_path / AccessorDistributions._FILENAME, "w", encoding="utf-8") as handle:
            handle.write(serialised)

    @classmethod
    def load(cls, cache_path: Path) -> Self:
        with open(cache_path / AccessorDistributions._FILENAME, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        distributions = AccessorDistributions(
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
        for t in streamProgress(removed, show_as="Removing types"):  # NOTE: The reason this loop is so slow is because it is practically O(|V|² x |D|): |V| because you remove some fraction of the vocabulary proportional to it, |V| because for every type you have to look through the neighbours, and |D| because those neighbours are stored in buckets whose amount grows proportionally to the corpus.
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

    def _assertDistributionalConsistency(self):
        """
        Because accessors for boundaries are not tracked, the left and right distributions get out of sync after
        filtering. For example, imagine the following corpus:

            ab 50
            ac 50
            cb 25

        We agree that 'a' appears 100 times, 'b' 75 times, and 'c' 75 times too. This gives the distributions

            left = {
                a: {BOUNDARY: 100}
                b: {a: 50, c: 25}
                c: {a: 50, BOUNDARY: 25}
            }
            right = {
                a: {b: 50, c: 50}
                b: {BOUNDARY: 75}
                c: {b: 25, BOUNDARY: 50}
            }

        Note how per type, left and right have equal sums, and that the sums across the entire left is the same as the sum
        over the entire right distribution. But now if we take out c:

            left = {
                a: {BOUNDARY: 100}
                b: {a: 50}
            }
            right = {
                a: {b: 50}
                b: {BOUNDARY: 75}
            }

        Neither of the invariants now holds. According to the right distribution, 'a' only appeared 50 times, and vice
        versa for 'b'. These are completely different frequency distributions, so computing a non-directional distribution
        is dangerous in this case.
        """
        assert self.left_of.unigramFrequencyDistribution() == self.right_of.unigramFrequencyDistribution(), "This computation cannot be performed on a filtered accessor distribution."

    # Note: All the properties below can also be obtained from running unigram analysers rather than a bigram analyser.

    def corpusCharacterCount(self) -> int:
        self._assertDistributionalConsistency()
        inverse_vocab = invertdict(self.vocab, noninjective_ok=False)
        return sum(len(inverse_vocab[id])*count for id, count in self.right_of.unigramFrequencyDistribution().items())

    def corpusTokenCount(self) -> int:
        self._assertDistributionalConsistency()
        return self.right_of.unigramFrequencyDistribution().total()

    def renyiEfficiency(self, alpha: float=DEFAULT_RENYI_ALPHA) -> float:
        self._assertDistributionalConsistency()
        frequencies = self.right_of.unigramFrequencyDistribution().values()
        return renyiEfficiency(frequencies, alpha=alpha, domain_size=len(self.vocab), sample_size=sum(frequencies))[1]  # (This function allows unnormalised input.)


########################################################################################################################


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


AGGREGATE_PREFIX = "$$$ "

@dataclass
class DistributionAccessorSummaries:
    per_type: Dict[str, TypeAccessorSummary]
    aggregates: MetaSummaries

    def save(self, path: Path) -> Path:
        assert path.suffix == ".csv"
        with open(path, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["type"] + list(TypeAccessorSummary().__dict__.keys()))
            writer.writeheader()
            for f, summary in self.aggregates.__dict__.items():
                writer.writerow({"type": AGGREGATE_PREFIX + f} | summary.__dict__)
            for t, summary in self.per_type.items():
                writer.writerow({"type": t} | summary.__dict__)
        return path

    @classmethod
    def load(cls, path: Path) -> Self:
        per_type = dict()
        aggregates = dict()
        with open(path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                t = row.pop("type")
                row = {k: float(v) for k,v in row.items()}
                if t.startswith(AGGREGATE_PREFIX):
                    t = t.removeprefix(AGGREGATE_PREFIX)
                    aggregates[t] = dacite.from_dict(TypeAccessorSummary, row, config=dacite.Config(check_types=False))  # dacite is strict about not having floats where you should have ints.
                else:
                    per_type[t] = dacite.from_dict(TypeAccessorSummary, row, config=dacite.Config(check_types=False))

        return DistributionAccessorSummaries(
            per_type=per_type,
            aggregates=dacite.from_dict(MetaSummaries, aggregates)
        )


@dataclass
class AllAccessorSummaries(Cacheable):
    left:  DistributionAccessorSummaries
    right: DistributionAccessorSummaries
    both:  DistributionAccessorSummaries
    min:   DistributionAccessorSummaries  # For each type separately, picks the accessor distribution with the fewest types (i.e. the most predictable side) and copies its metrics.

    @classmethod
    def exists(cls, cache_path: Path) -> bool:
        return (
            (cache_path / "summary_left.csv" ).exists() or
            (cache_path / "summary_right.csv").exists() or
            (cache_path / "summary_min.csv"  ).exists() or
            (cache_path / "summary_both.csv" ).exists()
        )

    @classmethod
    def load(cls, cache_path: Path) -> Self:
        return cls.loadFromPaths(
            left =cache_path / "summary_left.csv",
            right=cache_path / "summary_right.csv",
            both =cache_path / "summary_both.csv",
            min  =cache_path / "summary_min.csv"
        )

    def store(self, cache_path: Path):
        self.left .save(cache_path / "summary_left.csv"),
        self.right.save(cache_path / "summary_right.csv"),
        self.both .save(cache_path / "summary_both.csv"),
        self.min  .save(cache_path / "summary_min.csv")

    @classmethod
    def loadFromPaths(cls, left: Path, right: Path, both: Path, min: Path) -> "AllAccessorSummaries":
        return AllAccessorSummaries(
            left= DistributionAccessorSummaries.load(left),
            right=DistributionAccessorSummaries.load(right),
            both= DistributionAccessorSummaries.load(both),
            min=  DistributionAccessorSummaries.load(min)
        )


class AccessorCounting(FinallyObservableObserver[Tokens,AccessorDistributions]):

    def __init__(self, bucket_samples_every: int, disable_cache: bool=False, observers: List[Observer[AccessorDistributions]]=None):
        """
        :param bucket_samples_every: Every type in the vocabulary has a left and right counter associated with it that counts
                             how many tokens of each type are left resp. right of it. Those counts are bucketed in
                             the order they come in with, so that later on, you can average over fixed-size windows of samples.
        """
        super().__init__(disable_cache=disable_cache, observers=observers)
        self._bucket_size = bucket_samples_every

    def _nodeIdentifier(self) -> str:
        return f"AV-bucket-size={self._bucket_size}"

    def _cacheType(self):
        return AccessorDistributions

    def _cacheSubfolders(self) -> list[str]:
        return ["av", "counts"]

    def _initialiseAsObserver(self, identifier: str):
        self.max_id: VocabRef           = 0
        self.vocab: Dict[str, VocabRef] = dict()

        # Everything you have seen to the left and right of a given type.
        self.left_of:  Dict[VocabRef, ChainedCounter[VocabRef]] = defaultdict(lambda: ChainedCounter(self._bucket_size))
        self.right_of: Dict[VocabRef, ChainedCounter[VocabRef]] = defaultdict(lambda: ChainedCounter(self._bucket_size))
        self.left_bounds:  Dict[VocabRef, int] = defaultdict(int)
        self.right_bounds: Dict[VocabRef, int] = defaultdict(int)

    def _receive(self, sample: Tokens, weight: float):
        tokens, frequency = sample, weight
        if not tokens:
            return

        ids = []
        for token in tokens:
            try:
                ids.append(self.vocab[token])
            except:
                self.vocab[token] = self.max_id
                self.max_id += 1
                ids.append(self.vocab[token])

        # Edge tokens
        # - When a type appears at the start/end of an example, it probably still has an accessor to its left/right
        #   in reality, but we can't see it because the example is only an excerpt.
        # - An upper estimate on accessor variety is to always consider these unknown edges to be unique accessors.
        # - If you are studying morphology, you probably want to have an edge around every word, because it matters
        #   much less in such cases what the exact type was that came before (there is no connection between the
        #   characters of the previous word and of the current word, only the meanings).
        self.left_bounds[ids[0]]   += frequency
        self.right_bounds[ids[-1]] += frequency

        if len(ids) > 1:
            self.right_of[ids[0]][ids[1]]  += frequency  # Has no token to the left
            self.left_of[ids[-1]][ids[-2]] += frequency  # Has no token to the right

        # Middle tokens
        for i in range(1,len(ids)-1):
            center = ids[i]
            self.left_of[center][ids[i-1]]  += frequency
            self.right_of[center][ids[i+1]] += frequency

    def _compute(self) -> AccessorDistributions:
        return AccessorDistributions(
            self.vocab,
            AccessorDistribution(self.left_of, self.left_bounds),
            AccessorDistribution(self.right_of, self.right_bounds)
        )


class AccessorVariety(FinallyObservableObserver[AccessorDistributions,AllAccessorSummaries]):

    def __init__(self, predefined_vocab_size: Optional[int]=None, cache_disambiguator: str= "", disable_cache: bool=False, observers: List[Observer[AllAccessorSummaries]]=None):
        super().__init__(cache_disambiguator=cache_disambiguator, disable_cache=disable_cache, observers=observers)
        self._predefined_vocab_size = predefined_vocab_size

    def _nodeIdentifier(self) -> str:
        return ""

    def _cacheType(self):
        return AllAccessorSummaries

    def _cacheSubfolders(self) -> list[str]:
        return ["av", "summaries"]

    def _initialiseAsObserver(self, identifier: str):
        self.distributions = None

    def _receive(self, sample: AccessorDistributions, _):
        self.distributions = sample

    def _compute(self) -> AllAccessorSummaries:
        return summariseAccessors(self.distributions, self._predefined_vocab_size)


def summariseAccessors(accessors: AccessorDistributions, predefined_vocab_size: Optional[int]=None) -> AllAccessorSummaries:
    # Extract some information from distributions
    vocab, left_of, right_of = accessors.vocab, accessors.left_of, accessors.right_of
    default_subcounter_size = first(left_of.accessors.values())._max_size

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
        )
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
    #     for distribution in (left,right,both,min): <<< for type in vocab: fill(distribution[type]) >>>
    # ```
    # where <<< >>> is distribution-agnostic code. In practice, to save some double work, I have implemented it as
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
        summaries.min.per_type[t] = summaries.left.per_type[t] if summaries.left.per_type[t].av < summaries.right.per_type[t].av else summaries.right.per_type[t]

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
                        field.name: self.reduceField([getattr(s, field.name) for s in type_summaries], type_weights)
                        for field in fields(TypeAccessorSummary)
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
