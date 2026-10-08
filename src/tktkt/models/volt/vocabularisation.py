"""
Heavily refactored version of the code found at https://github.com/Jingjing-NLP/VOLT
for the Xu e.a. (2021) paper https://aclanthology.org/2021.acl-long.571/.

There are at least three inconsistencies between the paper and the code:
    1. Whether to order the seed lexicon by corpus substring frequency (paper) or BPE merge rank (code);
    2. Distance matrix calculation between characters and tokens being length-based (paper) or rank-based (code);
    3. Whether to maximise -(H2 - H1) (paper) or Sinkhorn output 2 - Sinkhorn output 1 (code).

We expose these differences where possible. See the constructor documentation.
"""
from abc import abstractmethod
from typing import Optional, Iterable, Union
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from math import log
from modest.formats.tsv import iterateTsv

from ..word.vocabularisation import CountSubstrings, CountCharacters
from ...evaluation import FutureObserver
from ...evaluation.entropy import Entropy, TokenUnigramDistribution
from ...factories.evaluators import evaluateTokeniser
from ...interfaces import Artifacts, TokeniserWithVocabulary, Vocab
from ...interfaces.artifactories import CacheableArtifactsWithVocabulary
from ...interfaces.identifiers import NoSpecials
from ...interfaces.vocabularisers import UnsupervisedVocabulariser, T_CacheableArtifact
from ...util.iterables import snd, most
from ...util.strings import shash
from ...util.types import NamedIterable


_DEFINE_MUV_BY_GRID_STEP = True  # If set to True, the ΔV in MUV == -ΔH/ΔV is defined as the difference in seed vocabulary sizes
                                 # (as seems to be the case in the paper) rather than the difference in pruned vocabulary sizes (even though it is the pruned vocabularies that produce the entropies for ΔH).

@dataclass
class VOLTResult:
    """One point in the grid search."""
    seed_size: int
    pruned_size: int
    sinkhorn_loss: float
    entropy: float

    def serialise(self) -> tuple[int,int,float,float]:
        return (self.seed_size, self.pruned_size, self.sinkhorn_loss, self.entropy)

    def __lt__(self, other: "VOLTResult") -> bool:
        return self.seed_size < other.seed_size


@dataclass
class VOLTResults:
    """All points in the grid search."""
    data: list[VOLTResult]

    def __iter__(self):
        return iter(self.data)

    def getMarginalLosses(self) -> dict[int,float]:
        size_with_loss = [(record.seed_size, record.sinkhorn_loss) for record in sorted(self.data)]
        return {size_curr: loss_curr - loss_prev for (_,loss_prev),(size_curr,loss_curr) in zip(size_with_loss, size_with_loss[1:])}

    def getMUVs(self, normalise_by_pruned_size: bool=True) -> dict[int,float]:
        """
        :param normalise_by_pruned_size: whether to normalise by the difference in sizes of the vocabularies that were
                                         used for segmentation, rather than the size of the seed vocabularies used to get to those vocabularies.
        """
        size_with_entropy = [(record.pruned_size if normalise_by_pruned_size else record.seed_size, record.entropy) for record in sorted(self.data)]
        return {size_curr: -(entropy_curr - entropy_prev)/(size_curr - size_prev) for (size_prev,entropy_prev),(size_curr,entropy_curr) in zip(size_with_entropy, size_with_entropy[1:])}

    def getOptimalSeedSize(self) -> int:
        if most(record.entropy == 0 for record in self.data):  # Then loss was used.
            scores = self.getMarginalLosses()
        else:  # Then MUV was used.
            scores = self.getMUVs(normalise_by_pruned_size=_DEFINE_MUV_BY_GRID_STEP)
        return max(scores.keys(), key=scores.get)


class VOLTArtifacts(Artifacts):

    @abstractmethod
    def getResults(self) -> VOLTResults:
        pass

    def getOptimalSeedSize(self) -> int:
        return self.getResults().getOptimalSeedSize()

    def getOptimalVocabSize(self) -> int:
        return len(self.getVocabulary())


class CacheableVOLTArtifacts(CacheableArtifactsWithVocabulary, VOLTArtifacts):

    _STATISTICS_FILENAME = "stats.tsv"

    def __init__(self, selected_types: Iterable[str], statistics: list[VOLTResult]):
        super().__init__(types=selected_types)
        self._stats = VOLTResults(statistics)

    def getResults(self) -> VOLTResults:
        return self._stats

    @classmethod
    def load(cls, cache_path: Path):
        return CacheableVOLTArtifacts(
            cls._loadTypes(cache_path),
            [VOLTResult(int(s1), int(s2), float(l), float(e)) for s1, s2, l, e in iterateTsv(cache_path / cls._STATISTICS_FILENAME)]
        )

    def store(self, cache_path: Path):
        super().store(cache_path)
        with open(cache_path / self._STATISTICS_FILENAME, "w", encoding="utf-8") as handle:
            for stat in self._stats:
                handle.write("\t".join(map(str,stat.serialise())) + "\n")


OrderedStringCounts = list[tuple[str,int]]

class VOLTVocabulariser(UnsupervisedVocabulariser[CacheableVOLTArtifacts]):

    def __init__(self,
                 initial_ordered_vocabulary: Artifacts,
                 vocab_size_max: int,
                 vocab_size_step: int=1000,

                 reorder_vocabulary_by_frequency: bool=False,
                 length_based_distance: bool=False,
                 segmenter_for_entropy: Optional[TokeniserWithVocabulary]=None,

                 max_iterations: int=500,
                 conservation_threshold: float=0.00001):
        """
        Vocabulary learning with optimal transport.

        The default values of this constructor reproduce the scripts of the VOLT authors, although it deviates from the paper
        in three different ways.
            1. The scripts consider a vocabulary ordered beforehand according to BPE, not by frequency.
            2. The scripts build the distance matrix based on ranks, not based on length.
            3. The scripts maximise the marginal Sinkhorn loss, not MUV. For MUV, you need to segment the corpus
               in some kind of way given the vocabulary produced by Sinkhorn, and so you need a segmenter.

        By changing the values of these parameters, you will get further from their scripts but closer to the paper.

        :param initial_ordered_vocabulary: Seed vocabulary from which the optimal subset will be selected.
                                           The process is repeated for all prefixes of this vocabulary
                                           of sizes n x vocab_size_step, which is why the order of the vocabulary matters.
                                           (Either you order by frequency as counted in the corpus, or you use the order given by the Artifacts, e.g. BPE ranking.)
        :param segmenter_for_entropy: If not given, marginal Sinkhorn loss is used to decide the optimal vocabulary.
                                      Else, this should be a fully configured tokeniser whose vocabulary will be swapped out by the Sinkhorn vocabularies.
                                      Do not use a tokeniser which has other internal structures that are tied to the vocabulary (e.g. a BPE or ULM tokeniser).
        :param conservation_threshold: Threshold above which tokens are kept. Higher threshold means more tokens won't make it and are thus removed from the initial vocabulary once
                                       optimal transport is finished. The paper recommends 0.001, the scripts recommend a more conservative 0.00001 and thus we go with that.
        """
        import ot  # Checks whether you have installed https://github.com/PythonOT/POT

        super().__init__(preprocessor=initial_ordered_vocabulary.preprocessorEffective())
        self.initial_vocabulary = initial_ordered_vocabulary
        self.vocab_size_step = vocab_size_step
        self.vocab_size_max  = vocab_size_max

        self.reorder_vocabulary_by_frequency = reorder_vocabulary_by_frequency
        self.length_based_distance = length_based_distance
        self.segmenter_for_entropy = segmenter_for_entropy

        self.max_iterations  = max_iterations
        self.threshold       = conservation_threshold

    def _cacheSubfolder(self) -> str:
        return "volt"

    def _cacheType(self):
        return CacheableVOLTArtifacts

    def _identifierPartial(self) -> str:
        return shash(f"{self.reorder_vocabulary_by_frequency}+{self.segmenter_for_entropy.getName()}+{self.length_based_distance}"
                     f"{self.vocab_size_step}+{self.vocab_size_max}+{self.max_iterations}+{self.threshold}") + "_" + self.initial_vocabulary.getVocabulary().fingerprint()

    def _vocabulariseFromWords(self, word_iterable: NamedIterable[tuple[str,int]]) -> CacheableVOLTArtifacts:
        return self._vocabulariseFromCounts(
            char_counts=list(CountCharacters(self.preprocessor).vocabulariseFromWordIterable(word_iterable).getFrequencies()),
            token_counts=list(CountSubstrings(self.initial_vocabulary).vocabulariseFromWordIterable(word_iterable).getFrequencies()),
            source=word_iterable
        )

    def _vocabulariseFromSentences(self, sentence_iterable: NamedIterable[str]) -> T_CacheableArtifact:
        raise NotImplementedError()

    def _vocabulariseFromCounts(self, char_counts: OrderedStringCounts, token_counts: OrderedStringCounts, source: Union[NamedIterable[str],NamedIterable[tuple[str,int]]]) -> CacheableVOLTArtifacts:
        if self.reorder_vocabulary_by_frequency:
            token_counts = sorted(token_counts, key=lambda t: t[1], reverse=True)

        # Grid search
        optimal_seed_size, diagnostics = self._searchBestSize(tokens=token_counts, characters=char_counts, source=source)

        # Re-create the best solution, just to get the vocabulary.
        _, P = self._solveOptimalTransport(tokens=token_counts[:optimal_seed_size], characters=char_counts, max_iterations=self.max_iterations, length_based_distance=self.length_based_distance)
        conserved_types = self._getPrunedVocabulary(token_counts, char_counts, P, self.threshold)
        return CacheableVOLTArtifacts(conserved_types, diagnostics.data)

    def _searchBestSize(self, tokens: OrderedStringCounts, characters: OrderedStringCounts, source: Union[NamedIterable[str],NamedIterable[tuple[str,int]]]) -> tuple[int, VOLTResults]:
        sizes      = []
        reductions = []
        losses     = []
        entropies  = []

        for size in range(self.vocab_size_step, self.vocab_size_max, self.vocab_size_step):
            loss, P = self._solveOptimalTransport(tokens[:size], characters, max_iterations=self.max_iterations, length_based_distance=self.length_based_distance)
            vocab   = Vocab(self._getPrunedVocabulary(tokens[:size], characters, P, threshold=self.threshold), specials=NoSpecials())
            sizes     .append(size)
            reductions.append(len(vocab))
            losses    .append(loss)
            if self.segmenter_for_entropy is not None:
                # Use the vocabulary to get entropy.
                self.segmenter_for_entropy.preprocessor = self.preprocessor
                self.segmenter_for_entropy.vocab = vocab
                future = FutureObserver()
                evaluateTokeniser(self._identifierPartial() + f"_{size}", source, self.segmenter_for_entropy, token_consumers=[
                    TokenUnigramDistribution(ensured_vocabulary=vocab, observers=[
                        Entropy(observers=[
                            future
                        ])])])
                entropies.append(future.resolve().entropy)  # .................................
            else:
                entropies.append(0)

        # Maximise
        results = VOLTResults([VOLTResult(*t) for t in zip(sizes, reductions, losses, entropies)])
        return results.getOptimalSeedSize(), results

    ### All the methods below are classmethods so that they are accessible externally. We may change this in the future.

    @classmethod
    def _getDistanceMatrix(cls, characters: OrderedStringCounts, tokens: OrderedStringCounts, length_based_distance: bool) -> np.ndarray:
        """
        Initialize distance matrix in optimal transport. if the i-th char in j-th token, their distance is set to be a very small value, otherwize a large value.
        Return a 2-dimension distance matrix.
        """
        BIG = 1e6

        matrix = np.zeros((len(characters), len(tokens)))
        for i, (char, _) in enumerate(characters):
            for j, (token, _) in enumerate(tokens):
                if char in token:
                    matrix[i][j] = 0.001 * j if not length_based_distance else log(len(token))  # == -log(1.0 / len(token))
                else:
                    matrix[i][j] = BIG

        return matrix

    @classmethod
    def _getPrunedVocabulary(cls, tokens: OrderedStringCounts, characters: OrderedStringCounts,
                             ot_matrix: np.ndarray, threshold: float) -> list[str]:
        """
        Conserve types based on the optimal-transport matrix P.

        :param ot_matrix: The chars x tokens P-matrix as generated by Sinkhorn optimal transport. Column j must correspond to
                          tokens[j] (same order used when P was built).
        :param threshold: The filter ratio from the optimal transportation matrix to the real vocabulary.
                          Higher threshold means that more tokens are removed from the token candidates.
        """
        assert ot_matrix.shape == (len(characters), len(tokens))

        vocab_size = len(tokens)
        relevant_characters_by_type: dict[str, dict[str, float]] = dict()  # Maps a type to each character, and each character to some kind of number. The number seems purely for logging reasons; what this dictionary actually measures is whether there exists any character that has a high enough score when interacting with a given type.
        for j, (typ, count_typ) in enumerate(tokens):
            if typ.strip() == "":  # TODO: Probably this is the wrong place for this check.
                continue

            relevant_characters_by_type[typ] = dict()
            for i, (char, _) in enumerate(characters):  # There is no constraint here that says the character must be in the type; it could be added but is probably not necessary since the distance matrix pushes OT away from this kind of solution.
                merge_versus_char_score = float(ot_matrix[i][j] * vocab_size)
                if ot_matrix[i][j] != 0 and merge_versus_char_score > threshold * count_typ:
                    relevant_characters_by_type[typ][char] = merge_versus_char_score  # * len(tokens[j][0])

        return [typ for typ, _ in tokens if relevant_characters_by_type[typ]]

    @classmethod
    def _solveOptimalTransport(cls, tokens: OrderedStringCounts, characters: OrderedStringCounts, max_iterations: int, length_based_distance: bool) -> tuple[float, np.ndarray]:
        total_chars  = getTotalFromCountIterable(characters)
        total_tokens = getTotalFromCountIterable(tokens)

        d_matrix = cls._getDistanceMatrix(characters, tokens, length_based_distance=length_based_distance)
        a = getExpectedCharacters(characters, total_chars, force_value_to_1=True)
        b = getExpectedCharacters(tokens, total_tokens)

        # TODO: Why are these here?
        from ot import sinkhorn
        epsilon = 0.1  # entropy parameter
        alpha = 1.0  # Unbalanced KL relaxation parameter
        loss, P = sinkhorn(a, b, d_matrix, reg=1.0, method='sinkhorn', numItermax=max_iterations, epsilon0=1e-6)
        return loss, P


##########################################################################################################


def getTotalFromCountIterable(counts: OrderedStringCounts) -> int:
    return sum(map(snd, counts))


def getExpectedCharacters(strings_with_counts: OrderedStringCounts, denominator: int, force_value_to_1: bool=False):
    """
    Compute how many characters each item in the given iterator is expected to produce given a unigram item generation process.

    :param force_value_to_1: whether to force the amount of characters produced to be 1.
    """
    return [
        (len(token) if not force_value_to_1 else 1) * freq/denominator + 1e-4
        for token,freq in strings_with_counts
    ]
