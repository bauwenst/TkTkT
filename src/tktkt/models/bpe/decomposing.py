"""
Tokenisers that have flags attached to types in the vocabulary and recursively decompose the results of BPE merging
until no token has a flag.
"""
from typing_extensions import Self
from pathlib import Path
from abc import abstractmethod

from functools import lru_cache
from collections import Counter
import numpy.random as npr

from .base import _DeterministicBPETokeniser, ClassicBPE, MergeList
from .vocabularisation import BPEArtifacts, CacheableBPEArtifacts
from ...util.iterables import fst
from ...interfaces.vocabularisers import *
from ...interfaces.tokenisers import *
from ...util.types import NamedIterable, Tokens
from ...util.strings import shash

__all__ = ["ScaffoldBPE", "TrimmedBPE"]


class RecursivelyDecomposingBPE(_DeterministicBPETokeniser[WithSpecials]):
    """
    First applies BPE like normal, and then recursively undoes applied merges until all tokens are NOT in a predefined set of illegal types.
    """

    def __init__(self, preprocessor: Preprocessor,
                 expanded_vocab: Vocab[WithSpecials], merges: MergeList,  # TODO: Although you need the expanded vocabulary in the constructor (you need the full BPE graph), you should report a smaller vocabulary size and have a compacted ID mapping for downstream models, lest you have unused embeddings. The same is true for BPE-knockout and PickyBPE.
                 disabled_set: set[str]):
        """
        :param expanded_vocab: Vocabulary including both the types that remain accessible AND the types to be disabled.
        """
        super().__init__(
            preprocessor=preprocessor,

            vocab=expanded_vocab,
            merges=merges
        )
        self._disabled = disabled_set & set(expanded_vocab.keys())

    @lru_cache(maxsize=1024*1024)
    def tokenise(self, pretoken: str) -> Tokens:
        # Do BPE
        old_tokens = super().tokenise(pretoken)

        # Undo BPE
        new_tokens = []
        for token in old_tokens:
            new_tokens.extend(self._recursivelyDecompose(token))
        return new_tokens

    def _recursivelyDecompose(self, token: str) -> Tokens:
        if token not in self.vocab:  # Might be a problem considering that BTE doesn't automatically convert unknown characters to [UNK].
            raise ValueError(f"Cannot decompose token that doesn't have a type in the vocabulary: {token}")

        if token not in self._disabled:
            return [token]
        else:
            part1, part2 = self.merge_graph.merges_of[token][0].parts
            return self._recursivelyDecompose(part1) + self._recursivelyDecompose(part2)


class ScaffoldBPE(RecursivelyDecomposingBPE):
    """
    ScaffoldBPE is a recursively decomposing BPE tokeniser which flags types based on their frequency during training.
    During segmentation (after vocabularisation), it is just another recursively decomposing BPE tokeniser.
    https://arxiv.org/abs/2404.17808v2
    """
    pass


class TrimmedBPE(RecursivelyDecomposingBPE):
    """
    Flags types for decomposition if they are too infrequent.
    https://aclanthology.org/2024.insights-1.7/
    """
    pass


########################################################################################################################


class AblatedBPEArtifacts(BPEArtifacts):
    @abstractmethod
    def getAblations(self) -> set[str]:
        pass


class CacheableAblatedBPEArtifacts(CacheableBPEArtifacts, AblatedBPEArtifacts):
    def __init__(self, types: list[str], merges: list[tuple[str,...]], ablated_types: set[str]):
        super().__init__(types=types, merges=merges)
        self._disabled = ablated_types

    def getAblations(self) -> set[str]:
        return self._disabled

    def store(self, cache_path: Path):
        super().store(cache_path)
        with open(cache_path / "ablations.txt", "w", encoding="utf-8") as handle:
            for t in self._disabled:
                handle.write(f"{t}\n")

    @classmethod
    def load(cls, cache_path: Path) -> Self:
        bpe_artifacts = super().load(cache_path)
        with open(cache_path / "ablations.txt", "r", encoding="utf-8") as handle:
            ablations = [line.rstrip() for line in handle]
        return cls(
            types=bpe_artifacts._types,
            merges=bpe_artifacts._merges,
            ablated_types=set(ablations)
        )

    @classmethod
    def exists(cls, cache_path: Path) -> bool:
        return super().exists(cache_path) and (cache_path / "ablations.txt").exists()


class _FrequencyBasedRecursivelyDecomposingBPEVocabulariser(UnsupervisedVocabulariser[CacheableAblatedBPEArtifacts]):
    """
    Vocabulariser that finds the types to disable for a RecursivelyDecomposingBPE tokeniser,
    by tokenising a corpus and computing the unigram distribution on which some operation is applied.

    Note: this vocabulariser does NOT train the BPE tokeniser used to tokenise the corpus.
    """

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, merges: MergeList):
        super().__init__(preprocessor=preprocessor)
        self._tokeniser = ClassicBPE(preprocessor=preprocessor, vocab=vocab, merges=merges)

    def _identifierPartial(self) -> str:
        return shash(repr(self.preprocessor)) + "_" + shash(" ".join(sorted(self._tokeniser.vocab)))

    def _cacheType(self):
        return CacheableAblatedBPEArtifacts

    @abstractmethod
    def _selectTypesToTrim(self, type_distribution: Counter[str]) -> set[str]:
        pass

    def _getNonAlphabet(self) -> set[str]:
        return {t for t in self._tokeniser.vocab if not self._tokeniser.merge_graph.inAlphabet(t)}

    def _vocabulariseFromWords(self, word_iterable: NamedIterable[tuple[str,int]]) -> CacheableAblatedBPEArtifacts:
        # Get counts and find types
        type_counts = Counter()
        for word, count in word_iterable:
            for token in self._tokeniser.prepareAndTokenise(word):
                type_counts[token] += count
        disabled_types = self._selectTypesToTrim(type_counts) & self._getNonAlphabet()

        return CacheableAblatedBPEArtifacts(
            types=list(self._tokeniser.types()),
            merges=[tuple(m.split(" ")) for m in self._tokeniser.merge_graph.getRawMerges()],
            ablated_types=disabled_types
        )

    def _vocabulariseFromSentences(self, sentence_iterable: NamedIterable[str]) -> CacheableAblatedBPEArtifacts:
        return self._vocabulariseFromWords(self._preprocessSentencesToPretokenCounts(sentence_iterable))


class TrimmedBPEVocabulariser(_FrequencyBasedRecursivelyDecomposingBPEVocabulariser):
    """
    Disable all types that appear <= threshold times when tokenising the given corpus.
    This process is done in parallel (find all types, then disable them) rather than serially (find a type, disable
    it, find a type with the updated tokeniser, disable it, ...).

    I suspect that this process is idempotent, but I'm too lazy to prove it.
    """

    def __init__(self, preprocessor: Preprocessor,
                 vocab: Vocab, merges: MergeList,
                 keep_type_if_more_frequent_than: int):
        super().__init__(preprocessor=preprocessor, vocab=vocab, merges=merges)
        self._threshold = keep_type_if_more_frequent_than

    def _cacheSubfolder(self) -> str:
        return "trimmedbpe"

    def _selectTypesToTrim(self, type_distribution: Counter[str]) -> set[str]:
        return {t for t in self._tokeniser.vocab if type_distribution[t] <= self._threshold}


class RandomDropBPE(_FrequencyBasedRecursivelyDecomposingBPEVocabulariser):
    """
    Flags k types for decomposition randomly from the N most frequent types.
    https://aclanthology.org/2024.lrec-main.1469.pdf
    """

    def __init__(self, preprocessor: Preprocessor,
                 vocab: Vocab, merges: MergeList,
                 top_N_sampling_domain: int, random_k_sampling_size: int, seed: int=0):
        super().__init__(
            preprocessor=preprocessor,
            vocab=vocab, merges=merges
        )
        self._N = top_N_sampling_domain
        self._k = random_k_sampling_size
        self._rng = npr.default_rng(seed)
        assert 0 <= self._k <= self._N <= len(self._getNonAlphabet())  # This assertion can only happen when the BPE graph is constructed.

    def _cacheSubfolder(self) -> str:
        return "randomdropbpe"

    def _selectTypesToTrim(self, type_distribution: Counter[str]) -> list[str]:
        subcounter = Counter()  # Contains only the counts for non-alphabet types (even if not part of the given counter).
        for t in self._getNonAlphabet():
            subcounter[t] = type_distribution[t]

        # Get most common N types, then sample k of them.
        most_common_types = list(map(fst, subcounter.most_common(self._N)))
        return [most_common_types[i] for i in self._rng.choice(self._N, size=self._k, replace=False)]
