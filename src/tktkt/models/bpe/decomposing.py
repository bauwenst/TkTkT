"""
Tokenisers that have flags attached to types in the vocabulary and recursively decompose the results of BPE merging
until no token has a flag.
"""
from typing import List, Set

from functools import lru_cache
from collections import Counter
import numpy.random as npr

from .base import DeterministicBPETokeniser, Preprocessor, BoundaryMarker, Vocab
from ...util.iterables import fst


class RecursivelyDecomposingBPE(DeterministicBPETokeniser):
    """
    First applies BPE like normal, and then recursively undoes applied merges until all tokens are NOT in a predefined set of illegal types.
    """

    def __init__(self, preprocessor: Preprocessor, marker: BoundaryMarker,
                 expanded_vocab: Vocab, merges: List[str],
                 disabled_set: Set[str]):
        """
        :param expanded_vocab: Vocabulary including both the types that remain accessible AND the types to be disabled.
        """
        super().__init__(
            preprocessor=preprocessor,
            boundary_marker=marker,

            vocab=expanded_vocab,
            merges=merges
        )
        self.disabled = disabled_set & set(expanded_vocab.keys())

    @lru_cache(maxsize=1024*1024)
    def tokenise(self, pretoken: str) -> List[str]:
        # Do BPE
        old_tokens = super().tokenise(pretoken)

        # Undo BPE
        new_tokens = []
        for token in old_tokens:
            new_tokens.extend(self._recursivelyDecompose(token))
        return new_tokens

    def _recursivelyDecompose(self, token: str) -> List[str]:
        if token not in self.vocab:  # Might be a problem considering that BTE doesn't automatically convert unknown characters to [UNK].
            raise ValueError(f"Cannot decompose token that doesn't have a type in the vocabulary: {token}")

        if token not in self.disabled:
            return [token]
        else:
            part1, part2 = self.merge_graph.merges_of[token][0].parts
            return self._recursivelyDecompose(part1) + self._recursivelyDecompose(part2)

    def _disableType(self, type_to_disable: str):
        if type_to_disable not in self.vocab or not self.merge_graph.merges_of[type_to_disable]:
            raise ValueError(f"Cannot trim a type from the vocabulary that cannot be decomposed further: {type_to_disable}")

        self.disabled.add(type_to_disable)


class ScaffoldBPETokeniser(RecursivelyDecomposingBPE):
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

    def __init__(self, preprocessor: Preprocessor, marker: BoundaryMarker,
                 vocab: Vocab, merges: List[str],
                 word_corpus: Counter[str], keep_type_if_more_frequent_than: int):
        super().__init__(
            preprocessor=preprocessor,
            marker=marker,

            expanded_vocab=vocab, merges=merges,
            disabled_set=set()
        )
        self._threshold = keep_type_if_more_frequent_than
        self.trim(word_corpus)

    def trim(self, corpus: Counter[str]):
        """
        Disable all types that appear <= threshold times when tokenising the given corpus.
        This process is done in parallel (find all types, then disable them) rather than serially (find a type, disable
        it, find a type with the updated tokeniser, disable it, ...).

        I suspect that this method is idempotent, but I'm too lazy to prove it.
        """
        # Get counts
        type_counts = Counter()
        for word, count in corpus.items():
            for token in self.prepareAndTokenise(word):
                type_counts[token] += count

        # Find and disable infrequent types
        for t in self._selectTypesToTrim(type_counts):
            try:
                self._disableType(t)
            except:  # t is part of the alphabet.
                pass

        # Reset cache
        self._syncWithGraph()

    def _selectTypesToTrim(self, type_distribution: Counter[str]) -> List[str]:
        return [t for t in self.vocab if type_distribution[t] <= self._threshold]


class RandomDropBPE(TrimmedBPE):
    """
    Flags k types for decomposition randomly from the N most frequent types.
    https://aclanthology.org/2024.lrec-main.1469.pdf
    """

    def __init__(self, preprocessor: Preprocessor, marker: BoundaryMarker,
                 vocab: Vocab, merges: List[str],
                 word_corpus: Counter[str], top_N_sampling_domain: int, random_k_sampling_size: int, seed: int=0):
        self._N = top_N_sampling_domain
        self._k = random_k_sampling_size
        self._rng = npr.default_rng(seed)

        super().__init__(  # First constructs BPE graph, then calls .trim().
            preprocessor=preprocessor, marker=marker,
            vocab=vocab, merges=merges,
            word_corpus=word_corpus, keep_type_if_more_frequent_than=-1
        )

    def trim(self, corpus: Counter[str]):
        assert 0 <= self._k <= self._N <= len(self._getNonAlphabet())  # This assertion can only happen when the BPE graph is constructed.
        super().trim(corpus)

    def _getNonAlphabet(self) -> List[str]:
        return [t for t in self.vocab if not self.merge_graph.inAlphabet(t)]

    def _selectTypesToTrim(self, type_distribution: Counter[str]) -> List[str]:
        subcounter = Counter()  # Contains only the counts for non-alphabet types (even if not part of the given counter).
        for t in self._getNonAlphabet():
            subcounter[t] = type_distribution[t]

        # Get most common N types, then sample k of them.
        most_common_types = list(map(fst, subcounter.most_common(self._N)))
        return [most_common_types[i] for i in self._rng.choice(self._N, size=self._k, replace=False)]
