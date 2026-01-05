"""
Constructs the set of most common character N-grams in pretokens.
(Has nothing to do with segmentation into N-grams since that's vocabulary-agnostic.)
"""
from typing import Tuple, Optional, Self
from pathlib import Path

import json
import warnings
import numpy as np
import numpy.random as npr
from collections import Counter

from ...interfaces import Artifacts, CacheableArtifacts
from ...interfaces.vocabularisers import *
from ...util.types import NamedIterable
from ...util.iterables import snd
from ...util.arrays import BatchNormalisation


class NgramArtifacts(Artifacts):
    pass


class CacheableNgramArtifacts(NgramArtifacts, CacheableArtifacts):
    def __init__(self, types: list[str]):
        super().__init__()
        self._types = types

    def _getVocabulary(self) -> UnidentifiedVocab:
        return self._types

    @classmethod
    def exists(cls, cache_path: Path) -> bool:
        return cls._existsTypes(cache_path)

    @classmethod
    def load(cls, cache_path: Path) -> Self:
        return CacheableNgramArtifacts(types=cls._loadTypes(cache_path))

    def store(self, cache_path: Path):
        self._storeTypes(cache_path, self._types)


class NgramVocabulariser(UnsupervisedVocabulariser[CacheableNgramArtifacts]):
    """
    Count the character N-grams (or 1...N-grams) in a corpus, truncate to the most frequent K, and then select |V| from
    that (randomly or just the top).
    """

    def __init__(self, preprocessor: Preprocessor, N_min: int, N_max: int, truncate_to_top: int, vocab_size: int,
                 sample_from_counts: Optional[BatchNormalisation]=None, seed: int=0):
        super().__init__(preprocessor=preprocessor)

        assert truncate_to_top >= vocab_size
        assert N_min <= N_max

        self._n_min = N_min
        self._n_max = N_max
        self._rng = npr.default_rng(seed)
        self._count_transform = sample_from_counts
        self._intermediate_size = truncate_to_top
        self._final_size        = vocab_size

    def _identifier(self) -> str:
        return f"ngram(N=({self._n_min},{self._n_max}))"

    def _cacheType(self):
        return CacheableNgramArtifacts

    def _vocabulariseFromWords(self, word_iterable: NamedIterable[Tuple[str,int]]) -> CacheableNgramArtifacts:
        # Step 1: Count ALL N-grams.
        ngram_counter = Counter()  # TODO: This is obviously a super naive way to do it. You probably want to periodically flush this to disk, like CountWords.
        N_range = range(self._n_min, self._n_max+1)
        for word, count in word_iterable:
            pretokens = self.preprocessor.do(word)
            for pretoken in pretokens:
                for N in N_range:
                    ngram_counter.update(pretoken[i:i+N] for i in range(len(pretoken)-N+1))

        # Step 2: Sort and truncate to top.  TODO: You likely want to save that sorted file too. It's massive tho.
        truncated = ngram_counter.most_common(self._intermediate_size)
        if len(truncated) < self._intermediate_size:
            warnings.warn(f"You asked to truncate to the top {self._intermediate_size} N-grams, but the corpus only had {len(truncated)}.")

        # Step 3: Sample from this.
        if self._count_transform:
            p = self._count_transform.normalise(np.fromiter(map(snd, truncated), dtype=float))
            indices = self._rng.choice(range(len(truncated)), size=self._final_size, replace=False, p=p)
            vocab   = {truncated[i][0]: id for id, i in enumerate(sorted(indices))}
        else:
            vocab = {truncated[i][0]: i for i in range(self._final_size)}

        return CacheableNgramArtifacts(types=sorted(vocab.keys(), key=vocab.get))

    def _vocabulariseFromSentences(self, sentence_iterable: NamedIterable[str]) -> CacheableNgramArtifacts:
        return self._vocabulariseFromWords(sentence_iterable.map(lambda s: (s,1)))
