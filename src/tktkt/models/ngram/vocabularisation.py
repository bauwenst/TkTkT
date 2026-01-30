"""
Constructs the set of most common character N-grams in pretokens.
(Has nothing to do with segmentation into N-grams since that's vocabulary-agnostic.)
"""
from typing import Optional, Self
from pathlib import Path

import numpy as np
import numpy.random as npr
from collections import Counter

from ..word.vocabularisation import CountWords, CountingConfig
from ...interfaces import Artifacts, CacheableArtifacts
from ...interfaces.preprocessors import Pretokeniser, Pretokens
from ...interfaces.vocabularisers import *
from ...preparation.splitters import PretokeniserSequence
from ...util.types import NamedIterable, generated
from ...util.arrays import BatchNormalisation
from ...util.strings import shash


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

    def __init__(self, preprocessor: Preprocessor, N_min: int, N_max: int,
                 frequency_minimum: int,
                 vocab_size: int, vocab_sampling: Optional[BatchNormalisation]=None, seed: int=0,
                 do_switch_loops: bool=True,
                 sentence_preprocessor: Preprocessor=None):
        """
        :param vocab_sampling: If None, the top vocab_size N-grams are selected as vocabulary. Otherwise, they are selected
                               with probability equal to whatever this function makes of their counts.
        :param do_switch_loops: If True, counts N-grams "for N, for word in corpus" rather than "for word in corpus, for N".
                                The time complexity of these are the same, but it allows saving some memory by disqualifying
                                larger N-grams if they don't contain any substrings with the minimal frequency.
        """
        super().__init__(preprocessor=preprocessor)

        assert 1 <= N_min <= N_max

        self._sentence_preprocessor = sentence_preprocessor
        self._n_min = N_min
        self._n_max = N_max
        self._freq_min = frequency_minimum
        self._do_bottomup = do_switch_loops

        self._rng = npr.default_rng(seed)
        self._count_transform = vocab_sampling
        self._final_size = vocab_size

    def _cacheSubfolder(self) -> str:
        return f"ngram"

    def _identifierPartial(self) -> str:
        return shash(repr(self.preprocessor)) + "_" + shash(f"n={self._n_min},N={self._n_max}_V={self._final_size}")  # TODO: Probably also add the seed.

    def _cacheType(self):
        return CacheableNgramArtifacts

    def _vocabulariseFromWords(self, word_iterable: NamedIterable[tuple[str,int]]) -> CacheableNgramArtifacts:
        folder = self._cachePath(word_iterable.name)
        folder_intermediate = folder / "shards"
        folder_intermediate.mkdir(exist_ok=True)

        if self._do_bottomup:  # TODO: Some kind of caching/status would be nice here.
            prev_counter = Counter()
            for N in range(1,self._n_max+1):
                # Step 1: Count exact N-grams.
                curr_counter = Counter()
                for word, frequency in word_iterable:
                    for pretoken in self.preprocessor.do(word):
                        if N == 1:
                            for c in word:
                                curr_counter[c] += frequency
                        else:
                            for i in range(len(pretoken)-N+1):
                                ngram = word[i:i+N]
                                if ngram[:-1] in prev_counter or ngram[1:] in prev_counter:  # Because superstring > f => substring > f, knowing the biggest substrings allows already excluding superstrings that we know will definitely be filtered in the filtering step.
                                    curr_counter[ngram] += frequency

                # Step 2: Filter and store result in prev_counter for next iteration.
                del prev_counter
                prev_counter = Counter({typ: f for typ,f in curr_counter.items() if f >= self._freq_min})

                # Step 3: Write out.
                with open(folder_intermediate / f"N={N}.tsv", "w", encoding="utf-8") as handle:
                    for k,v in prev_counter.most_common():
                        handle.write(f"{k}\t{v}\n")

            del curr_counter
            del prev_counter

            # Step 4: Aggregate into two disconnected lists that allow easy truncation and sampling.
            types  = []
            counts = []
            for N in range(self._n_min, self._n_max+1):
                with open(folder_intermediate / f"N={N}.tsv", "r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.rstrip()
                        if line:
                            ngram, count = line.split("\t")
                            types.append(ngram)
                            counts.append(int(count))
        else:
            # Step 1: Count all N-grams with length in [N_min, N_max].
            #         We do this using the safety valves built into CountWords, by pretending that sentences->words is words->N-grams.
            class NgramPretokeniser(Pretokeniser):
                def __init__(self, n_min: int, n_max: int):
                    self.n_range = range(n_min, n_max+1)

                def split(self, text: str) -> Pretokens:
                    return [pretoken[i:i+N] for N in self.n_range for i in range(len(pretoken)-N+1)]

                def invertTokens(self, pretokens: Pretokens) -> Pretokens:
                    raise NotImplementedError()

            class NgramPreprocessor(Preprocessor):
                def __init__(self, preprocessor: Preprocessor, n_min: int, n_max: int):
                    super().__init__(
                        uninvertible_mapping=preprocessor.irreversible,
                        invertible_mapping=preprocessor.reversible,
                        splitter=PretokeniserSequence([
                            preprocessor.splitter,
                            NgramPretokeniser(n_min=n_min, n_max=n_max)
                        ])
                    )

            c = CountWords(
                word_extractor=NgramPreprocessor(self.preprocessor, n_min=self._n_min, n_max=self._n_max),
                frequency_minimum=self._freq_min,
                sort_before_write=True,
                config=CountingConfig(
                    checkpoint_every_examples=1_000_000_000_000,
                    shard_if_keys_exceed=1_000_000,
                    drop_if_multiple_exceeded=10,
                    delete_shards_after=True
                )
            )
            serialised_ngram_counts = c.vocabulariseFromWordIterable(word_iterable)
            types, counts = zip(*serialised_ngram_counts.getFrequencies())

        if len(counts) < self._final_size:
            raise RuntimeError(f"Requested {self._final_size} N-grams but only found {len(counts)}.")

        if self._count_transform:
            p = self._count_transform.normalise(np.fromiter(counts, dtype=float))
            indices = sorted(self._rng.choice(range(len(counts)), size=self._final_size, replace=False, p=p))
            types = [types[i] for i in indices]
        else:
            indices = np.argsort(counts)  # Sorts small-to-big
            types = [types[i] for i in indices[-self._final_size:]]

        return CacheableNgramArtifacts(types=sorted(types, key=lambda t: (len(t), t)))

    def _vocabulariseFromSentences(self, sentence_iterable: NamedIterable[str]) -> CacheableNgramArtifacts:
        if self._sentence_preprocessor is None:
            raise ValueError("Missing sentence preprocessor.")

        word_counter = CountWords(word_extractor=self._sentence_preprocessor, frequency_minimum=1, sort_before_write=True,
                                  config=CountingConfig(checkpoint_every_examples=1_000_000_000_000_000_000, shard_if_keys_exceed=1_000_000, drop_if_multiple_exceeded=10, delete_shards_after=True))
        counts = word_counter._vocabulariseFromSentences(sentence_iterable)
        return self._vocabulariseFromWords(NamedIterable(generated(lambda: counts.getFrequencies()), name=sentence_iterable.name))
