"""
Builds a vocabulary using the SaGe algorithm described in
    https://aclanthology.org/2023.eacl-main.45.pdf

During vocabulary building, words are kept in the context of their sentences rather than coming from a word frequency
list. The resulting vocabulary is just a set of subwords without context, however.
"""
from pathlib import Path
from typing import Self

import warnings

from ...interfaces import Artifacts, CacheableArtifacts
from ...interfaces.vocabularisers import *
from ...util.iterables import deduplicate
from ...util.strings import shash
from .schedules import *


class SageArtifacts(Artifacts):
    pass


class CacheableSageArtifacts(SageArtifacts, CacheableArtifacts):
    def __init__(self, types: list[str]):
        super().__init__()
        self._types = types

    def _getVocabulary(self) -> UnidentifiedVocab:
        return self._types

    def store(self, cache_path: Path):
        self._storeTypes(cache_path, self._types)

    @classmethod
    def load(cls, cache_path: Path) -> Self:
        return CacheableSageArtifacts(types=cls._loadTypes(cache_path))

    @classmethod
    def exists(cls, cache_path: Path) -> bool:
        return cls._existsTypes(cache_path)


class SageVocabulariser(UnsupervisedVocabulariser[CacheableSageArtifacts]):

    def __init__(self, initial_artifacts: Artifacts, seed: int=0,
                 vocabulary_schedule: Schedule=DoubleLinearSchedule(262144, 65536, 16384, t_mid=0.5), n_vocab_samples: int=13,
                 embedding_schedule: Dilation=ExponentialDilation(1.35), n_embedding_samples: int=6):
        """
        Note that a substantial part of the SaGe paper was dedicated to describing the hyperparameters it introduced,
        whereas the current implementation at https://github.com/MeLeLbgu/SaGe, which is SaGe v2.0, has lost most of
        those due to speedups making them obsolete.

        Hence, you will not see hyperparameters V, n, k, l, m and M in this constructor. That means the default values
        of n = 1.25, l = 4, k = 100, M = 1500, m = 10 have also become obsolete.

        The schedule parameters replace SaGe v2.0's hardcoded schedules, which are respectively
            vocab: 262144 229376 196608 163840 131072 98304 65536 57344 49152 40960 32768 16384
            embed: 262144                      131072       65536       49152 40960 32768

        :param initial_artifacts: Contains the vocabulary from which to start the pruning process, as well as the preprocessor
                                  that maps strings into that vocabulary's character space.
                                  Under the hood, all types in the vocabulary are mapped to bytes. This is not an issue
                                  even if the given vocabulary uses pseudo-bytes: types in the vocabulary will just
                                  become longer internally, but since the preprocessor also maps input text to pseudo-bytes and
                                  those are also converted to bytes by SaGe, the input and vocabulary will match in character space.
        :param vocabulary_schedule: Shape of the sequence of vocabulary sizes you transition to at each pruning step.
                                    The shape is on a normalised time axis, meaning SaGe starts at t=0.0 and ends at t=1.0.
        :param n_vocab_samples: How many equidistant samples to take from the schedule on that axis.

        :param embedding_schedule: At some vocabulary sizes, you recompute the embeddings. To select which of the
                                   n_vocab_samples (0, 1, 2, ..., n_vocab_samples-1) that will happen, we sample indices
                                   again from a normalised index axis, meaning t=0.0 is the first vocabulary size and t=1.0 the last.
                                   This schedule applies time dilation to that axis. In parts of the schedule that increase
                                   more slowly, the selected indices will be denser.
        :param n_embedding_samples: How many equidistant samples to take on the index scale.
        """
        super().__init__(preprocessor=initial_artifacts.preprocessorEffective())
        self._initial_artifacts = initial_artifacts

        import sage_tokenizer  # Just to check that you have it.

        self.vocabulary_points: list[int] = [
            round(vocabulary_schedule.get(i/(n_vocab_samples-1))) for i in range(n_vocab_samples)  # This round() produces vocabulary sizes in the thousands. The -1 is because we want to normalise the N samples 0, 1, ..., N-1 such that the extrema are 0.0 and 1.0.
        ]
        self.recompute_embeddings_at: list[int] = [self.vocabulary_points[j] for j in sorted({
            round(embedding_schedule.get(i/(n_embedding_samples-1)) * (n_vocab_samples - 2)) for i in range(n_embedding_samples)  # This round produces integers between 0 and len(self.vocabulary_points)-2, i.e. all possible indices into self.vocabulary_points except the last one (since SaGe STOPS at the last vocab size, so it's pointless to recompute embeddings at that point).
        })]

        self.seed = seed

    def _cacheSubfolder(self) -> str:
        return "sage"

    def _cacheType(self):
        return CacheableSageArtifacts

    def _identifierPartial(self) -> str:
        return shash(repr(self.preprocessor)) + "_" + shash(repr(self.vocabulary_points) + repr(self.recompute_embeddings_at))

    @classmethod
    def _toHexString(cls, typ: str) -> str:
        return typ.encode(encoding="utf-8").hex()

    def _vocabulariseFromWords(self, word_iterable: NamedIterable[tuple[str,int]]) -> CacheableSageArtifacts:
        raise RuntimeError("SaGe operates on contextual corpora, not on word lists.")

    def _vocabulariseFromSentences(self, sentence_iterable: NamedIterable[str]) -> CacheableSageArtifacts:
        from sage_tokenizer import SaGeVocabBuilder, setSageFolder

        hex_vocab = list(map(SageVocabulariser._toHexString, self._initial_artifacts.getVocabulary()))

        builder = SaGeVocabBuilder(
            full_vocab_schedule=self.vocabulary_points,
            embeddings_schedule=self.recompute_embeddings_at,
            max_len=max(len(t)//2 for t in hex_vocab),  # Each byte is represented by two hex digits, so the vocabulary in hex form is twice as long.

            random_seed=self.seed
        )

        setSageFolder(self._cachePath(sentence_iterable.name))

        hex_vocab_path = builder.build_vocab(
            experiment_name="sage",
            initial_vocabulary=deduplicate([bytes([i]).hex() for i in range(256)] + hex_vocab),
            corpus=self._preprocessSentencesToSentences(sentence_iterable, sep=" "),

            k_corpus_examples=None,
            corpus_cache="",  # Don't use corpus caching. Slower, but it is what you would expect by coming from an iterable.
            do_log_stdout=True
        )

        types = []
        with open(hex_vocab_path, "r", encoding="utf-8") as handle:
            for line in handle.readlines():
                hex_string = line.rstrip()
                utf8_bytes = bytes.fromhex(hex_string)

                # We skip all single-byte results, because these are purely there for internal reasons.
                if len(utf8_bytes) == 1:
                    continue

                type_string = utf8_bytes.decode("utf-8")  # It is impossible that this cannot decode because the bytes came from UTF-8 encoding.
                types.append(type_string)

        return CacheableSageArtifacts(types=list(deduplicate(self.preprocessor.getAlphabet() + types)))
