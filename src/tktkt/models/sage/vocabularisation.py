"""
Builds a vocabulary using the SaGe algorithm described in
    https://aclanthology.org/2023.eacl-main.45.pdf

During vocabulary building, data are kept in-context rather than coming from a word frequency list. The resulting
vocabulary is just a set of subwords without context, however.
"""
from pathlib import Path
from typing import Tuple, List

from sage_tokenizer import SaGeVocabBuilder, setSageFolder

from ...interfaces.vocabulariser import Vocabulariser, Preprocessor, UnidentifiedVocab, NamedIterable
from .schedules import *


class SageVocabulariser(Vocabulariser):

    def __init__(self, preprocessor: Preprocessor, seed: int=0,
                 vocabulary_schedule: Schedule=DoubleLinearSchedule(262144, 65536, 16384, t_mid=0.5), n_vocab_samples: int=13,
                 embedding_schedule: Dilation=ExponentialDilation(1.35), n_embedding_samples: int=6):
        """
        Note that a substantial part of the SaGe paper was dedicated to describing the hyperparameters it introduced,
        whereas the current implementation at https://github.com/MeLeLbgu/SaGe, which is SaGe v2.0, has lost most of
        those due to speedups making them obsolete.

        Hence, you will not see hyperparameters V, n, k, l, m and M in this constructor. That means the default values
        of n = 1.25, l = 4, k = 100, M = 1500, m = 10 have also become obsolete.
        """
        super().__init__(name="sage", preprocessor=preprocessor)

        self.vocabulary_points: List[int] = [
            round(vocabulary_schedule.get(i/(n_vocab_samples-1))) for i in range(n_vocab_samples)
        ]
        self.recompute_embeddings_at: List[int] = [self.vocabulary_points[j] for j in sorted({
            round(embedding_schedule.get(i/(n_embedding_samples-1)) * (n_vocab_samples - 2)) for i in range(n_embedding_samples)
        })]

        self.initial_hex_vocab = None
        self.seed = seed

    def initialiseVocabulary(self, vocab: UnidentifiedVocab):
        self.initial_hex_vocab = set(map(SageVocabulariser._toHexString, vocab))

    @classmethod
    def _toHexString(cls, typ: str) -> str:
        return typ.encode(encoding="utf-8").hex()

    @classmethod
    def _load(cls, file_or_folder: Path) -> UnidentifiedVocab:
        with open(file_or_folder, "r", encoding="utf-8") as handle:
            return [bytes.fromhex(line).decode(encoding="utf-8") for line in handle]

    def _vocabulariseFromWords(self, word_iterable: NamedIterable[Tuple[str,int]]) -> Path:
        raise RuntimeError("SaGe operates on contextual corpora, not on word lists.")

    def _vocabulariseFromSentences(self, sentence_iterable: NamedIterable[str]) -> Path:
        if not self.initial_hex_vocab:
            raise RuntimeError("SaGe vocabulary wasn't yet initialised.")

        builder = SaGeVocabBuilder(
            full_vocab_schedule=self.vocabulary_points,
            embeddings_schedule=self.recompute_embeddings_at,
            max_len=max(len(t) for t in self.initial_hex_vocab),

            random_seed=self.seed
        )

        folder = self._makeOutputFolder()
        setSageFolder(folder.with_stem(folder.stem + "_" + sentence_iterable.name))

        return builder.build_vocab(
            experiment_name="sage",
            initial_vocabulary=self.initial_hex_vocab,
            corpus=sentence_iterable,  # TODO: Possible replace this by self._preprocessSentencesToListsAsStrings(sentence_iterable).

            k_corpus_examples=None,
            corpus_cache=""  # Don't use corpus caching. Slower, but it is what you would expect by coming from an iterable.
        )
