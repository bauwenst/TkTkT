"""
Builds a vocabulary using the SaGe algorithm described in
    https://aclanthology.org/2023.eacl-main.45.pdf

During vocabulary building, words are kept in the context of their sentences rather than coming from a word frequency
list. The resulting vocabulary is just a set of subwords without context, however.
"""
import warnings
from pathlib import Path
from typing import Tuple, List

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

        The schedule parameters replace SaGe v2.0's hardcoded schedules, which are respectively
            vocab: 262144 229376 196608 163840 131072 98304 65536 57344 49152 40960 32768 16384
            embed: 262144                      131072       65536       49152 40960 32768

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
        super().__init__(name="sage", preprocessor=preprocessor)

        import sage_tokenizer  # Just to check that you have it.

        self.vocabulary_points: List[int] = [
            round(vocabulary_schedule.get(i/(n_vocab_samples-1))) for i in range(n_vocab_samples)  # This round() produces vocabulary sizes in the thousands. The -1 is because we want to normalise the N samples 0, 1, ..., N-1 such that the extrema are 0.0 and 1.0.
        ]
        self.recompute_embeddings_at: List[int] = [self.vocabulary_points[j] for j in sorted({
            round(embedding_schedule.get(i/(n_embedding_samples-1)) * (n_vocab_samples - 2)) for i in range(n_embedding_samples)  # This round produces integers between 0 and len(self.vocabulary_points)-2, i.e. all possible indices into self.vocabulary_points except the last one (since SaGe STOPS at the last vocab size, so it's pointless to recompute embeddings at that point).
        })]

        self.initial_hex_vocab = None
        self.seed = seed

    def initialiseVocabulary(self, vocab: UnidentifiedVocab):
        """
        Set the initial tokens that the tokeniser can use to segment the output of the preprocessor.

        Under the hood, we will map them to bytes (stored and later read out in hex format). Note that this is not
        an issue even if the given vocabulary already uses pseudo-bytes: types in the vocabulary will become longer, yes,
        but since the preprocessor also maps input text to pseudo-bytes and those are converted to bytes by SaGe already,
        the input will contain the same bytes-of-pseudobytes units we get in the vocabulary.
        """
        self.initial_hex_vocab = set(map(SageVocabulariser._toHexString, vocab)) | {bytes([i]).hex() for i in range(256)}

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

        from sage_tokenizer import SaGeVocabBuilder, setSageFolder

        builder = SaGeVocabBuilder(
            full_vocab_schedule=self.vocabulary_points,
            embeddings_schedule=self.recompute_embeddings_at,
            max_len=max(len(t) for t in self.initial_hex_vocab),

            random_seed=self.seed
        )

        setSageFolder(self._makeOutputFolder(sentence_iterable.name))

        return builder.build_vocab(
            experiment_name="sage",
            initial_vocabulary=self.initial_hex_vocab,
            corpus=sentence_iterable,  # TODO: Possible replace this by self._preprocessSentencesToListsAsStrings(sentence_iterable).

            k_corpus_examples=None,
            corpus_cache="",  # Don't use corpus caching. Slower, but it is what you would expect by coming from an iterable.
            do_log_stdout=True
        )
