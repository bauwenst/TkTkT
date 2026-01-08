from typing import Iterable, Union, TypeVar
from abc import ABC, abstractmethod
from pathlib import Path
from collections import Counter

from modest.formats.tsv import iterateTsv
from modest.interfaces.datasets import ModestDataset

from . import Preprocessor
from .artifactories import Artifacts, CacheableArtifacts
from .identifiers import UnidentifiedVocab
from ..paths import TkTkTPaths
from ..util.iterables import streamProgress
from ..util.strings import prefixIfNotEmpty
from ..util.types import NamedIterable, HuggingfaceDataset, anypartial, generated
from ..util.interfaces import Cache

__all__ = ["Vocabulariser", "UnsupervisedVocabulariser", "SegmentationSupervisedVocabulariser",
           "Preprocessor", "NamedIterable", "UnidentifiedVocab"]


T_CacheableArtifact = TypeVar("T_CacheableArtifact", bound=CacheableArtifacts)

class Vocabulariser(Cache[T_CacheableArtifact], ABC):
    """
    Builds subword vocabularies.
    """

    def __init__(self, preprocessor: Preprocessor, disable_cache: bool=False, disambiguator: str=""):
        super().__init__(disable_cache=disable_cache, disambiguator=disambiguator)
        self.preprocessor = preprocessor

    @abstractmethod
    def _cacheSubfolder(self) -> str:
        """A short string used for defining this Vocabulariser's output subdirectory."""
        pass

    def _cachePath(self, external_identifier: str) -> Path:
        """
        Get a new folder in which to store any files you want to store during vocabularisation.
        """
        return TkTkTPaths.pathToModels(self._cacheSubfolder(), self._cacheSubfolder() + prefixIfNotEmpty("_", self._identifierFull(external_identifier)))

    def _cacheFinalise(self, loaded: T_CacheableArtifact) -> T_CacheableArtifact:
        loaded.setPreprocessors(self.preprocessor)
        return loaded


class UnsupervisedVocabulariser(Vocabulariser[T_CacheableArtifact]):
    """
    Vocabulariser which consumes unlabelled text (either as sentences or as words) and produces a vocabulary out of it.
    """

    # Core computation

    @abstractmethod
    def _vocabulariseFromWords(self, word_iterable: NamedIterable[tuple[str,int]]) -> T_CacheableArtifact:
        """
        Construct a subword vocabulary based on contextless words and frequencies, and save it to disk.

        The result is a folder path which can be given to loadFromFolder(). The reason why the result isn't a vocabulary
        set/dict is to allow the vocabulariser to save as many files as it wants and save them in any format it wants.
        """
        pass

    @abstractmethod
    def _vocabulariseFromSentences(self, sentence_iterable: NamedIterable[str]) -> T_CacheableArtifact:
        """
        Construct a subword vocabulary based on corpus sentences, and save it to disk.
        """
        pass

    # Pre-implemented backend methods

    def _preprocessSentencesToPretokens(self, sentence_iterable: NamedIterable[str]) -> NamedIterable[list[str]]:
        """
        Run the preprocessor to get pretokens from sentences.
        """
        return sentence_iterable.map(self.preprocessor.do)

    def _preprocessSentencesToSentences(self, sentence_iterable: NamedIterable[str], sep: str=" ") -> NamedIterable[str]:
        """
        Run the preprocessor to get pretokens from sentences, and concatenate them with spaces.
        Some packages expect such a "list as string" explicitly (one example is GenSim). Others like SentencePiece,
        BPEasy and even HuggingFace tokenizers allow specifying a whitespace tokeniser, implicitly accepting pretoken lists this way.
        """
        return sentence_iterable.map(self.preprocessor.do).map(sep.join)

    def _preprocessSentencesToPretokenCounts(self, sentence_iterable: NamedIterable[str]) -> NamedIterable[tuple[str,int]]:
        counter = Counter()
        def iterator():
            if not counter:
                for word in streamProgress(sentence_iterable, "Counting pretokens"):
                    counter.update(self.preprocessor.do(word))

            yield from counter.items()
        return NamedIterable(generated(iterator), name=sentence_iterable.name)

    def _preprocessWordsToSentences(self, word_iterable: NamedIterable[tuple[str, int]]) -> NamedIterable[str]:
        """
        Converts an iterable
            apple 5
            banana-split 3
            ...
        to sentences
            Ġapple Ġapple Ġapple Ġapple Ġapple
            Ġbanana - split Ġbanana - split Ġbanana - split
        """
        LARGEST_STRING_LEN = 1_000

        def iterator():
            for word, count in word_iterable:
                word = " ".join(self.preprocessor.do(word))

                count       = int(count)  # Note: can't just multiply the word by this count, because you'll run into memory errors.
                max_at_once = max(1, LARGEST_STRING_LEN // (len(word) + 1))
                while count > 0:
                    new_count = max(0, count - max_at_once)
                    diff      = count - new_count
                    count -= diff
                    yield diff * (" " + word)

        return NamedIterable(generated(iterator), name=word_iterable.name)

    def _preprocessWordsToTsv(self, word_iterable: NamedIterable[tuple[str,int]]) -> NamedIterable[str]:
        return word_iterable.map(lambda tup: f"{tup[0]}\t{tup[1]}\n")

    def _preprocessWordsToPretokenCounts_approx(self, word_iterable: NamedIterable[tuple[str,int]]) -> NamedIterable[tuple[str,int]]:
        """
        Apply the preprocessor onto the given words, CONCATENATE the resulting pretokens, and return the result with
        the given counts. Loses the pretoken boundaries, but unlike _preprocessWordsToPretokens_counter, you don't have
        to keep more than the current word in memory.
        """
        return word_iterable.map(lambda tup: ("".join(self.preprocessor.do(tup[0])), tup[1]))

    def _preprocessWordsToPretokenCounts(self, word_iterable: NamedIterable[tuple[str,int]]) -> NamedIterable[tuple[str,int]]:
        """
        Apply the preprocessor to each word, count the pretokens separately, and return the pretoken counts.
        This requires loading all pretokens into memory.
        """
        counter = Counter()
        def iterator():
            if not counter:
                for word,count in streamProgress(word_iterable, "Counting pretokens"):
                    for pretoken in self.preprocessor.do(word):
                        counter[pretoken] += count
            yield from counter.items()
        return NamedIterable(generated(iterator), name=word_iterable.name)

    # User-facing interface (all cached)

    def vocabulariseFromTsv(self, word_frequency_tsv: Path, name_instead_of_stem: str="") -> T_CacheableArtifact:
        iterable = NamedIterable(
            generated(lambda: ((word,int(count)) for word, count in iterateTsv(word_frequency_tsv, verbose=True))),
            name=name_instead_of_stem or word_frequency_tsv.stem
        )
        return self._cacheRun(iterable.name, lambda: self._vocabulariseFromWords(iterable))

    def vocabulariseFromCounter(self, word_frequency_counter: Counter, name: str="[unnamed_counter]") -> T_CacheableArtifact:
        iterable = NamedIterable(generated(lambda: word_frequency_counter.items()), name=name)
        return self._cacheRun(iterable.name, lambda: self._vocabulariseFromWords(iterable))

    def vocabulariseFromWordIterable(self, word_iterable: Union[NamedIterable[tuple[str,int]],Iterable[tuple[str,int]]], name_if_not_named: str="[unnamed_iterable]") -> T_CacheableArtifact:
        iterable = word_iterable if isinstance(word_iterable, NamedIterable) else NamedIterable(word_iterable, name=name_if_not_named)
        return self._cacheRun(iterable.name, lambda: self._vocabulariseFromWords(iterable))

    def vocabulariseFromStringIterable(self, string_iterable: Union[NamedIterable[str],Iterable[str]], name_if_not_named: str="[unnamed_iterable]") -> T_CacheableArtifact:
        iterable = string_iterable if isinstance(string_iterable, NamedIterable) else NamedIterable(string_iterable, name=name_if_not_named)
        return self._cacheRun(iterable.name, lambda: self._vocabulariseFromSentences(iterable))

    def vocabulariseFromHf(self, dataset: HuggingfaceDataset, text_field: str, name_if_not_named: str="[unnamed_HF]") -> T_CacheableArtifact:
        iterable = NamedIterable(dataset, name=dataset.info.dataset_name if dataset.info.dataset_name is not None else name_if_not_named) \
            .map(anypartial(dict.get, ..., text_field))  # Equivalent to `lambda example: example[text_field]` but this one can be pickled.
        return self._cacheRun(iterable.name, lambda: self._vocabulariseFromSentences(iterable))


class SegmentationSupervisedVocabulariser(Vocabulariser[T_CacheableArtifact]):
    """
    Vocabulariser which takes in a corpus of pre-segmented strings and builds a vocabulary based on that.
    """

    # Core computation

    @abstractmethod
    def _vocabulariseFromModest(self, reference: ModestDataset) -> T_CacheableArtifact:
        pass

    # User-facing

    def vocabulariseFromModest(self, reference: ModestDataset) -> T_CacheableArtifact:
        return self._cacheRun(reference.identifier(), lambda: self._vocabulariseFromModest(reference))
