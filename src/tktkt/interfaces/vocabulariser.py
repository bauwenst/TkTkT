from typing import Callable, Iterable, Tuple, Optional, Union, List
from abc import ABC, abstractmethod
from pathlib import Path
from collections import Counter

from modest.formats.tsv import iterateTsv
from modest.interfaces.datasets import ModestDataset

from .identifiers import NoSpecials, UnidentifiedVocab, Vocab, WithSpecials
from .preparation import Preprocessor
from ..paths import TkTkTPaths
from ..util.iterables import streamProgress
from ..util.timing import datetimeDashed
from ..util.types import Comparable, NamedIterable, HuggingfaceDataset, anypartial

__all__ = ["Vocabulariser", "UnsupervisedVocabulariser", "SegmentationSupervisedVocabulariser",
           "Preprocessor", "NamedIterable", "UnidentifiedVocab"]


class Vocabulariser(ABC):
    """
    Builds subword vocabularies.
    """

    def __init__(self, name: str):
        self._name = name

    def _makeOutputFolder(self, extra_suffix: str="") -> Path:
        """
        Get a new folder in which to store any files you want to store during vocabularisation.
        """
        return TkTkTPaths.extend(TkTkTPaths.pathToModels(), [self._name, self._name + f"_{extra_suffix}"*bool(extra_suffix) + f"_{datetimeDashed()}"])

    @classmethod
    @abstractmethod
    def _load(cls, file_or_folder: Path) -> UnidentifiedVocab:
        """
        Load a vocabulary trained with this vocabulariser from its save path.
        Depending on the vocabulariser, this can be a file or folder.
        """
        pass

    @classmethod
    def load(cls, file_or_folder: Path, specials: WithSpecials=NoSpecials(), unk_type: Optional[int]=0, filtered_types: set[str]=None, type_sorting_key: Optional[Callable[[str], Comparable]]=None) -> Vocab[WithSpecials]:
        """
        Combines the Vocabulariser-specific knowledge of ._load() with the abilities to choose
            1. which of the loaded types to remove;
            2. the order of the remaining types;
            3. which specials to add to them.
        """
        filtered_types = filtered_types or set()
        types = sorted(cls._load(file_or_folder), key=type_sorting_key) if type_sorting_key is not None else cls._load(file_or_folder)
        return Vocab(filter(lambda t: t not in filtered_types, types), specials, unk_id=unk_type)


class UnsupervisedVocabulariser(Vocabulariser):
    """
    Vocabulariser which consumes unlabelled text (either as sentences or as words) and produces a vocabulary out of it.
    """

    def __init__(self, name: str, preprocessor: Preprocessor):
        super().__init__(name=name)
        self.preprocessor = preprocessor

    @abstractmethod
    def _vocabulariseFromWords(self, word_iterable: NamedIterable[Tuple[str,int]]) -> Path:
        """
        Construct a subword vocabulary based on contextless words and frequencies, and save it to disk.

        The result is a folder path which can be given to loadFromFolder(). The reason why the result isn't a vocabulary
        set/dict is to allow the vocabulariser to save as many files as it wants and save them in any format it wants.
        """
        pass

    @abstractmethod
    def _vocabulariseFromSentences(self, sentence_iterable: NamedIterable[str]) -> Path:
        """
        Construct a subword vocabulary based on corpus sentences, and save it to disk.
        """
        pass

    # Pre-implemented backend methods

    def _preprocessSentencesToPretokens(self, sentence_iterable: NamedIterable[str]) -> NamedIterable[List[str]]:
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

    def _preprocessSentencesToPretokenCounts(self, sentence_iterable: NamedIterable[str]) -> NamedIterable[Tuple[str,int]]:
        counter = Counter()
        for word in streamProgress(sentence_iterable, "Counting pretokens"):
            counter.update(self.preprocessor.do(word))
        return NamedIterable(list(counter.items()), name=sentence_iterable.name)

    def _preprocessWordsToSentences(self, word_iterable: NamedIterable[Tuple[str, int]]) -> NamedIterable[str]:
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

        def generate():
            for word, count in word_iterable:
                word = " ".join(self.preprocessor.do(word))

                count       = int(count)  # Note: can't just multiply the word by this count, because you'll run into memory errors.
                max_at_once = max(1, LARGEST_STRING_LEN // (len(word) + 1))
                while count > 0:
                    new_count = max(0, count - max_at_once)
                    diff      = count - new_count
                    count -= diff
                    yield diff * (" " + word)

        return NamedIterable(generate(), name=word_iterable.name)

    def _preprocessWordsToTsv(self, word_iterable: NamedIterable[Tuple[str,int]]) -> NamedIterable[str]:
        return word_iterable.map(lambda tup: f"{tup[0]}\t{tup[1]}\n")

    def _preprocessWordsToPretokenCounts_approx(self, word_iterable: NamedIterable[Tuple[str,int]]) -> NamedIterable[Tuple[str,int]]:
        """
        Apply the preprocessor onto the given words, CONCATENATE the resulting pretokens, and return the result with
        the given counts. Loses the pretoken boundaries, but unlike _preprocessWordsToPretokens_counter, you don't have
        to keep more than the current word in memory.
        """
        return word_iterable.map(lambda tup: ("".join(self.preprocessor.do(tup[0])), tup[1]))

    def _preprocessWordsToPretokenCounts(self, word_iterable: NamedIterable[Tuple[str,int]]) -> NamedIterable[Tuple[str,int]]:
        """
        Apply the preprocessor to each word, count the pretokens separately, and return the pretoken counts.
        This requires loading all pretokens into memory.
        """
        counter = Counter()
        for word,count in streamProgress(word_iterable, "Counting pretokens"):
            for pretoken in self.preprocessor.do(word):
                counter[pretoken] += count
        return NamedIterable(list(counter.items()), name=word_iterable.name)

    # User-facing interface

    def vocabulariseFromTsv(self, word_frequency_tsv: Path, name_instead_of_stem: str="") -> Path:
        return self._vocabulariseFromWords(NamedIterable(
            [(word,int(count)) for word, count in iterateTsv(word_frequency_tsv, verbose=True)], name=name_instead_of_stem or word_frequency_tsv.stem
        ))

    def vocabulariseFromCounter(self, word_frequency_counter: Counter, name: str="[unnamed_counter]") -> Path:
        return self._vocabulariseFromWords(NamedIterable(word_frequency_counter.items(), name=name))

    def vocabulariseFromStringIterable(self, string_iterable: Union[NamedIterable[str],Iterable[str]], name_if_not_named: str="[unnamed_iterable]") -> Path:
        return self._vocabulariseFromSentences(string_iterable if isinstance(string_iterable, NamedIterable) else NamedIterable(string_iterable, name=name_if_not_named))

    def vocabulariseFromHf(self, dataset: HuggingfaceDataset, text_field: str, name_if_not_named: str="[unnamed_HF]") -> Path:
        return self._vocabulariseFromSentences(
            NamedIterable(dataset, name=dataset.info.dataset_name if dataset.info.dataset_name is not None else name_if_not_named)
                .map(anypartial(dict.get, ..., text_field))  # Equivalent to `lambda example: example[text_field]` but this one can be pickled.
        )


class SegmentationSupervisedVocabulariser(Vocabulariser):
    """
    Vocabulariser which takes in a corpus of pre-segmented strings and builds a vocabulary based on that.
    """

    @abstractmethod
    def vocabulariseFromModest(self, reference: ModestDataset) -> Path:
        pass
