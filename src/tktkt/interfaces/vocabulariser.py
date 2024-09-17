from abc import ABC, abstractmethod
from typing import Set, Dict, Callable, Iterable, Tuple
from pathlib import Path
from collections import Counter

from modest.formats.tsv import iterateTsv
from datasets import Dataset

from .preparation import Preprocessor
from ..files.paths import TkTkTPaths
from ..util.timing import datetimeDashed
from ..util.types import Comparable


UnorderedVocab = Set[str]
Vocab = Dict[str, int]


class Vocabulariser(ABC):
    """
    Builds subword vocabularies.
    """

    def __init__(self, name: str, preprocessor: Preprocessor):
        self._name = name
        self.preprocessor = preprocessor

    @abstractmethod
    def loadFromFolder(self, folder: Path) -> UnorderedVocab:
        """
        Load a vocabulary trained with this vocabulariser from its save folder.
        """
        pass

    @abstractmethod
    def _vocabulariseFromWords(self, word_iterable: Iterable[Tuple[str,int]]) -> Path:
        """
        Construct a subword vocabulary based on contextless words and frequencies, and save it to disk.

        The result is a folder path which can be given to loadFromFolder(). The reason why the result isn't a vocabulary
        set/dict is to allow the vocabulariser to save as many files as it wants and save them in any format it wants.
        """
        pass

    @abstractmethod
    def _vocabulariseFromSentences(self, sentence_iterable: Iterable[str]) -> Path:
        """
        Construct a subword vocabulary based on corpus sentences, and save it to disk.
        """
        pass

    def _makeOutputFolder(self) -> Path:
        """
        Get a new folder in which to store any files you want to store during vocabularisation.
        """
        return TkTkTPaths.extend(TkTkTPaths.pathToModels(), [self._name, f"{self._name}_{datetimeDashed()}"])

    # User-facing interface

    def vocabulariseFromTsv(self, word_frequency_tsv: Path) -> Path:
        return self._vocabulariseFromWords((word, int(count)) for word, count in iterateTsv(word_frequency_tsv, verbose=True))

    def vocabulariseFromCounter(self, word_frequency_counter: Counter) -> Path:
        return self._vocabulariseFromWords(word_frequency_counter.items())

    def vocabulariseFromStringIterable(self, string_iterable: Iterable[str]) -> Path:
        return self._vocabulariseFromSentences(string_iterable)

    def vocabulariseFromHf(self, dataset: Dataset, text_field: str):
        return self._vocabulariseFromSentences(example[text_field] for example in dataset)

    def assignIdentifiers(self, vocab: UnorderedVocab, sorting_key: Callable[[str],Comparable], starting_id: int=0) -> Vocab:
        return {t:i for i,t in enumerate(sorted(vocab, key=sorting_key), start=starting_id)}
