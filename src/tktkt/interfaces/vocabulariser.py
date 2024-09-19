from abc import ABC, abstractmethod
from typing import Set, Dict, Callable, Iterable, Tuple, Optional, Union
from pathlib import Path
from collections import Counter

from modest.formats.tsv import iterateTsv
from datasets import Dataset

from .preparation import Preprocessor
from ..files.paths import TkTkTPaths
from ..util.timing import datetimeDashed
from ..util.types import Comparable


UnidentifiedVocab = Iterable[str]  # Vocabulary without identifiers.
Vocab = Dict[str, int]

TokenSortingKey = Callable[[str], Comparable]


class Vocabulariser(ABC):
    """
    Builds subword vocabularies.
    """

    def __init__(self, name: str, preprocessor: Preprocessor):
        self._name = name
        self.preprocessor = preprocessor

    @classmethod
    @abstractmethod
    def _load(cls, file_or_folder: Path) -> UnidentifiedVocab:
        """
        Load a vocabulary trained with this vocabulariser from its save path.
        Depending on the vocabulariser, this can be a file or folder.
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

    # Pre-implemented backend methods

    def _makeOutputFolder(self) -> Path:
        """
        Get a new folder in which to store any files you want to store during vocabularisation.
        """
        return TkTkTPaths.extend(TkTkTPaths.pathToModels(), [self._name, f"{self._name}_{datetimeDashed()}"])

    @classmethod
    def _assignIdentifiers(cls, vocab: UnidentifiedVocab, sorting_key: Optional[TokenSortingKey], starting_id: int=0) -> Vocab:
        return {t:i for i,t in enumerate(sorted(vocab, key=sorting_key), start=starting_id)}

    # User-facing interface

    def vocabulariseFromTsv(self, word_frequency_tsv: Path) -> Path:
        return self._vocabulariseFromWords((word, int(count)) for word, count in iterateTsv(word_frequency_tsv, verbose=True))

    def vocabulariseFromCounter(self, word_frequency_counter: Counter) -> Path:
        return self._vocabulariseFromWords(word_frequency_counter.items())

    def vocabulariseFromStringIterable(self, string_iterable: Iterable[str]) -> Path:
        return self._vocabulariseFromSentences(string_iterable)

    def vocabulariseFromHf(self, dataset: Dataset, text_field: str):
        return self._vocabulariseFromSentences(example[text_field] for example in dataset)

    @classmethod
    def load(cls, file_or_folder: Path, sorting_key: Optional[TokenSortingKey], existing_types: Union[Vocab,UnidentifiedVocab]=None) -> Vocab:
        if existing_types is None:
            existing_types = dict()

        n_specials = len(existing_types)
        if isinstance(existing_types, dict):
            assert sorted(existing_types.values()) == list(range(n_specials))
        else:
            existing_types = {t: i for i,t in enumerate(existing_types)}

        tokeniser_vocab = cls._assignIdentifiers(cls._load(file_or_folder), sorting_key=sorting_key, starting_id=n_specials)
        for t in existing_types:
            if t in tokeniser_vocab:
                print(f"Warning: special token {t} (id: {existing_types[t]}) is already part of the vocabulary (id: {tokeniser_vocab[t]}). The latter id will be kept. "
                      f"In the future, there will likely be support to keep both at the same time.")

        return existing_types | tokeniser_vocab
