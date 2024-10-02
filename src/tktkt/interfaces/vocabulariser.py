from abc import ABC, abstractmethod
from typing import Dict, Callable, Iterable, Tuple, Optional, Union, List
from pathlib import Path
from collections import Counter

from datasets.arrow_dataset import DatasetInfoMixin
from modest.formats.tsv import iterateTsv

from .preparation import Preprocessor
from ..files.paths import TkTkTPaths
from ..util.timing import datetimeDashed
from ..util.types import Comparable, NamedIterable

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

    def _preprocessSentencesToSentences(self, sentence_iterable: NamedIterable[str]) -> NamedIterable[str]:
        """
        Run the preprocessor to get pretokens from sentences, and concatenate them with spaces.
        Some packages expect such a "list as string" explicitly (one example is GenSim). Others like SentencePiece,
        BPEasy and even HuggingFace tokenizers allow specifying a whitespace tokeniser, implicitly accepting pretoken lists this way.
        """
        return sentence_iterable.map(self.preprocessor.do).map(lambda pretokens: " ".join(pretokens))

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

        for word, count in word_iterable:
            word = " ".join(self.preprocessor.do(word))

            count       = int(count)  # Note: can't just multiply the word by this count, because you'll run into memory errors.
            max_at_once = max(1, LARGEST_STRING_LEN // (len(word) + 1))
            while count > 0:
                new_count = max(0, count - max_at_once)
                diff      = count - new_count
                count -= diff
                yield diff * (" " + word)

    def _makeOutputFolder(self) -> Path:
        """
        Get a new folder in which to store any files you want to store during vocabularisation.
        """
        return TkTkTPaths.extend(TkTkTPaths.pathToModels(), [self._name, f"{self._name}_{datetimeDashed()}"])

    @classmethod
    def _assignIdentifiers(cls, vocab: UnidentifiedVocab, sorting_key: Optional[TokenSortingKey], starting_id: int=0) -> Vocab:
        if sorting_key is None:  # Note that sorted(key=None) still sorts. Here, we allow using the raw order too.
            return {t:i for i,t in enumerate(vocab, start=starting_id)}
        else:
            return {t:i for i,t in enumerate(sorted(vocab, key=sorting_key), start=starting_id)}

    # User-facing interface

    def vocabulariseFromTsv(self, word_frequency_tsv: Path) -> Path:
        return self._vocabulariseFromWords(NamedIterable(
            ( (word, int(count)) for word, count in iterateTsv(word_frequency_tsv, verbose=True) ), name=word_frequency_tsv.stem
        ))

    def vocabulariseFromCounter(self, word_frequency_counter: Counter) -> Path:
        return self._vocabulariseFromWords(NamedIterable(word_frequency_counter.items(), name=""))

    def vocabulariseFromStringIterable(self, string_iterable: Union[NamedIterable[str],Iterable[str]]) -> Path:
        return self._vocabulariseFromSentences(string_iterable if isinstance(string_iterable, NamedIterable) else NamedIterable(string_iterable, name=""))

    def vocabulariseFromHf(self, dataset: DatasetInfoMixin, text_field: str):
        return self._vocabulariseFromSentences(NamedIterable(map(lambda example: example[text_field], dataset), name=dataset.info.dataset_name if dataset.info.dataset_name is not None else ""))

    @classmethod
    def load(cls, file_or_folder: Path, sorting_key: Optional[TokenSortingKey]=None, existing_types: Union[Vocab,UnidentifiedVocab]=None) -> Vocab:
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
