from abc import ABC, abstractmethod
from typing import Dict, Callable, Iterable, Tuple, Optional, Union, List
from pathlib import Path
from collections import Counter

from transformers import SpecialTokensMixin
from modest.formats.tsv import iterateTsv

from .preparation import Preprocessor
from ..paths import TkTkTPaths
from ..util.timing import datetimeDashed
from ..util.types import Comparable, NamedIterable, HuggingfaceDataset
from ..util.iterables import streamProgress

UnidentifiedVocab = Iterable[str]  # Vocabulary without identifiers, but in some order.
Vocab = Dict[str, int]

TokenSortingKey = Callable[[str], Comparable]

DEFAULT_FIVE_SPECIALS = SpecialTokensMixin(
    pad_token="<pad>",
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    mask_token="<mask>"
)  # The argument mapping is reconstructed with .special_tokens_map; the list of values is .all_special_tokens


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

    def _preprocessSentencesToSentences(self, sentence_iterable: NamedIterable[str], sep: str=" ") -> NamedIterable[str]:
        """
        Run the preprocessor to get pretokens from sentences, and concatenate them with spaces.
        Some packages expect such a "list as string" explicitly (one example is GenSim). Others like SentencePiece,
        BPEasy and even HuggingFace tokenizers allow specifying a whitespace tokeniser, implicitly accepting pretoken lists this way.
        """
        return sentence_iterable.map(self.preprocessor.do).map(lambda pretokens: sep.join(pretokens))

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

    def _preprocessWordsToPretokens_approx(self, word_iterable: NamedIterable[Tuple[str,int]]) -> NamedIterable[Tuple[str,int]]:
        """
        Apply the preprocessor onto the given words, concatenate the resulting pretokens, and return the result with
        the given counts. Approximates _preprocessWordsToPretokens_counter without loading all pretokens into memory at once.
        """
        return word_iterable.map(lambda tup: ("".join(self.preprocessor.do(tup[0])), tup[1]))

    def _preprocessWordsToPretokens_counter(self, word_iterable: NamedIterable[Tuple[str,int]]) -> NamedIterable[Tuple[str,int]]:
        """
        Apply the preprocessor to each word, count the pretokens separately, and return the pretoken counts.
        This requires loading all pretokens into memory.
        """
        counter = Counter()
        for word,count in streamProgress(word_iterable, "Counting pretokens"):
            for pretoken in self.preprocessor.do(word):
                counter[pretoken] += count
        return NamedIterable(counter.most_common(), name=word_iterable.name)

    def _makeOutputFolder(self, extra_suffix: str="") -> Path:
        """
        Get a new folder in which to store any files you want to store during vocabularisation.
        """
        return TkTkTPaths.extend(TkTkTPaths.pathToModels(), [self._name, f"{self._name}_{extra_suffix}_{datetimeDashed()}"])

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
        return self._vocabulariseFromSentences(string_iterable if isinstance(string_iterable, NamedIterable) else NamedIterable(string_iterable, name="[unnamed]"))

    def vocabulariseFromHf(self, dataset: HuggingfaceDataset, text_field: str) -> Path:
        return self._vocabulariseFromSentences(
            NamedIterable(dataset, name=dataset.info.dataset_name if dataset.info.dataset_name is not None else "")
                .map(lambda example: example[text_field])
        )

    @classmethod
    def load(cls, file_or_folder: Path, existing_types: Union[Vocab,UnidentifiedVocab]=None, extras_first: bool=False, sorting_key: Optional[TokenSortingKey]=None) -> Vocab:
        """
        Load a vocabulary, i.e. a mapping from strings to integers, from a file/folder stored by this vocabulariser.
        If you declare extra types with an identifier, those identifiers get priority assignment.
        If you declare extra types without identifier, you can choose whether you want them in the front or the back
        of the vocabulary. In both cases, when an extra type exists in the rest of the vocabulary, it is ignored there.
        """
        if existing_types is None:
            existing_types = dict()

        # Reserve IDs (jump over these when you enumerate identifiers) and types (pretend like these don't exist)
        if isinstance(existing_types, dict):
            reserved_types = set(existing_types.keys())
            reserved_ids = set(existing_types.values())
        else:
            existing_types = list(existing_types)
            reserved_types = set(existing_types)
            reserved_ids   = set(range(len(existing_types))) if extras_first else set()

        # Do the actual loading.
        vocabulary = dict()
        id = 0
        for typ in cls._load(file_or_folder):
            while id in reserved_ids:
                id += 1

            if typ in reserved_types:
                print(f"Warning: special token {typ} is part of the vocabulary. "
                      f"In the future, there will likely be support to keep both at the same time. "
                      f"For now, we will keep the newly requested ID and overwrite its place in the original vocabulary.")
                continue
            elif typ in vocabulary:
                print(f"Warning: token {typ} was generated more than once in the {cls.__name__}. Skipping its duplicate.")
                continue

            vocabulary[typ] = id
            id += 1

        # Finally, add the reserved types.
        if isinstance(existing_types, dict):
            vocabulary |= existing_types
            assert list(sorted(vocabulary.values())) == list(range(len(vocabulary))), f"Some of the special tokens {existing_types} fall outside the contiguous vocabulary range."
        else:
            id = 0 if extras_first else len(vocabulary)
            for typ in existing_types:
                vocabulary[typ] = id
                id += 1

        return vocabulary
        # tokeniser_vocab = cls._assignIdentifiers(cls._load(file_or_folder), sorting_key=sorting_key, starting_id=len(missing_types))
        # special_vocab   = {t: i for i,t in enumerate(missing_types)}
        # return special_vocab | tokeniser_vocab
