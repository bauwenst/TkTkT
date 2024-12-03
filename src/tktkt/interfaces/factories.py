from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Union, List

from .preparation import Preprocessor
from .vocabulariser import Vocab, DEFAULT_FIVE_SPECIALS
from .tokeniser import Tokeniser


T = TypeVar("T", bound=Tokeniser)

class TokeniserFactory(Generic[T], ABC):
    """
    Object that instantiates a tokeniser with mostly default parameters.
    """

    @abstractmethod
    def buildTokeniser(self) -> T:
        pass


class Deserialiser(ABC):
    """
    Loads a specific instance of vocabularisation results stored on disk somewhere (usually remotely).

    Note that it is vocabularisers that already have the knowledge of how to store to disk (and therefore also how to
    load from disk), except when the file format exists somewhere out there but TkTkT doesn't support that training paradigm.

      - Vocabularisers represent an algorithm and file format, e.g. the BPE algorithm and the Sennrich format.

      - Deserialisers represent the result of an algorithm and file content, e.g. the BPE vocab and merges resulting from
        applying BPE with |V| = 32k to SlimPajama's first 3M examples.

      - TokeniserFactories, lastly, are objects that abstract over the constructor of a Tokeniser, often using a Deserialiser in the
        process, but with customisable parameters. Deserialisers have basically no customisability: they load one very specific
        instance. Different results come from different subclasses, not different initialisation.
    """

    def __init__(self, specials: Union[Vocab,List[str]]=DEFAULT_FIVE_SPECIALS.all_special_tokens):
        self._specials = specials
        self._vocab_cache = None

    def buildVocabulary(self) -> Vocab:
        if self._vocab_cache is None:
            self._vocab_cache = self._buildVocabulary()
        return self._vocab_cache

    @abstractmethod
    def _buildVocabulary(self) -> Vocab:
        pass

    @abstractmethod
    def preprocessor(self) -> Preprocessor:
        """
        The preprocessor that creates pretokens that can be tokenised into tokens of this vocabulary.
        If the vocabulariser had no built-in preprocessor, then this matches the preprocessor used during vocabularisation.
        This is e.g. not the case for packages like SentencePiece where spaces are converted into underscores after the
        TkTkT preprocessor has finished its work.
        """
        pass
