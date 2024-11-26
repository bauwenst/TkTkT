from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Union, List

from ..interfaces.vocabulariser import Vocab, DEFAULT_FIVE_SPECIALS
from ..interfaces.tokeniser import Tokeniser


T = TypeVar("T", bound=Tokeniser)

class TokeniserBuilder(Generic[T], ABC):
    """
    Object that instantiates a tokeniser with mostly default parameters.
    """

    @abstractmethod
    def buildTokeniser(self) -> T:
        pass

A = TypeVar("A")

class VocabularyBuilder(Generic[A], ABC):
    """
    Object that instantiates a vocabulary and additional objects.
    """

    def __init__(self, specials: Union[Vocab,List[str]]=DEFAULT_FIVE_SPECIALS.all_special_tokens):
        self._specials = specials

    @abstractmethod
    def buildVocabulary(self) -> Vocab:  # TODO: We may want to cache this.
        pass

    @abstractmethod
    def buildAdditionals(self) -> A:
        """For example: builds BPE merges, ULM probabilities, ..."""
        pass