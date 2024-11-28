from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Union, List

from ..interfaces.tokeniser import Tokeniser


T = TypeVar("T", bound=Tokeniser)

class TokeniserBuilder(Generic[T], ABC):
    """
    Object that instantiates a tokeniser with mostly default parameters.
    """

    @abstractmethod
    def buildTokeniser(self) -> T:
        pass
