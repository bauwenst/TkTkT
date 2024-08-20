from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from ..interfaces.tokeniser import Tokeniser


T = TypeVar("T", bound=Tokeniser)

class TokeniserBuilder(Generic[T], ABC):

    @abstractmethod
    def buildTokeniser(self) -> T:
        pass
