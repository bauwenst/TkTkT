from abc import ABC, abstractmethod

from ...interfaces.tokeniser import Tokeniser


class TokeniserBuilder(ABC):

    @abstractmethod
    def buildTokeniser(self) -> Tokeniser:
        pass
