from abc import ABC, abstractmethod

from ...interfaces.tokeniser import Tokeniser


class TokeniserFactory(ABC):

    @abstractmethod
    def buildTokeniser(self) -> Tokeniser:
        pass
