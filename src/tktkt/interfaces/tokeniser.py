from abc import ABC, abstractmethod
from typing import List, Dict

from .preparation import Preprocessor


Vocab = Dict[str, int]


class Tokeniser(ABC):

    def __init__(self, preprocessor: Preprocessor):
        self.preprocessor = preprocessor

    @abstractmethod
    def tokenise(self, pretoken: str) -> List[str]:
        pass

    def prepareAndTokenise(self, text: str) -> List[str]:
        tokens = []
        for pretoken in self.preprocessor.do(text):
            tokens.extend(self.tokenise(pretoken))
        return tokens

    def getName(self):  # Can be overridden if your name varies depending on the configuration.
        return self.__class__.__name__


class TokeniserWithVocab(Tokeniser):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab):
        super().__init__(preprocessor)
        self.vocab = vocab
