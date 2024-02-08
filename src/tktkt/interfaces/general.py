from abc import ABC, abstractmethod
from typing import List, Dict

from ..preparation.splitters import Pretokeniser


Vocab = Dict[str, int]


class Tokeniser(ABC):

    def __init__(self, pretokeniser: Pretokeniser):
        self.pretokeniser = pretokeniser

    @abstractmethod
    def tokenise(self, pretoken: str) -> List[str]:
        pass

    def prepareAndTokenise(self, text: str) -> List[str]:
        tokens = []
        for pretoken in self.pretokeniser.splitSentence(text):
            tokens.extend(self.tokenise(pretoken))
        return tokens


class TokeniserWithVocab(Tokeniser):

    @abstractmethod
    def getVocab(self) -> Vocab:
        pass

    def getVocabSize(self) -> int:
        return len(self.getVocab())
