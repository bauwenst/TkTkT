import warnings
from abc import ABC, abstractmethod
from typing import List, Dict, Mapping

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

    def __init__(self, preprocessor: Preprocessor):
        super().__init__(preprocessor)

    @abstractmethod
    def typeToId(self, t: str) -> int:
        pass

    @abstractmethod
    def idToType(self, i: int) -> str:
        pass

    @abstractmethod
    def getVocabMapping(self, ) -> Mapping[str,int]:
        """
        Return an object which has the four methods .get(), .keys(), .values(), .items() representing the vocabulary.
        Does not have to be a dictionary!
        """
        pass

    @abstractmethod
    def getVocabSize(self) -> int:
        pass


class TokeniserWithVocabDict(TokeniserWithVocab):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab):
        super().__init__(preprocessor)
        self.vocab         = vocab
        self.reverse_vocab = {v:k for k,v in vocab.items()}

        if len(self.vocab) != len(self.reverse_vocab):
            warnings.warn("Tokeniser with non-injective vocabulary instantiated. This means some types will never be returned given their ID, since another type has the same ID.")

    def typeToId(self, t: str) -> int:
        return self.vocab.get(t)

    def idToType(self, i: int) -> str:
        return self.reverse_vocab.get(i)

    def getVocabMapping(self):
        return self.vocab

    def getVocabSize(self) -> int:
        return len(self.vocab)
