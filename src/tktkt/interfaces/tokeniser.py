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
    def getVocabMapping(self) -> Mapping[str,int]:
        """
        Return an object which has the four methods .get(), .keys(), .values(), .items() representing the vocabulary.
        Does not have to be a dictionary!
        """
        pass

    @abstractmethod
    def getVocabSize(self) -> int:
        pass


class TokeniserWithVocabDict(TokeniserWithVocab):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, unk_type: str=None):
        super().__init__(preprocessor)
        self.vocab         = vocab
        self.reverse_vocab = {v:k for k,v in vocab.items()}
        self.unk = unk_type

        if self.unk is not None and self.unk not in self.vocab:
            raise ValueError(f"The given vocabulary does not have an ID defined for the given UNK type '{unk_type}'.")
        if len(self.vocab) != len(self.reverse_vocab):
            warnings.warn("Tokeniser with non-injective vocabulary instantiated. This means some types will never result from decoding their ID, since another type has the same ID.")

    def typeToId(self, t: str) -> int:
        try:  # try-except is the fastest method to check+return a key in use cases where most lookups are valid (https://stackoverflow.com/a/28860508/9352077).
            return self.vocab[t]
        except:
            if self.unk is not None:
                return self.vocab[self.unk]
            else:
                raise RuntimeError(f"Unknown vocabulary type '{t}' cannot be converted to ID, and no UNK is defined.")

    def idToType(self, i: int) -> str:
        try:
            return self.reverse_vocab[i]
        except:
            raise RuntimeError(f"Unknown ID {i} cannot be decoded to a string type.")

    def getVocabMapping(self):
        return self.vocab

    def getVocabSize(self) -> int:
        return len(self.vocab)
