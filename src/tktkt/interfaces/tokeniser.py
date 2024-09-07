import warnings
from abc import ABC, abstractmethod
from typing import List, Dict, Mapping, Iterable

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


class TokeniserWithFiniteIdRange(Tokeniser):

    @abstractmethod
    def typeToId(self, t: str) -> int:
        pass

    @abstractmethod
    def ids(self) -> Iterable[int]:
        pass

    def hasId(self, i: int) -> bool:
        return i in self.ids()

    def getVocabSize(self) -> int:
        return len(set(self.ids()))


class TokeniserWithFiniteTypeDomain(TokeniserWithFiniteIdRange):

    @abstractmethod
    def idToType(self, i: int) -> str:
        pass

    @abstractmethod
    def types(self) -> Iterable[str]:
        pass

    def hasType(self, t: str) -> bool:
        return t in self.types()


class TokeniserWithVocabDict(TokeniserWithFiniteTypeDomain):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, unk_type: str=None):
        super().__init__(preprocessor)
        self.vocab         = vocab
        self.reverse_vocab = {v:k for k,v in vocab.items()}
        self.unk = unk_type

        self._accept_all_types = False  # if True, overrides hasType() by always returning True.

        if self.unk is not None and not self.hasType(self.unk):
            raise ValueError(f"The given vocabulary does not have an ID defined for the given UNK type '{unk_type}'.")
        if len(self.vocab) != len(self.reverse_vocab):
            warnings.warn("Tokeniser with non-injective vocabulary instantiated. This means some types will never result from decoding their ID, since another type has the same ID.")

    ########################################################

    def typeToId(self, t: str) -> int:
        try:  # try-except is the fastest method to check+return a key in use cases where most lookups are valid (https://stackoverflow.com/a/28860508/9352077).
            return self.vocab[t]
        except:
            if self.unk is not None:
                return self.vocab[self.unk]
            else:
                raise RuntimeError(f"Unknown vocabulary type '{t}' cannot be converted to ID, and no UNK is defined.")

    def types(self) -> Iterable[str]:
        return self.vocab.keys()

    def hasType(self, t: str) -> bool:
        return self._accept_all_types or t in self.vocab

    def enableInfiniteDomain(self, enable: bool):
        """
        If set to True, .types() and .ids() will still output the actual domain and range of the dictionary vocab,
        but .hasType() will always return True.

        This is technically incorrect, but has its uses. In particular, use this if you want to test the behaviour of a
        TokeniserWithVocabDict when its output isn't constrained.
        """
        self._accept_all_types = enable

    ########################################################

    def idToType(self, i: int) -> str:
        try:
            return self.reverse_vocab[i]
        except:
            raise RuntimeError(f"Unknown ID {i} cannot be decoded to a string type.")

    def ids(self) -> Iterable[int]:
        return self.vocab.values()

    def hasId(self, i: int) -> bool:
        return i in self.reverse_vocab

    ########################################################

    def getVocabSize(self) -> int:
        return len(self.reverse_vocab)
