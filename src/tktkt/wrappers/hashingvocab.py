import warnings
from typing import Mapping, List, Iterable
from abc import ABC, abstractmethod

from ..interfaces.tokeniser import Tokeniser, TokeniserWithFiniteIdRange, Dict


class ExtensibleMapping(Mapping, ABC):

    def __init__(self):
        self.hardcoded = dict()

    @abstractmethod
    def _get(self, key):
        pass

    @abstractmethod
    def _keys(self) -> Iterable:
        pass

    @abstractmethod
    def _values(self) -> Iterable:
        pass

    @abstractmethod
    def _items(self) -> Iterable:
        pass

    def get(self, key):
        try:
            return self.hardcoded[key]
        except:
            try:
                return self._get(key)
            except:
                return None

    def set(self, key, value):
        original_value = value
        while value in set(self.values()):
            value += 1
        self.hardcoded[key] = value
        if value != original_value:
            warnings.warn(f"Requested to set value {original_value} for key {key}, but was increased to {value} due to collisions.")

    def keys(self):
        yield from self.hardcoded.keys()
        yield from self._keys()

    def values(self):
        yield from self.hardcoded.values()
        yield from self._values()

    def items(self):
        yield from self.hardcoded.items()
        yield from self._items()


class HashingMapping(ExtensibleMapping):

    def __init__(self, hash_size: int):
        super().__init__()
        self.size = hash_size

    def _get(self, key: str) -> int:
        return hash(key) % self.size  # FIXME: CANINE has sharded embeddings and multi-hashing, so doesn't really fit this framework. Per embedding shard, it uses the function '((1 + ord(character)) * prime) % shard_size'.

    def _keys(self) -> Iterable:
        warnings.warn("Hashing mapping has an infinite key domain.")
        return []

    def _items(self) -> Iterable:
        warnings.warn("Hashing mapping has an infinite key domain.")
        return []

    def _values(self) -> Iterable:
        return range(self.size)


class TokeniserWithHashingVocab(TokeniserWithFiniteIdRange):
    """
    Wraps an existing Tokeniser such that it gets a vocabulary mapping from tokens to IDs.

    The Tokeniser is allowed to produce an infinite amount of strings as tokens, which will be hashed to a finite set of
    integers and hence have an ID defined for all of them (sometimes overlapping).

    Note that this class has a fundamentally different usage than TokeniserWithVocabDict. The latter is used as the
    parent class for tokenisers that actually make use of a fixed set of types inside their algorithm, whilst this class
    is used as an afterthought to wrap tokenisers that don't use such a fixed set nor enforce that the produced tokens
    are inside one. Tokenisers that aren't defined as TokeniserWithVocabDict can't be wrapped into one, so if you still
    want a vocabulary for them, you should use this class (or possible future alternatives that are also dict-less).
    """

    def __init__(self, tokeniser: Tokeniser, vocab_size: int, default_id_decodes: Dict[int, str]=None):
        """
        :param default_id_decodes: An approximation of the reverse vocabulary mapping. An infinite amount of strings
                                   map to each ID, so an ID can technically not be decoded to one string.
                                   However, if you know the possible set of tokens for your application, you can enforce
                                   default decodings with a dictionary (although you should probably just be using a
                                   TokeniserWithVocabDict in that case).
        """
        super().__init__(tokeniser.preprocessor)
        self.core = tokeniser
        self.vocabulary_mapping = HashingMapping(vocab_size)
        self.default_decodings = default_id_decodes if default_id_decodes else dict()
        self._assertDefaultsMatch()

    def _assertDefaultsMatch(self):
        for id, typ in self.default_decodings.items():
            actual_id = self.typeToId(typ)
            if id != actual_id:
                raise ValueError(f"Default decoding of ID {id} is {typ}, yet this hashes to ID {actual_id}.")

    def tokenise(self, pretoken: str) -> List[str]:
        return self.core.tokenise(pretoken)

    ########################################################

    def typeToId(self, t: str) -> int:
        return self.vocabulary_mapping.get(t)  # This cannot error nor give None (all strings have a vocab ID), so there is no need for UNK logic.

    def ids(self) -> Iterable[int]:
        return self.vocabulary_mapping.values()

    def idToType(self, i: int) -> str:
        return self.default_decodings.get(i, f"[ID={i}]")

    def getVocabSize(self) -> int:
        return self.vocabulary_mapping.size + len(set(self.vocabulary_mapping.hardcoded.values()))
