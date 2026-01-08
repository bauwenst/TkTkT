from typing import Iterable, TypeVar, Generic
from abc import ABC, abstractmethod

from ..interfaces.tokenisers import *
from ..util.iterables import arePositive, areContiguous
from ..util.strings import shash


T = TypeVar("T")

# class HashingMapping(ExtensibleMapping):
#
#     def __init__(self, hash_size: int):
#         super().__init__()
#         self.size = hash_size
#
#     def _get(self, key: str) -> int:
#         return hash(key) % self.size  # FIXME: CANINE has sharded embeddings and multi-hashing, so doesn't really fit this framework. Per embedding shard, it uses the function '((1 + ord(character)) * prime) % shard_size'.
#
#     def _keys(self) -> Iterable:
#         warnings.warn("Hashing mapping has an infinite key domain.")
#         return []
#
#     def _items(self) -> Iterable:
#         warnings.warn("Hashing mapping has an infinite key domain.")
#         return []
#
#     def _values(self) -> Iterable:
#         return range(self.size)


class _ShiftedIntegerMapping(ABC, Generic[T]):
    """
    Map from keys to integers, but shifted up whenever a reserved integers is encountered.

    That is: if the raw mapping is
        ##########
    and we reserve position 2, then the mapping becomes
        ##.########
    """

    def __init__(self, raw_size: int, reserved_integers: Iterable[int]):
        """
        :param size: Size of the range of integers WITHOUT counting the reserved ones.
        :param reserved_integers: These integers will not be used by .get(), and as a result, the entire
                                  range of .get() will shift up by the amount of reserved integers (without
                                  counting reserved integers that are so large nothing can collide with them).
        """
        assert arePositive(reserved_integers)
        self._size = raw_size
        self._reserved_integers = list(sorted(set(reserved_integers)))

        # Validate that the integers that   ###...#.....#
        last_image = raw_size - 1
        for r in reserved_integers:
            if r <= last_image:
                last_image += 1
        reserved_above_last = [i for i in reserved_integers if i > last_image]
        if reserved_above_last:
            assert min(reserved_above_last) == last_image + 1
            assert areContiguous(reserved_above_last)

    @abstractmethod
    def _get(self, key: T) -> int:
        """Raw image of the given key. Can be shifted up afterwards."""
        pass

    def get(self, key: T) -> int:
        id = self._get(key)
        for reserved in self._reserved_integers:
            if reserved <= id:
                id += 1
            else:
                break
        return id

    def size(self) -> int:
        return self._size + len(self._reserved_integers)

    def rangeShifted(self) -> Iterable[int]:
        return (i for i in range(self.size()) if i not in self._reserved_integers)


class ModuloMapping(_ShiftedIntegerMapping[int]):  # For testing.
    def _get(self, key: int) -> int:
        return key % self._size


class HashAndModuloMapping(_ShiftedIntegerMapping[T]):
    def _get(self, key: T) -> int:
        return shash(key) % self._size


class TokeniserWithHashingVocab(TokeniserWithVocabulary[WithSpecials]):
    """
    Wraps an existing Tokeniser such that it gets a vocabulary mapping from tokens to IDs.

    The Tokeniser is allowed to produce an infinite amount of strings as tokens, which will be hashed to a finite set of
    integers and hence have an ID defined for all of them (sometimes overlapping).

    Note that this class is a very special kind of TokeniserWithVocabulary. The latter is normally a tokeniser that
    actually actively knows and uses a fixed set of surface strings inside its algorithm, whilst this class wraps
    tokenisers that have no idea about such a set and thus also can't enforce that the produced tokens are inside one.
    """

    def __init__(self, tokeniser: Tokeniser, vocab_size: int, specials: WithSpecials, default_id_decodes: dict[int, str]=None):
        """
        :param default_id_decodes: An approximation of the reverse vocabulary mapping. An infinite amount of strings
                                   map to each ID, so an ID can technically not be decoded to one string.
                                   However, if you know the possible set of tokens for your application, you can enforce
                                   default decodings with a dictionary (although you should probably just be using a
                                   TokeniserWithVocabulary in that case).
        """
        self.hash = HashAndModuloMapping(vocab_size, list(specials))
        super().__init__(preprocessor=tokeniser.preprocessor, vocab=Vocab(
                ordered_types=[f"[ID={i}]" for i in range(self.hash.size()) if i not in specials],
                specials=specials,
                unk_id=None
            )
        )
        self.core = tokeniser
        self.default_decodings = default_id_decodes or dict()
        self._assertDefaultsMatch()

    def _assertDefaultsMatch(self):
        for id, typ in self.default_decodings.items():
            actual_id = self.typeToId(typ)
            if id != actual_id:
                raise ValueError(f"Default decoding of ID {id} is {typ}, yet this hashes to ID {actual_id}.")

    def tokenise(self, pretoken: str) -> Tokens:
        return self.core.tokenise(pretoken)

    ########################################################

    def types(self) -> Iterable[str]:  # No obligation to have this.
        raise NotImplementedError()

    def ids(self) -> Iterable[int]:
        return self.hash.rangeShifted()

    def typeToId(self, t: str) -> int:
        return self.hash.get(t)  # This cannot error nor give None (all strings have a vocab ID), so there is no need for UNK logic.

    def idToType(self, i: int) -> str:
        return self.default_decodings.get(i, f"[ID={i}]")

    def hasType(self, t: str) -> bool:
        return self.hasId(self.typeToId(t))

    def hasId(self, i: int) -> bool:
        return i in self.ids()
