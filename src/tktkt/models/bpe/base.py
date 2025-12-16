from abc import abstractmethod
from functools import lru_cache

from bpe_knockout.model.tokeniser import BTE, MergeList  # This is for other files.

from ...interfaces.tokenisers import Tokens
from ...interfaces.identifiers import WithSpecials


class _ByteTupleEncoding(BTE[WithSpecials]):
    """
    Wrapper class around the bpe_knockout package's tokeniser, for easily adding methods.
    """

    def getName(self):
        return self.__class__.__name__

    @abstractmethod
    def isDeterministic(self) -> bool:
        pass


class _NonDeterministicBPETokeniser(_ByteTupleEncoding[WithSpecials]):
    """
    By default, the interface doesn't apply any caching to tokenisation. This supports non-determinism for e.g. BPE-dropout.
    """
    def isDeterministic(self) -> bool:
        return False


class _DeterministicBPETokeniser(_ByteTupleEncoding[WithSpecials]):
    """
    Under the condition that the same string is always tokenised the same, you can add a cache on top of the tokeniser.
    """
    def isDeterministic(self) -> bool:
        return True

    @lru_cache(maxsize=1024*1024)
    def tokenise(self, pretoken: str) -> Tokens:
        return super().tokenise(pretoken)

    def _syncWithGraph(self):
        # Because syncing is what changes the tokeniser, you must invalidate the LRU cache.
        self.tokenise.cache_clear()
        super()._syncWithGraph()


class ClassicBPE(_DeterministicBPETokeniser[WithSpecials]):
    """
    BPE with binary merges (that's what the P stands for).
    Technically the constructor allows merges with more than two types.
    """
    pass
