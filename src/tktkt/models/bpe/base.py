from typing import List
from typing_extensions import Self
from abc import abstractmethod

from functools import lru_cache

from bpe_knockout.auxiliary.tokenizer_interface import HuggingFaceTokeniserPath
from bpe_knockout import *
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from ...interfaces.preparation import TextMapper, Preprocessor
from ...interfaces.tokeniser import Vocab
from ...preparation.boundaries import BoundaryMarker
from ...preparation.huggingface import detectBoundaryMarker, detectByteBased, HuggingFacePreprocessor

MergeList = List[str]


class SimplifiedBTEInterface(BTE):
    """
    Wrapper class around the BPE-knockout library for easier usage.
    """

    def __init__(self, preprocessor: Preprocessor, boundary_marker: BoundaryMarker,
                 vocab: Vocab, merges: MergeList, do_morphemic_knockout: bool, unk_type: str=None):
        super().__init__(
            # Init
            BteInitConfig() if not do_morphemic_knockout else BteInitConfig(knockout=RefMode.MORPHEMIC),
            starting_vocab=vocab, starting_mergelist=merges,
            unk_type=unk_type,

            # Prep
            preprocessor=preprocessor,
            boundary_marker=boundary_marker,

            # Niche parameters
            autorun_modes=True,
            holdout=None,
            quiet=True
        )

    @classmethod
    def fromHuggingFace(cls, hf_bpe_tokenizer: PreTrainedTokenizerFast) -> Self:
        """
        Assuming the given tokeniser is a BPE tokeniser, convert it to a native TkTkT BPE tokeniser
        (rather than wrapping it).
        """
        vocab_and_merges = HuggingFaceTokeniserPath.fromTokeniser(hf_bpe_tokenizer)
        marker = detectBoundaryMarker(hf_bpe_tokenizer)
        return cls(
            preprocessor=HuggingFacePreprocessor(hf_bpe_tokenizer),

            vocab=vocab_and_merges.loadVocabulary(),
            merges=vocab_and_merges.loadMerges(),
            boundary_marker=marker,

            unk_type=hf_bpe_tokenizer.unk_token,
            do_morphemic_knockout=False
        )

    def getName(self):
        return self.__class__.__name__

    @abstractmethod
    def isDeterministic(self) -> bool:
        pass


class NonDeterministicBPETokeniser(SimplifiedBTEInterface):
    """
    By default, the interface doesn't apply any caching to tokenisation. This supports non-determinism for e.g. BPE-dropout.
    """
    def isDeterministic(self) -> bool:
        return False


class DeterministicBPETokeniser(SimplifiedBTEInterface):
    """
    Under the condition that the same string is always tokenised the same, you can add a cache on top of the tokeniser.
    """
    def isDeterministic(self) -> bool:
        return True

    @lru_cache(maxsize=1024*1024)
    def tokenise(self, pretoken: str) -> List[str]:
        return super().tokenise(pretoken)

    def _syncWithGraph(self):
        # Because syncing is what changes the tokeniser, you must invalidate the LRU cache.
        self.tokenise.cache_clear()
        super()._syncWithGraph()


class ClassicBPE(DeterministicBPETokeniser):
    """
    BPE with binary merges (that's what the P stands for).
    Technically the constructor allows merges with more than two types.
    """
    pass
