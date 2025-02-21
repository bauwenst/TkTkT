from typing import List
from typing_extensions import Self
from abc import abstractmethod

from functools import lru_cache

from bpe_knockout import *
from bpe_knockout.auxiliary.tokenizer_interface import HuggingFaceTokeniserPath
from bpe_knockout.knockout.core import MergeList
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from ...interfaces.preparation import TextMapper, Preprocessor
from ...interfaces.tokeniser import Vocab
from ...preparation.boundaries import BoundaryMarker
from ...preparation.huggingface import detectBoundaryMarkerFromTokeniser, HuggingFacePreprocessor, HuggingFacePreprocessorForWords


class SimplifiedBTEInterface(BTE):
    """
    Wrapper class around the BPE-knockout library that abstracts away the config.
    """

    def __init__(self, preprocessor: Preprocessor, boundary_marker: BoundaryMarker,
                 vocab: Vocab, merges: MergeList,
                 do_morphemic_knockout: bool=False, do_reification: bool=False, backwards_compatible: bool=False, iterations: int=1,
                 unk_type: str=None):
        """
        :param backwards_compatible: Whether to stay within the same initial vocab or not. Knockout always does, reification doesn't.
        """
        config = BteInitConfig(iterations=iterations)
        if do_morphemic_knockout:
            config.knockout = RefMode.MORPHEMIC
        if do_reification:
            if backwards_compatible:
                config.reify = ReifyMode.BACKWARDS_COMPATIBLE
            else:
                config.reify = ReifyMode.ALL

        super().__init__(
            # Init
            init_config=config,
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
    @abstractmethod
    def fromHuggingFace(cls, hf_bpe_tokenizer: PreTrainedTokenizerFast, for_words: bool=True) -> Self:
        """
        Assuming the given tokeniser is a BPE tokeniser, convert it to a native TkTkT BPE tokeniser
        (rather than wrapping it).
        """
        pass

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

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, merges: MergeList, boundary_marker: BoundaryMarker,
                 unk_type: str=None):
        super().__init__(
            preprocessor=preprocessor,
            boundary_marker=boundary_marker,

            vocab=vocab,
            merges=merges,
            unk_type=unk_type,

            do_morphemic_knockout=False,
            do_reification=False
        )

    @classmethod
    def fromHuggingFace(cls, hf_bpe_tokenizer: PreTrainedTokenizerFast, for_words: bool=True) -> Self:
        vocab_and_merges = HuggingFaceTokeniserPath.fromTokeniser(hf_bpe_tokenizer)
        marker = detectBoundaryMarkerFromTokeniser(hf_bpe_tokenizer)
        return cls(
            preprocessor=HuggingFacePreprocessorForWords(hf_bpe_tokenizer) if for_words else HuggingFacePreprocessor(hf_bpe_tokenizer),

            vocab=vocab_and_merges.loadVocabulary(),
            merges=vocab_and_merges.loadMerges(),
            boundary_marker=marker,

            unk_type=hf_bpe_tokenizer.unk_token,
        )
