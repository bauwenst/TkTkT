from typing import List

from bpe_knockout.auxiliary.tokenizer_interface import HuggingFaceTokeniserPath
from bpe_knockout.knockout.core import BTE, BteInitConfig, ByteBasedMode
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from ...interfaces.preparation import TextMapper, Preprocessor
from ...interfaces.tokeniser import Vocab
from ...preparation.spacemarking import BoundaryMarker
from ...preparation.huggingface import detectBoundaryMarker, detectByteBased, HuggingFacePreprocessor

MergeList = List[str]


class ClassicBPE(BTE):
    """
    BPE with binary merges (that's what the P stands for).
    """

    def __init__(self, preprocessor: Preprocessor, boundary_marker: BoundaryMarker,
                 vocab: Vocab, merges: MergeList):
        super().__init__(
            # Init
            BteInitConfig(),
            starting_vocab=vocab, starting_mergelist=merges,

            # Prep
            preprocessor=preprocessor,
            boundary_marker=boundary_marker,

            # Niche parameters
            autorun_modes=True,
            holdout=None,
            quiet=True
        )

    @staticmethod
    def fromHuggingFace(hf_bpe_tokenizer: PreTrainedTokenizerFast) -> "ClassicBPE":
        """
        Assuming the given tokeniser is a BPE tokeniser, convert it to a native TkTkT BPE tokeniser
        (rather than wrapping it).
        """
        vocab_and_merges = HuggingFaceTokeniserPath.fromTokeniser(hf_bpe_tokenizer)
        marker = detectBoundaryMarker(hf_bpe_tokenizer)
        # byte_based = detectByteBased(tokenizer)
        return ClassicBPE(
            preprocessor=HuggingFacePreprocessor(hf_bpe_tokenizer),

            vocab=vocab_and_merges.loadVocabulary(),
            merges=vocab_and_merges.loadMerges(),
            boundary_marker=marker,
            # byte_based=byte_based
        )