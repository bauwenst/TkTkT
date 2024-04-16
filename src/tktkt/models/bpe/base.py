from typing import List

from bpe_knockout.knockout.core import BTE, BteInitConfig, ByteBasedMode

from ...interfaces.preparation import TextMapper
from ...interfaces.tokeniser import Vocab
from ...preparation.spacemarking import SpaceMarker


MergeList = List[str]


class ClassicBPE(BTE):
    """
    BPE with binary merges (that's what the P stands for).
    """

    def __init__(self, vocab: Vocab, merges: MergeList,
                 boundary_marker: SpaceMarker, byte_based: bool=True, normaliser: TextMapper=None):
        super().__init__(
            BteInitConfig(
                bytebased=ByteBasedMode.NONE if not byte_based else ByteBasedMode.INPUT_TO_BYTES
            ),
            starting_vocab=vocab, starting_mergelist=merges,
            autorun_modes=True,
            holdout=None,
            quiet=True,
            normalisation=normaliser,
            boundary_marker=boundary_marker
        )
