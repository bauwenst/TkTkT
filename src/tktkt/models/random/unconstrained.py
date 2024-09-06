"""
Purely random segmentation without a vocabulary.
"""
from typing import List
import numpy.random as npr

from ...interfaces.tokeniser import Tokeniser, Preprocessor
from ...util.strings import segmentUsingBitmap


class RandomSegmentation(Tokeniser):
    """
    Unconstrained random segmentation.
    Will give mostly small-token segmentation because there are more segmentations with small tokens.
    """

    def __init__(self, preprocessor: Preprocessor):
        super().__init__(preprocessor)
        self.rng = npr.default_rng(0)

    def tokenise(self, pretoken: str) -> List[str]:
        n_positions = len(pretoken) - 1
        n_segmentations = 2**n_positions

        segmentation_index = self.rng.integers(n_segmentations)
        bitmap = bin(segmentation_index)[2:].zfill(n_positions)
        return segmentUsingBitmap(pretoken, bitmap)
