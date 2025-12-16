"""
Purely random segmentation without a vocabulary.

All constrained random tokenisers have the precondition that there exists at least one segmentation, otherwise they will
break in some way.
    - The generator-based one will generate nothing and choose a number between 0 and 0 not including 0.
    - The rejection-sampling-based one will sample forever.
    - The Markov-based one will hit a node during decoding which has 0 incoming edges and hence you're stuck.

There are many ways to solve this, most obviously using pseudobytes (they won't be interpretable but at least your
tokeniser doesn't crash), and otherwise by checking beforehand if at least all characters are in the vocab and if
not you return the characters and let them be [UNK]'ed.
"""
import numpy.random as npr

from ...interfaces.tokenisers import *
from ...util.strings import bitstringToTokens


class RandomSegmentation(Tokeniser):
    """
    Unconstrained random segmentation with no other functionality.

    Will give mostly small-token segmentation because there are more segmentations with small tokens.
    If you want to be able to bias the results towards another distribution, use any of the other random
    tokenisers that are found in this file's parent folder, and call .enableInfiniteDomain(True) on them.
    """

    def __init__(self, preprocessor: Preprocessor):
        super().__init__(preprocessor)
        self.rng = npr.default_rng(0)

    def tokenise(self, pretoken: str) -> Tokens:
        n_positions = len(pretoken) - 1
        n_segmentations = 2**n_positions

        segmentation_index = self.rng.integers(n_segmentations)
        bitmap = bin(segmentation_index)[2:].zfill(n_positions)
        return bitstringToTokens(pretoken, bitmap)
