"""
Extension of N-gram tokenisers that allows degenerate segmentations by striding.
"""
from math import ceil
from ...interfaces.tokenisers import *


class StridedNgramTokeniser(Tokeniser):
    """
    N-gram tokeniser with potential overlap. For example, for stride=2 and N=3, turns a pretoken ABCDEFGH into
        ABC CDE EFG GH
    """

    def __init__(self, preprocessor: Preprocessor, N: int, stride: int):
        assert 0 < stride <= N
        super().__init__(preprocessor=preprocessor)
        self._N = N
        self._stride = stride

    def tokenise(self, pretoken: str) -> Tokens:
        return [pretoken[i*self._stride:i*self._stride+self._N]
                for i in range(max(0,ceil((len(pretoken)-self._N)/self._stride)) + 1)]


# ABCDEF, len = 6
# N = 2 stride = 2, then
# AB CD EF -> 3 iterations
# N = 4, stride = 3:
# ABCD
# DEF