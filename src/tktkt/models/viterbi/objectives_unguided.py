"""
Unguided score functions. These are informed by nothing except the substring being scored as-is, e.g. its length.
"""
from .framework import *


class RandomScoreGenerator(ViterbiStepScoreGenerator):
    """
    Dummy score generator with random step scores.
    """

    def __init__(self, min: float=0, max: float=1):
        self.min = min
        self.max = max

        import numpy.random as npr
        self.rng = npr.default_rng(0)

    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScores:
        scores = ViterbiStepScores(len(string), max_k, default=-INFTY)
        self.rng.random(out=scores.grid, dtype=np.float32)
        scores.grid = self.min + (self.max - self.min)*scores.grid
        return scores


class ConstantScore(ViterbiStepScoreGenerator):
    """
    Sets a constant score for each step smaller than K characters.
    To minimise the amount of tokens used to segment a word, you should accumulate these scores with subtraction,
    so that more tokens lead to a more negative score and hence will be less likely picked by the maximisation framework.
    """

    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScores:
        N = len(string)
        scores = ViterbiStepScores(N, max_k, default=+INFTY)  # Note that the bottom-right triangle of the grid will never be used by the algorithm. It can have any value, doesn't matter, won't be used.
        for n in range(N):
            for k in range(max_k):
                scores.set(n, k, 1)  # You want "maximise the sum of edges" to mean "minimise amount of edges". The more edges you walk over, the worse it gets.

        return scores


class TokenLength(ViterbiStepScoreGenerator):

    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScores:
        N = len(string)
        scores = ViterbiStepScores(N, max_k, default=-INFTY)
        for n in range(N):
            for k in range(max_k):
                scores.set(n, k, k+1)

        return scores
