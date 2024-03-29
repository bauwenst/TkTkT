"""
Unguided score functions. These are informed by nothing except the substring being scored as-is, e.g. its length.
"""
from .framework import *


class ConstantScore(ViterbiStepScoreGenerator):

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
