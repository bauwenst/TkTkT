"""
Unguided score functions. These are informed by nothing except the substring being scored as-is, e.g. its
length or its presence in a given set.
"""
from .framework import *


class MinimiseTokenAmount(ViterbiStepScoreGenerator):

    def __init__(self, vocab):
        self.vocab = vocab

    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScores:
        N = len(string)
        scores = ViterbiStepScores(N, max_k, default=-INFTY)  # Note that the bottom-right triangle of the grid will never be used by the algorithm. It can have any value, doesn't matter, won't be used.
        for n in range(len(string)):
            for k in range(max_k):
                subword = string[n:n+(k+1)]
                if subword in self.vocab:
                    scores.set(n, k, -1)  # You want "maximise the sum of edges" to mean "minimise amount of edges". The more edges you walk over, the worse it gets.

        return scores


class MaximiseTokenLength(ViterbiStepScoreGenerator):

    def __init__(self, vocab):
        self.vocab = vocab

    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScores:
        N = len(string)
        scores = ViterbiStepScores(N, max_k, default=-INFTY)
        for n in range(len(string)):
            for k in range(max_k):
                subword = string[n:n+(k+1)]
                if subword in self.vocab:
                    scores.set(n, k, len(subword))

        return scores
