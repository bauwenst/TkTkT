import numpy as np

from ...interfaces.general import Vocab
from .framework import ViterbiStepScoreGenerator, ViterbiStepScores


class ConstrainVocabulary(ViterbiStepScoreGenerator):
    """
    Post-processor for a score grid that resets all steps that aren't allowed by the given subword vocabulary.
    """

    def __init__(self, nested_generator: ViterbiStepScoreGenerator, subword_vocabulary: Vocab, reset_value: float=0.0):
        self.nested_generator = nested_generator
        self.vocab = subword_vocabulary
        self.default = reset_value

    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScores:
        grid = self.nested_generator.generateGrid(string, max_k)
        for n in range(len(string)):
            for k in range(max_k):  # It doesn't really matter that for large n, n:n+k is the same string every iteration.
                if string[n:n+(k+1)] not in self.vocab:
                    grid.set(n,k, self.default)
        return grid


class ConvertToProbabilities(ViterbiStepScoreGenerator):

    def __init__(self, nested_generator: ViterbiStepScoreGenerator):
        self.nested_generator = nested_generator

    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScores:
        scores = self.nested_generator.generateGrid(string, max_k)
        scores.grid = np.exp(scores.grid)  # e^ln(p) == p
        return scores