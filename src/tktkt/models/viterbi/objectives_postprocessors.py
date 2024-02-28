from typing import List

import numpy as np

from ...interfaces.tokeniser import Vocab
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

    def getAllPossibleSegmentations(self, string: str, max_k: int) -> List[List[str]]:
        N = len(string)
        segmentations_to_here = [[] for _ in range(N+1)]
        segmentations_to_here[0].append("")
        for n in range(N):
            for segmentation_so_far in segmentations_to_here[n]:
                K = min(max_k, N-n)  # K is the amount of steps. When you're in front of character n == N-1, i.e. the last character, there is N-n == 1 more step.
                for k in range(K):
                    step = string[n:n+(k+1)]
                    if step in self.vocab:
                        segmentations_to_here[n+k+1].append(segmentation_so_far + " "*(segmentation_so_far != "") + step)

        return [segmentation.split(" ") for segmentation in segmentations_to_here[N]]


class ConvertToProbabilities(ViterbiStepScoreGenerator):

    def __init__(self, nested_generator: ViterbiStepScoreGenerator):
        self.nested_generator = nested_generator

    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScores:
        scores = self.nested_generator.generateGrid(string, max_k)
        scores.grid = np.exp(scores.grid)  # e^ln(p) == p
        return scores