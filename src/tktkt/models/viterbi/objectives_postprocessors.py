from typing import List

import numpy as np

from ...interfaces.tokeniser import Vocab
from .framework import ViterbiStepScoreGenerator, ViterbiStepScores, ViterbiStepScoreGeneratorWithTokens, ViterbiStepScoresWithTokens


class ConvertToProbabilities(ViterbiStepScoreGenerator):

    def __init__(self, nested_generator: ViterbiStepScoreGenerator):
        self.nested_generator = nested_generator

    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScores:
        scores = self.nested_generator.generateGrid(string, max_k)
        scores.grid = np.exp(scores.grid)  # e^ln(p) == p
        return scores


class VocabularyConstraintExact(ViterbiStepScoreGenerator):
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


from ...util.trie import TrieNode

class VocabularyConstraintAtLeastAll(ViterbiStepScoreGeneratorWithTokens):
    """
    A step is allowed if there is a subword in the vocab that is AT LEAST that step.
    Gives slightly more freedom than an exact constraint.
    """

    def __init__(self, nested_generator: ViterbiStepScoreGenerator, subword_vocabulary: Vocab, reset_value: float=0.0):
        self.nested_generator = nested_generator
        self.default = reset_value

        # Compile vocabulary
        self.vocab = TrieNode()
        for typ in subword_vocabulary:
            self.vocab.add(typ)
        self.vocab.compile()
        self.vocab.compileRoots()

    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScoresWithTokens:
        old_grid = self.nested_generator.generateGrid(string, max_k)
        new_grid = ViterbiStepScoresWithTokens(len(string), max_k)

        N = len(string)
        for n in range(N):
            for k in range(max_k):
                # Copy data from old grid
                new_grid.set(n, k, old_grid.get(n,k))

                if k >= min(max_k, N-n):  # Don't look up nor store strings if you don't need to.
                    new_grid.set(n, k, self.default)
                else:
                    # Get vocab type that is at least the current step
                    step = string[n:n+(k+1)]
                    step_with_possible_suffix = self.vocab.getNodesWithPrefix(step, only_first=True)
                    if not step_with_possible_suffix:
                        new_grid.set(n, k, self.default)
                    else:
                        new_grid.setToken(n, k, step_with_possible_suffix[0].root)

        return new_grid


class WithStrings(ViterbiStepScoreGeneratorWithTokens):

    def __init__(self, nested_generator: ViterbiStepScoreGenerator):
        self.nested_generator = nested_generator

    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScoresWithTokens:
        old_grid = self.nested_generator.generateGrid(string, max_k)
        new_grid = ViterbiStepScoresWithTokens(len(string), max_k)

        N = len(string)
        for n in range(N):
            for k in range(max_k):
                new_grid.set(n, k, old_grid.get(n,k))
                if k < min(max_k, N-n):
                    new_grid.setToken(n, k, string[n:n+(k+1)])

        return new_grid
