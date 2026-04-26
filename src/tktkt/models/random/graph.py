"""
Implements random sampling from the segmentation graph with objects.
Generalisation of the logic in GRaMPa.
"""
from typing import Optional
from abc import ABC, abstractmethod

import numpy as np
import numpy.random as npr

from ...interfaces.tokenisers import *
from ...util.strings import indicesToTokens
from ...util.arrays import *


class SegmentationGraph:
    """
    Weighted segmentation graph, i.e. a (|s|+1)-node DAG whose source-to-sink paths correspond 1:1 to possible
    segmentations of a string, with a probability distribution over next nodes at every node.

    The constructor merely asks for edge 'scores'. The probability distributions are computed on-the-fly to save on
    operations. At node i, given the outgoing scores s_{i,j} for j = 1 ... n_i, the probabilities sampled to get to
    the next node are

        p_{i,j} = renormalisation(s_{i,j} / sum_j s_{i,j})
    """
    def __init__(self, pointers: list[list[int]], scores: list[list[float]],
                 renormalisation: BatchNormalisation=IdentityBatchNormalisation(), precompute: bool=False):
        """
        :param pointers: For each node i, gives the list of next node indices.
        :param scores:   For each node i, gives the corresponding weight for the edge on each next node.
        :param precompute: Precompute all probability distributions and denominators.
        """
        self.pointers = pointers
        self.scores   = scores
        self.renormalisation = renormalisation

        # Caches and precomputations
        n_nodes = len(pointers)
        self._denominators: list[Optional[float]]     = [None for _ in range(n_nodes)]
        self._probabilities: list[Optional[np.array]] = [None for _ in range(n_nodes)]

        self.precomputed = precompute
        if precompute:
            for node in range(n_nodes):
                self.getProbabilities(node)  # Caches probabilities and denominators.

    def getProbabilities(self, node: int) -> np.array:
        cache = self._probabilities[node]
        if cache is not None:
            return cache

        distribution = self.renormalisation.normalise(np.array(self.scores[node]) / self.getDenominator(node))
        self._probabilities[node] = distribution
        return distribution

    def getDenominator(self, node: int) -> float:
        cache = self._denominators[node]
        if cache is not None:
            return cache

        denominator = sum(self.scores[node])  # This is faster than using np.sum() because segmentation graphs are never going to be on the order of 100s of characters.
        self._denominators[node] = denominator
        return denominator

    @abstractmethod
    def samplePath(self, rng: npr.Generator) -> list[int]:
        """
        Stochastically sample a path with probability equal to the product of the probabilities of the walked edges.
        Not the most probable path. Not rejection-sampled to compensate for the properties of the path probability
        distribution that emerges this way.
        """
        pass

    @abstractmethod
    def samplePathAndProb(self, rng: npr.Generator) -> tuple[list[int], float]:
        pass

    @abstractmethod
    def samplePathAndDenom(self, rng: npr.Generator) -> tuple[list[int], float]:
        pass

    @abstractmethod
    def pathToProb(self, path: list[int]) -> float:
        pass

    @abstractmethod
    def totalPaths(self) -> int:
        pass

    @abstractmethod
    def maximalDenominatorProduct(self) -> float:
        """
        Computes max_{path in paths} prod_{node in path} denom_node, a quantity useful for applications where
        it is desirable for the path probabilities to be a constant multiple of the product of the edge scores
        (by means of rejection sampling), see Appendix B in https://aclanthology.org/2025.acl-long.1180/.

        NOTE: This quantity is meaningless when any non-identity renormalisation is applied to the probabilities.
        """
        pass


class ForwardSegmentationGraph(SegmentationGraph):  # "Forward" refers to how it is sampled (left-to-right). It is likely constructed in reverse.

    def samplePath(self, rng: npr.Generator) -> list[int]:
        indices = []
        current_node = 0
        while current_node < len(self.pointers) - 1:  # The last node in every path is len(self.pointers) - 1 and it has no outbound arcs.
            indices.append(current_node)
            current_node = rng.choice(self.pointers[current_node], p=self.getProbabilities(current_node))
        return indices

    def samplePathAndProb(self, rng: npr.Generator) -> tuple[list[int], float]:
        indices     = []
        probability = 1

        current_node = 0
        while current_node < len(self.pointers) - 1:
            indices.append(current_node)

            local_distribution = self.getProbabilities(current_node)
            new_pointer_index  = rng.choice(len(self.pointers[current_node]), p=local_distribution)

            current_node = self.pointers[current_node][new_pointer_index]
            probability *= local_distribution[new_pointer_index]

        return indices, probability

    def samplePathAndDenom(self, rng: npr.Generator) -> tuple[list[int], float]:
        indices = []
        denominator = 1

        current_node = 0
        while current_node < len(self.pointers) - 1:
            indices.append(current_node)
            denominator *= self.getDenominator(current_node)
            current_node = rng.choice(self.pointers[current_node], p=self.getProbabilities(current_node))

        return indices, denominator

    def pathToProb(self, path: list[int]) -> float:
        path = path + [len(self.pointers)-1]  # If path already includes the final index, this doesn't hurt. Otherwise, it allows current_node to actually run until the final node.
        probability = 1

        token_index  = 0
        current_node = 0
        while current_node < len(self.pointers) - 1:
            token_index += 1
            next_node = path[token_index]

            local_distribution = self.getProbabilities(current_node)
            new_pointer_index = self.pointers[current_node].index(next_node)  # Will throw an error when the path is illegal in this graph.

            current_node = next_node
            probability *= local_distribution[new_pointer_index]

        return probability

    def totalPaths(self) -> int:
        n = len(self.pointers)
        totals = [0 for _ in range(n)]
        totals[0] = 1
        for i in range(n):
            for j in self.pointers[i]:
                totals[j] += totals[i]
        return totals[-1]

    def maximalDenominatorProduct(self) -> float:
        n = len(self.pointers)
        grid = [0.0 for _ in range(n)]
        grid[0] = 1
        for i in range(n):  # Index n only receives, it doesn't send
            best_to_i = grid[i]
            for j in self.pointers[i]:
                grid[j] = max(grid[j], best_to_i * self.getDenominator(j))
        return grid[-1]


class BackwardSegmentationGraph(SegmentationGraph):  # "Backward" refers to how it is sampled (right-to-left).

    def samplePath(self, rng: npr.Generator) -> list[int]:
        indices = []
        current_node = len(self.pointers) - 1
        while current_node > 0:  # The last node in every path is 0 and it has no outbound arcs.
            current_node = rng.choice(self.pointers[current_node], p=self.getProbabilities(current_node))  # "Greedy": rather than sampling across the full segmentation, we sample across one step and commit to it without care for the future. As long as there is a path to the start, its entry in p is nonzero.
            indices.append(current_node)
        return indices[::-1]

    def samplePathAndProb(self, rng: npr.Generator) -> tuple[list[int], float]:
        indices     = []
        probability = 1

        current_node = len(self.pointers) - 1
        while current_node > 0:
            local_distribution = self.getProbabilities(current_node)
            new_pointer_index  = rng.choice(len(self.pointers[current_node]), p=local_distribution)

            current_node = self.pointers[current_node][new_pointer_index]
            probability *= local_distribution[new_pointer_index]

            indices.append(current_node)

        return indices[::-1], probability

    def samplePathAndDenom(self, rng: npr.Generator) -> tuple[list[int], float]:
        indices = []
        denominator = 1

        current_node = len(self.pointers) - 1
        while current_node > 0:
            denominator *= self.getDenominator(current_node)
            current_node = rng.choice(self.pointers[current_node], p=self.getProbabilities(current_node))
            indices.append(current_node)

        return indices[::-1], denominator

    def pathToProb(self, path: list[int]) -> float:
        if path[-1] != len(self.pointers)-1:
            path = path + [len(self.pointers)-1]
        probability = 1

        token_index  = len(path) - 1
        current_node = len(self.pointers) - 1
        while current_node > 0:
            token_index -= 1
            next_node = path[token_index]

            local_distribution = self.getProbabilities(current_node)
            new_pointer_index = self.pointers[current_node].index(next_node)  # Will throw an error when the path is illegal in this graph.

            current_node = next_node
            probability *= local_distribution[new_pointer_index]

        return probability

    def totalPaths(self) -> int:
        n = len(self.pointers)
        totals = [0 for _ in range(n)]
        totals[-1] = 1
        for i in range(n-1,-1,-1):
            for j in self.pointers[i]:
                totals[j] += totals[i]
        return totals[0]

    def maximalDenominatorProduct(self) -> float:
        n = len(self.pointers)
        grid = [0.0 for _ in range(n)]
        grid[-1] = 1
        for i in range(n-1, 0, -1):  # Index 0 only receives, doesn't send
            best_to_i = grid[i]
            for j in self.pointers[i]:
                grid[j] = max(grid[j], best_to_i * self.getDenominator(j))
        return grid[0]


class GraphTokeniser(TokeniserWithVocabulary[WithSpecials]):
    """
    Tokeniser that walks a path through a segmentation graph to find a segmentation.
    The default implementation does a single-pass sample where the probability of a path is the product of its edge probabilities.
    """

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab[WithSpecials], seed: int=0):
        super().__init__(preprocessor=preprocessor, vocab=vocab)
        self.rng = npr.default_rng(seed)

    @abstractmethod
    def generateGraph(self, pretoken: str) -> SegmentationGraph:  # You could cache this.
        pass

    def tokenise(self, pretoken: str) -> Tokens:
        graph = self.generateGraph(pretoken)
        indices = graph.samplePath(self.rng)
        return indicesToTokens(pretoken, starts_of_tokens=indices)
