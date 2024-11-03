"""
Implements random sampling from the segmentation graph with objects.
Generalisation of the logic in GRaMPa.
"""
from typing import List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
import numpy.random as npr

from ...util.strings import segmentUsingIndices
from ...util.arrays import *


@dataclass
class SegmentationGraph:
    pointers: List[List[int]]
    probabilities: List[List[float]]


class SegmentationGraphSampler(ABC):

    def __init__(self, probabilities_to_probabilities: BatchNormalisation=IdentityBatchNormalisation()):
        self.rng = npr.default_rng(0)
        self.renormalisation = probabilities_to_probabilities

    @abstractmethod
    def samplePath(self, graph: SegmentationGraph) -> List[int]:
        pass

    @abstractmethod
    def samplePathAndProb(self, graph: SegmentationGraph) -> Tuple[List[int], float]:
        pass

    @abstractmethod
    def pathToProb(self, graph: SegmentationGraph, path: List[int]) -> float:
        pass


class ForwardGraphSampler(SegmentationGraphSampler):

    def samplePath(self, graph: SegmentationGraph) -> List[int]:
        indices = []
        current_node = 0
        while current_node < len(graph.pointers):
            indices.append(current_node)
            current_node = self.rng.choice(graph.pointers[current_node], p=self.renormalisation.normalise(np.array(graph.probabilities[current_node])))
        return indices

    def samplePathAndProb(self, graph: SegmentationGraph) -> Tuple[List[int], float]:
        indices     = []
        probability = 1

        current_node = 0
        while current_node < len(graph.pointers):
            indices.append(current_node)

            local_distribution = self.renormalisation.normalise(np.array(graph.probabilities[current_node]))
            new_pointer_index  = self.rng.choice(len(graph.pointers[current_node]), p=local_distribution)

            current_node = graph.pointers[current_node][new_pointer_index]
            probability *= local_distribution[new_pointer_index]

        return indices, probability

    def pathToProb(self, graph: SegmentationGraph, path: List[int]) -> float:
        probability = 1

        token_index  = 0
        current_node = 0
        while current_node < len(graph.pointers):
            next_node = path[token_index]

            local_distribution = self.renormalisation.normalise(np.array(graph.probabilities[current_node]))
            new_pointer_index = graph.pointers[current_node].index(next_node)  # Will throw an error when the path is illegal in this graph.

            current_node = next_node
            probability *= local_distribution[new_pointer_index]
            token_index += 1

        return probability


class BackwardGraphSampler(SegmentationGraphSampler):

    def samplePath(self, graph: SegmentationGraph) -> List[int]:
        indices = []
        current_node = len(graph.pointers) - 1
        while current_node > 0:
            current_node = self.rng.choice(graph.pointers[current_node], p=self.renormalisation.normalise(np.array(graph.probabilities[current_node])))  # "Greedy": rather than sampling across the full segmentation, we sample across one step and commit to it without care for the future. As long as there is a path to the start, its entry in p is nonzero.
            indices.append(current_node)
        return indices[::-1]

    def samplePathAndProb(self, graph: SegmentationGraph) -> Tuple[List[int], float]:
        indices     = []
        probability = 1

        current_node = len(graph.pointers) - 1
        while current_node > 0:
            local_distribution = self.renormalisation.normalise(np.array(graph.probabilities[current_node]))
            new_pointer_index  = self.rng.choice(len(graph.pointers[current_node]), p=local_distribution)

            current_node = graph.pointers[current_node][new_pointer_index]
            probability *= local_distribution[new_pointer_index]

            indices.append(current_node)

        return indices[::-1], probability

    def pathToProb(self, graph: SegmentationGraph, path: List[int]) -> float:
        probability = 1

        token_index  = len(path) - 1
        current_node = len(graph.pointers)
        while current_node > 0:
            next_node = path[token_index]

            local_distribution = self.renormalisation.normalise(np.array(graph.probabilities[current_node]))
            new_pointer_index = graph.pointers[current_node].index(next_node)  # Will throw an error when the path is illegal in this graph.

            current_node = next_node
            probability *= local_distribution[new_pointer_index]
            token_index -= 1

        return probability


class GraphTokeniser:

    def __init__(self, sampler: SegmentationGraphSampler):
        self.sampler = sampler

    @abstractmethod
    def generateGraph(self, pretoken: str) -> SegmentationGraph:  # You could cache this.
        pass

    def tokenise(self, pretoken: str) -> List[str]:
        graph = self.generateGraph(pretoken)
        indices = self.sampler.samplePath(graph)
        return segmentUsingIndices(pretoken, starts_of_tokens=indices)
