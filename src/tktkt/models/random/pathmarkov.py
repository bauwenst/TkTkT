from typing import List, Tuple
import numpy as np
import numpy.random as npr

from .generationbased import TokeniserWithVocabDict, Preprocessor, Vocab
from ...util.arrays import *
from ...util.strings import segmentUsingIndices


class RandomVocabSegmentation_GreedyMarkov(TokeniserWithVocabDict):
    """
    First computes the graph of possible token paths through the given string, then samples a random path
    by starting at the end of the string, choosing one random incoming token with probability proportional to how
    many paths it causes to arrive at the current node, following that token back and repeating this process.
    """

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, unk_type: str=None,
                 probabilities_to_probabilities: BatchNormalisation=IdentityBatchNormalisation(), minimal_token_length: int=1, decode_backwards: bool=False):
        """
        :param probabilities_to_probabilities: Transformation to apply to the Markov graph's probabilities, which by default
                                               are the fraction of paths entering through each incoming arc (causing the
                                               tokeniser to sample uniformly across all segmentations, of which there are
                                               more with smaller tokens than with larger tokens).
        :param minimal_token_length: Shortest token that can be produced *unless* no token of at least this length
                                     can be chosen at the current character during decoding, in which case we use the largest token that can be.
        :param decode_backwards: If True, the Markov graph is constructed forwards (i.e. every node remembers how many
                                 paths exist to it from the start) and decoding is done backwards. It is hypothesised
                                 that for any probability transformation that isn't an identity, this encourages longer
                                 tokens at the end vs. at the start.
        """
        super().__init__(preprocessor, vocab, unk_type)
        self.rng = npr.default_rng(0)
        self.renormalisation = probabilities_to_probabilities
        self.min_len = minimal_token_length
        self.decode_backwards = decode_backwards

    def tokenise(self, pretoken: str) -> List[str]:
        if self.decode_backwards:
            edges, weights = self.constructMarkovGraphForwards(pretoken)

            indices = []
            current_index = len(edges)-1
            while current_index > 0:
                current_index = self.rng.choice(edges[current_index], p=self.renormalisation.normalise(np.array(weights[current_index])))  # "Greedy": rather than sampling across the full segmentation, we sample across one step and commit to it without care for the future. As long as there is a path to the start, its entry in p is nonzero.
                indices.append(current_index)
            indices = indices[::-1]
        else:
            edges, weights = self.constructMarkovGraphBackwards(pretoken)  # Note: you can't decode forwards using the forwards Markov graph, because in that case, the node weight on the other end of each arc that hits a node has little relevance to the amount of paths passing through that node.

            indices = []
            current_index = 0
            while current_index < len(pretoken):
                indices.append(current_index)
                current_index = self.rng.choice(edges[current_index], p=self.renormalisation.normalise(np.array(weights[current_index])))

        return segmentUsingIndices(pretoken, starts_of_tokens=indices)

    def constructMarkovGraphForwards(self, pretoken: str) -> Tuple[List[List[int]],List[List[float]]]:  # Separate method so that it can be cached.
        """
        Produces two lists:
            - For each position in the string, the prior positions in the string that can reach it with 1 token.
            - For each position in the string, the fraction of all possible paths (starting at the start) arriving there
              that arrive by using the corresponding token in the first list.
        """
        # Get raw counts and arcs in the graph.
        options_to_get_before_char = [0 for _ in range(len(pretoken)+1)]
        options_to_get_before_char[0] = 1
        backpointers = [[] for _ in range(len(pretoken)+1)]
        for i in range(len(pretoken)):
            for j in range(i+1, len(pretoken)+1):
                if (j-i >= self.min_len or not backpointers[j]) and self.hasType(pretoken[i:j]):  # not j+1 because we step to BEFORE character j, so it is an exclusive bound
                    options_to_get_before_char[j] += options_to_get_before_char[i]
                    backpointers[j].append(i)

        # Assign a normalised probability to each backpointer.
        probabilities = [[options_to_get_before_char[backpointer]/options_to_get_before_char[node] for backpointer in backpointers[node]]
                         for node in range(len(pretoken)+1)]
        return backpointers, probabilities

    def constructMarkovGraphBackwards(self, pretoken: str) -> Tuple[List[List[int]],List[List[float]]]:
        """
        Produces two lists:
            - For each position in the string, the future positions in the string that can reach it with 1 token.
            - For each position in the string, the fraction of all possible paths (starting at the end) arriving there
              that arrive by using the corresponding token in the first list.
        """
        # Get raw counts and arcs in the graph.
        options_to_get_before_char = [0 for _ in range(len(pretoken)+1)]
        options_to_get_before_char[len(pretoken)] = 1
        backpointers = [[] for _ in range(len(pretoken)+1)]
        for i in range(len(pretoken), 0, -1):  # Starts at index n, ends at index 1.
            for j in range(i-1, -1, -1):  # Last index is 0.
                if (i-j >= self.min_len or not backpointers[j]) and self.hasType(pretoken[j:i]):  # To get from BEFORE character i to BEFORE character j, you do include j and don't include i.
                    options_to_get_before_char[j] += options_to_get_before_char[i]
                    backpointers[j].append(i)

        # Assign a normalised probability to each backpointer.
        probabilities = [[options_to_get_before_char[backpointer]/options_to_get_before_char[node] for backpointer in backpointers[node]]
                         for node in range(len(pretoken)+1)]
        return backpointers, probabilities

    def getJointProbability(self, tokens: List[str]) -> float:
        probability = 1

        pretoken = "".join(tokens)
        if self.decode_backwards:
            edges, weights = self.constructMarkovGraphForwards(pretoken)

            token_index = len(tokens)-1
            cur_node = len(pretoken)
            while cur_node > 0:
                next_node = cur_node-len(tokens[token_index])
                assert next_node in edges[cur_node]
                probability *= self.renormalisation.normalise(np.array(weights[cur_node]))[edges[cur_node].index(next_node)]
                cur_node = next_node
                token_index -= 1
        else:
            edges, weights = self.constructMarkovGraphBackwards(pretoken)

            token_index = 0
            cur_node = 0
            while cur_node < len(pretoken):
                next_node = cur_node+len(tokens[token_index])
                assert next_node in edges[cur_node]
                probability *= self.renormalisation.normalise(np.array(weights[cur_node]))[edges[cur_node].index(next_node)]
                cur_node = next_node
                token_index += 1

        return probability

    def getName(self):  # Properties in order of how much they alter the behaviour of the tokeniser.
        return "GRaMPa(" + \
            ("inf," if self._accept_all_types else "") + \
            (f"S(t={self.renormalisation.tau})" if isinstance(self.renormalisation, SoftmaxNormalisation)
            else f"P(t={self.renormalisation.tau})" if isinstance(self.renormalisation, PowerNormalisation)
            else "") + \
            f",l={self.min_len}" + \
            (",R2L" if self.decode_backwards else ",L2R") + \
        ")"
