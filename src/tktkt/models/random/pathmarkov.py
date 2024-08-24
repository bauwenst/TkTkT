"""
TODO: Because there is a slight edge effect where the end of a decode may be forced to add e.g. a single-character token
      that would never be used if decoding started on that side of the string, the algorithm is not entirely equivalent
      left-to-right versus right-to-left once you apply a transformation that makes it no longer uniform across segmentations.
      You should hence have a right-to-left trellis and a left-to-right decode.
"""
from typing import List, Tuple
import numpy as np
import numpy.random as npr

from .randomfromvocab import TokeniserWithVocabDict, Preprocessor, Vocab
from ...util.arrays import BatchNormalisation, IdentityBatchNormalisation
from ...util.strings import segmentUsingIndices


class RandomVocabSegmentation_GreedyMarkov(TokeniserWithVocabDict):
    """
    First computes the graph of possible token paths through the given string, then samples a random path
    by starting at the end of the string, choosing one random incoming token with probability proportional to how
    many paths it causes to arrive at the end, following that token back and repeating this process.
    """

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, unk_type: str=None,
                 probabilities_to_probabilities: BatchNormalisation=IdentityBatchNormalisation(), minimal_token_length: int=1):
        """
        :param probabilities_to_probabilities: Transformation to apply to the Markov graph's probabilities, which by default
                                               are the fraction of paths entering through each incoming arc (causing the
                                               tokeniser to sample uniformly across all segmentations, of which there are
                                               more with smaller tokens than with larger tokens).
        :param minimal_token_length: Shortest token that can be produced *unless* no token of at least this length
                                     arrives at the current character during decoding, in which case we use the largest token that does.
        """
        super().__init__(preprocessor, vocab, unk_type)
        self.rng = npr.default_rng(0)
        self.renormalisation = probabilities_to_probabilities
        self.min_len = minimal_token_length

    def tokenise(self, pretoken: str) -> List[str]:
        edges, weights = self.constructMarkovGraph(pretoken)

        indices = []
        current_index = len(edges)-1
        while current_index:
            current_index = self.rng.choice(edges[current_index], p=self.renormalisation.normalise(np.array(weights[current_index])))  # "Greedy": we commit to each step without care for the future. As long as there is a path to the start, its entry in p is nonzero.
            indices.append(current_index)

        return segmentUsingIndices(pretoken, starts_of_tokens=indices[::-1])

    def constructMarkovGraph(self, pretoken: str) -> Tuple[List[List[int]],List[List[float]]]:  # Separate method so that it can be cached.
        """
        Produces two lists:
            - For each position in the string, the positions in the string that can reach it with 1 token.
            - For each position in the string, the fraction of all possible paths arriving there that arrive
              by using the corresponding token in the first list.
        """
        # Get raw counts and arcs in the graph.
        options_to_get_before_char = [0 for _ in range(len(pretoken)+1)]
        options_to_get_before_char[0] = 1
        backpointers = [[] for _ in range(len(pretoken)+1)]
        for i in range(len(pretoken)):
            for j in range(i+1, len(pretoken)+1):
                if (j-i >= self.min_len or not backpointers[j]) and pretoken[i:j] in self.vocab:  # not j+1 because we step to BEFORE character j, so it is an exclusive bound
                    options_to_get_before_char[j] += options_to_get_before_char[i]
                    backpointers[j].append(i)

        # Assign a normalised probability to each backpointer.
        probabilities = [[options_to_get_before_char[backpointer]/options_to_get_before_char[node] for backpointer in backpointers[node]]
                         for node in range(len(pretoken)+1)]
        return backpointers, probabilities
