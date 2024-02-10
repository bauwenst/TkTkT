"""
All Viterbi algorithms for tokenisation, whether they are guided or not, can run through the exact same implementation
if you abstract enough. My contention is the following:
    - All you need for the algorithm is to be able to query the score for each step in the given string of length N, and
      steps have a maximum size K.
    - To identify a step score, all you need is the starting point n in [0,N[ and the step size in [1,K].
      So, you only have to supply an N x K grid to the algorithm, along with a combination function that can combine
      a previous score with entry (n,k) in the grid.
    - For tiebreaker objectives, just repeat the above and keep the results of all objectives in tuples for waterfall comparison.

Here's how you would implement all the Viterbi variants I have thought of in this one single framework:
    - Least-amount-of-tokens: the entire grid is just a -1. The combination function is +.
    - Biggest-token: (n,k) is just k. The combination function is max().
    - Most-suggestions-hit: (n,k) is 1 iff label n+k is 1. The combination function is +.
        - Soft: if the model that suggests splits is probabilistic, use the probabilities, not 1.
    - Neural probability generators:
        - Autoregressive transformer: run the string through a decoder character-by-character to produce an embedding
                                      for each character.
        - Masked transformer: give the string through an encoder N times. For repeat n, replace characters n:n+k by a
                              generic mask token, then generate embeddings. Each time, only store the embedding at
                              position n and throw out the rest, giving O(n) storage and O(n³) complexity.
        At the end of both of these, you have n embeddings. Send each through a linear+softmax layer. Position (n,k) in
        the Viterbi grid is what the nth softmax produces for subword string[n:n+k]. The combination function is *.
        Alternatively, take the ln() of the grid and have the combination function be +.

TODO: You can probably extend this framework to non-invertible segmentations by also storing a string for every step.
      Rather than backtracing by taking substrings, you backtrace and map the step's (n,k) to a string.

Deciding whether a substring doesn't belong to the vocabulary, and hence a step can't be taken, should NOT be done in
the Viterbi decoder, but rather in the scoring grid. The grid might map the step to a different string, e.g.
"""
from dataclasses import dataclass
from typing import List, Tuple
from abc import abstractmethod
import numpy as np

from ...interfaces.general import Tokeniser, Pretokeniser

INFTY = float("inf")


class ViterbiStepScores:

    def __init__(self, N: int, K: int, default=0):
        self.grid = np.full(shape=(N,K), fill_value=default)

    def get(self, n: int, k: int):
        return self.grid[n,k]

    def set(self, n: int, k: int, value: float):
        self.grid[n,k] = value

    def __repr__(self):
        return self.grid.T.__repr__()


class ViterbiStepScoreGenerator:

    @abstractmethod
    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScores:
        pass


class ViterbiAccumulator:

    @abstractmethod
    def combine(self, previous_value: float, edge_score: float):
        pass


@dataclass
class ViterbiObjective:
    initial_score: float
    score_generator: ViterbiStepScoreGenerator
    score_combiner: ViterbiAccumulator


ViterbiObjectives = List[ViterbiObjective]


@dataclass
class ViterbiTrellis:
    best_objectives: List[Tuple[float,...]]  # character -> objective value (and tiebreakers) of best path that has reached it so far.
    backpointers: List[int]                  # character -> index of where that path stepped from.

    def __init__(self, N: int, objectives: ViterbiObjectives):
        self.best_objectives = [tuple((o.initial_score if n == 0 else -INFTY) for o in objectives) for n in range(N)]
        self.backpointers    = [-1 for _ in range(N)]


class ViterbiTokeniser(Tokeniser):
    """
    Maximum-score segmenter using O(N²) Viterbi decoding, for any function that scores substrings based only on their
    characters, characters around them, and their position in the string, but NOT the current segmentation.

    The scores can be laid out as the edges of a graph. In a string ABC, the following edges would get a score:

      /----------\
     /-------\    \
    /----\    \    \
    A    B    C----END
         \----/    /
          \-------/

    An edge from x to y corresponds to the substring containing x and everything between x and y, but not y. See the
    DPE paper (He et al., 2020) for a similar graph.

    Multiple scoring functions can be given, with additional scoring functions being fallbacks for when a comparison
    between all earlier scores is inconclusive. For example, you could have a least-tokens-used objective, and to then
    discern between the cumulative scores of ("abcd","e","f") and ("ab", "cd", "ef"), both being 3, you could use a
    fallback score of longest-token-used, here 4 vs. 2.
    """

    def __init__(self, pretokeniser: Pretokeniser,
                 objectives: ViterbiObjectives, max_stepsize: int):
        super().__init__(pretokeniser)
        self.objectives = objectives
        self.K = max_stepsize

    def tokenise(self, string: str):
        N = len(string)
        K = min(self.K, N)  # There's no point having a bigger step than the entire string's length. Biggest step is from character 0 to character N, the end-of-string position.

        # 1. There is a different set of edge weights per objective and per string. Generate these for the given string.
        graphs = [o.score_generator.generateGrid(string, K) for o in self.objectives]

        # 2. Walk forwards through the graphs.
        t = ViterbiTrellis(N+1, self.objectives)  # N+1 because there is a node (node index N) after the whole string.
        for n in range(N):
            clipped_K = min(K, N-n)  # There are K jumps by default, but when you're at e.g. node n == N-1, there is only 1 jump to do.
            for k in range(clipped_K):
                offered_objective_values = tuple([o.score_combiner.combine(t.best_objectives[n][i], graphs[i].get(n, k))
                                                  for i,o in enumerate(self.objectives)])
                existing_objective_values = t.best_objectives[n+(k+1)]
                if offered_objective_values > existing_objective_values:
                    t.best_objectives[n+(k+1)] = offered_objective_values
                    t.backpointers[n+(k+1)]    = n

        # 3. Walk backwards over the best path.
        tokens = []

        prev_index    = N
        current_index = t.backpointers[prev_index]
        while current_index != -1:
            tokens.insert(0, string[current_index:prev_index])

            prev_index    = current_index
            current_index = t.backpointers[prev_index]

        return tokens
