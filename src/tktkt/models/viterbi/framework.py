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
        - Masked transformer: send the string through an encoder N times. At time n, replace characters n:n+K by a
                              generic mask token, then generate embeddings. Each time, only store the embedding at
                              position n and throw out the rest, giving O(N) storage and O(N³) complexity.
        At the end of both of these, you have n embeddings. Send each through a linear+softmax layer of size h x |V|.
        Position (n,k) in the Viterbi grid is what the nth softmax produces for subword string[n:n+k]. The combination
        function is *. Alternatively, take the ln() of the grid and have the combination function be +.

Deciding whether a substring doesn't belong to the vocabulary, and hence a step can't be taken, should NOT be done in
the Viterbi decoder, but rather in the scoring grid. The grid might map the step to a different string, e.g.
"""
from dataclasses import dataclass
from typing import List, Tuple, Iterable
from typing_extensions import Self
from abc import abstractmethod
import numpy as np

from ...interfaces.tokeniser import Tokeniser, Preprocessor
from ...util.printing import gridify
from ...util.iterables import transpose

INFTY = float("inf")


class ViterbiStepScores:
    """
    Stores, at index (n,k), the score gained from stepping k+1 characters starting BEFORE character n.
    """

    DTYPE = np.float32

    def __init__(self, N: int, K: int, default=0):
        self.grid = np.full(shape=(N,K), fill_value=default, dtype=ViterbiStepScores.DTYPE)

    def get(self, n: int, k: int):
        return self.grid[n,k]

    def set(self, n: int, k: int, value: float):
        self.grid[n,k] = value

    def add(self, n: int, k: int, value: float):
        self.grid[n,k] += value

    def __repr__(self):
        with np.printoptions(linewidth=200):
            return self.grid.T.__repr__()  # Print transpose because we are used to seeing strings horizontally.

    def getEffectiveK(self) -> int:
        """
        Assuming that all Viterbi accumulators that accumulate with +/-INFTY also return +/-INFTY and that this is the
        least desirable score, that means that if after a certain k the grid consists of +/-INFTY values entirely for
        all n, paths with steps bigger than that k don't even need to be considered.
        """
        # ks_with_full_inf = np.nonzero(np.all(np.isinf(self.grid), axis=0))[0]  # This is actually not what you need because it is possible that k1 has all-inf steps while k2 > k1 does not!
        ks_with_any_noninf = np.nonzero(np.any(np.isfinite(self.grid), axis=0))[0]  # Last one of these is the last index you need to support.
        try:
            return int(ks_with_any_noninf[-1]) + 1
        except:
            return self.grid.shape[1]

    @classmethod
    def fromExisting(cls, grid: Iterable[Iterable[float]], needs_transpose: bool=False) -> Self:
        """
        :param needs_transpose: If True, the given score grid is transposed before use, which you need to do when it is laid
                                out in the way we visualise it (steps x characters), i.e. with the vertical dimension representing step sizes.
        """
        this = cls(0,0)
        this.grid = np.array(grid, dtype=ViterbiStepScores.DTYPE)  # This also checks non-raggedness.
        assert len(this.grid.shape) == 2, f"Score grid must be 2D, not {len(this.grid.shape)}D."

        if needs_transpose:
            this.grid = this.grid.T  # Note: in PyTorch, a Tensor being a view can raise errors downstream. NumPy has no .contiguous() method though, so I assume this is okay.
        return this


class ViterbiStepScoresWithTokens(ViterbiStepScores):
    """
    Optional subclass that allows storing a token alongside a step. This way, you can replace the Viterbi path by custom
    strings that cannot be concatenated to invert the tokenisation into the original string.

    One obvious example of such degenerate tokenisation is the CELEX dataset itself.
    """

    def __init__(self, N: int, K: int, default=0):
        super().__init__(N, K, default)
        self.tokens = [["---" for _ in range(K)] for _ in range(N)]  # If you see the default anywhere, something is really wrong.

    def getToken(self, n: int, k: int) -> str:
        return self.tokens[n][k]

    def setToken(self, n: int, k: int, step: str):
        self.tokens[n][k] = step

    def __repr__(self):
        return gridify(transpose(self.tokens)) + "\n" + super().__repr__()

    @classmethod
    def fromExisting(cls, grid: Iterable[Iterable[float]], tokens: Iterable[Iterable[str]], needs_transpose: bool=False) -> Self:
        this = super().fromExisting(grid=grid, needs_transpose=needs_transpose)  # Returns object of this subclass, not of the super type. Python magic.

        this.tokens = [list(tokens_from_character) for tokens_from_character in tokens]
        if needs_transpose:
            this.tokens = transpose(this.tokens)

        # Check that there are as many tokens as there are scores. Since the scores have been checked on non-raggedness,
        # this implicitly also checks non-raggedness.
        N,K = this.grid.shape
        assert len(this.tokens) == N
        for tokens_from_character in this.tokens:
            assert len(tokens_from_character) == K

        return this


class ViterbiStepScoreGenerator:

    @abstractmethod
    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScores:
        pass


class ViterbiStepScoreGeneratorWithTokens(ViterbiStepScoreGenerator):

    @abstractmethod
    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScoresWithTokens:
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

    def __init__(self, preprocessor: Preprocessor, max_stepsize: int,
                 objectives: ViterbiObjectives, degenerate: bool=False, trimmed: bool=True):
        """
        :param degenerate: Whether the first objective has a generator that produces its own tokens AND you want
                           to use those when tokenising.
                           When true, you hence likely cannot concatenate the produced tokens to reconstruct the input.
                           When false, you can still use a degenerate objective, but its suggestions will not be used.
                           That might mean you cannot convert the produced tokens to IDs, but you can concatenate them.
        :param trimmed: Whether to limit the step size to the largest step size for which all objectives have at least one
                        non-infinite score starting from some character. In other words: if even just one objective has an infinite
                        score for a certain step size at all characters, then all objectives can only make steps smaller than that step.
                        This means you can e.g. put a vocabulary constraint on one objective and it applies to all automatically.
                        Time complexity becomes O(L x N x K') for L objectives and for some K' <= K,
                        space complexity stays O(L x N x K).
        """
        super().__init__(preprocessor)
        if not objectives:
            raise ValueError("At least one Viterbi objective is needed to construct a trellis.")
        if degenerate and not isinstance(objectives[0].score_generator,ViterbiStepScoreGeneratorWithTokens):
            raise ValueError("To support degenerate tokenisation, the first objective must generate a token grid too.")

        self.objectives = objectives
        self.K = max_stepsize
        self._degenerate_output = degenerate
        self._trim_fully_infinite_steps = trimmed

    def tokenise(self, string: str):
        N = len(string)
        K = min(self.K, N)  # There's no point having a bigger step than the entire string's length. Biggest step is from character 0 to character N, the end-of-string position.

        # 1. There is a different set of edge weights per objective and per string. Generate these for the given string.
        graphs = [o.score_generator.generateGrid(string, K) for o in self.objectives]
        if self._trim_fully_infinite_steps:
            K = min(graph.getEffectiveK() for graph in graphs)  # We assume that this cannot be larger than the K we already had. You're in trouble otherwise.

        # 2. Walk forwards through the graphs.
        t = ViterbiTrellis(N+1, self.objectives)  # N+1 because there is a node (node index N) after the whole string.
        for n in range(N):
            clipped_K = min(K, N-n)  # There are K jumps by default, but when you're at e.g. node n == N-1 (i.e. in front of the last character), there is only 1 jump to do.
            for k in range(clipped_K):
                offered_objective_values = tuple([o.score_combiner.combine(t.best_objectives[n][i], graphs[i].get(n, k))
                                                  for i,o in enumerate(self.objectives)])
                existing_objective_values = t.best_objectives[n+(k+1)]
                if offered_objective_values > existing_objective_values:
                    t.best_objectives[n+(k+1)] = offered_objective_values
                    t.backpointers[n+(k+1)]    = n

        # 3. Walk backwards over the best path.
        tokens = []

        prev_index    = N  # bigger index
        current_index = t.backpointers[prev_index]  # smaller index
        if not self._degenerate_output:
            indices_to_token = lambda string, left_index, right_index: string[left_index:right_index]
        else:
            indices_to_token = lambda string, left_index, right_index: graphs[0].getToken(left_index, right_index-left_index-1)

        while current_index != -1:
            token = indices_to_token(string, current_index, prev_index)
            tokens.insert(0, token)

            prev_index    = current_index
            current_index = t.backpointers[prev_index]

        return tokens
