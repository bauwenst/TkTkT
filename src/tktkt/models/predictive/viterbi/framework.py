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
from typing import Iterable, Union

from torch.fx.experimental.symbolic_shapes import lru_cache
from typing_extensions import Self
from abc import abstractmethod, ABC
import numpy as np

from ....interfaces import TokeniserWithVocabulary, Vocab
from ....interfaces.preprocessors import Pretokeniser
from ....interfaces.tokenisers import Tokeniser, Preprocessor
from ....util.printing import gridify
from ....util.iterables import transpose

INFTY = float("inf")
CACHE_SIZE = 1024


class ViterbiStepScores:
    """
    Stores, at index (n,k), the score gained from stepping k+1 characters starting BEFORE character n.
    """

    DTYPE = np.float32

    def __init__(self, N: int, K: int, default=0):
        self.grid = np.full(shape=(N,K), fill_value=default, dtype=ViterbiStepScores.DTYPE)

    def get(self, n: int, k: int) -> float:
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


class ViterbiStepScoreGenerator(ABC):

    def generateGrids(self, string: str, max_k: int, widths: list[int]) -> list[ViterbiStepScores]:
        """
        :param max_k: Defines the relative walls which each starting position in the grid must hit.
        :param widths: Defines the absolute walls that will be hit no matter the starting position.
        :return: len(widths) grids of sizes width1 x max_k, width2 x max_k, width3 x max_k, ...
        """
        pass


class ViterbiStepScoreGeneratorWithTokens(ViterbiStepScoreGenerator):

    @abstractmethod
    def generateGrids(self, string: str, max_k: int, widths: list[int]) -> list[ViterbiStepScoresWithTokens]:
        pass


class ViterbiAccumulator(ABC):

    @abstractmethod
    def combine(self, previous_value: float, edge_score: float):
        pass


########################################################################################################################


@dataclass
class ViterbiObjective:
    initial_score: float
    score_generator: ViterbiStepScoreGenerator
    score_combiner: ViterbiAccumulator


ViterbiObjectives = list[ViterbiObjective]

DEFAULT_VITERBI_POINTER = -1

@dataclass
class ViterbiIntermediates:
    best_objectives: list[tuple[float,...]]  # character -> objective value (and tiebreakers) of best path that has reached it so far.
    backpointers: list[int]                  # character -> index of where that path stepped from.

    def __init__(self, N: int, objectives: ViterbiObjectives):
        self.best_objectives = [tuple((o.initial_score if n == 0 else -INFTY) for o in objectives) for n in range(N)]
        self.backpointers    = [DEFAULT_VITERBI_POINTER for _ in range(N)]


@dataclass
class ViterbiPreprocessor:
    """
    Two preprocessors: one decides the context of the segmentation graph generators, the other decides forced boundaries
    and thus where the segmentation graph has bottlenecks that turn it into a sequence of smaller segmentation graphs.
    """
    stage1: Preprocessor  # Turns a raw text string into stage-1 pretokens that will be given to the grid generators, and the stage-2 pretokeniser.
    stage2: Pretokeniser  # Turns a stage-1 pretoken into (the boundaries of) stage-2 pretokens, which will each get their own Viterbi grid from the grid generators, whilst the grid generators still have access to the stage-1 pretokens.


class ViterbiTokeniser(Tokeniser):
    """
    Maximum-score segmenter using O(N²) Viterbi decoding, for any token scoring function that scores based only on their
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

    def __init__(self, preprocessor: ViterbiPreprocessor, max_stepsize: int,
                 objectives: ViterbiObjectives, degenerate: bool=False, trimmed: bool=True):
        """
        :param max_stepsize: Defines relative hard boundaries, i.e. from index i, index i+max_stepsize looks like a hard boundary even if it isn't.

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
                        space complexity stays O(L x N x K) because we don't truncate the grids, just how far we look in them.
        """
        Tokeniser.__init__(self, preprocessor=preprocessor.stage1)  # Instead of super() because this class is inherited in multiple inheritance and therefore super() would be a reference to the second inherited class (the sibling/spouse of this class), not to this class's parent.
        if not objectives:
            raise ValueError("At least one Viterbi objective is needed to construct a trellis.")
        if degenerate and not isinstance(objectives[0].score_generator,ViterbiStepScoreGeneratorWithTokens):
            raise ValueError("To support degenerate tokenisation, the first objective must generate a token grid too.")

        self.boundaries = preprocessor.stage2
        self.objectives = objectives
        self.K = max_stepsize
        self._degenerate_output = degenerate
        self._trim_fully_infinite_steps = trimmed

    @lru_cache(maxsize=None)
    def tokenise(self, pretoken: str):
        # 0. Get the length of the graphs that live within this pretoken. We call each span of pretoken to which a graph applies a "slice".
        slice_widths = [len(t) for t in self.boundaries.split(pretoken)]
        assert sum(slice_widths) == len(pretoken)

        S = len(slice_widths)
        K = min(self.K, max(slice_widths))  # Relaxation of the given K to a potentially better value, since there's no point having a bigger step than the entire grid's width. When there is only one grid, this reduces to the biggest step being from character 0 to character N, the end-of-string position.
        B = len(self.objectives)

        # 1. There is a different set of edge weights per objective, per string slice.
        all_graphs = [o.score_generator.generateGrids(pretoken, K, slice_widths) for o in self.objectives]  # len(objectives) x len(slice_widths) matrix of width x k score grids. So technically a ragged tensor.
        assert all(len(slices) == S for slices in all_graphs)

        # 2 and 3: Per slice, do a forward and backward pass. Because we decode backwards, we also do the slices in reverse order, so we can reverse the resulting list of tokens in one pass.
        tokens = []
        for slice_idx, width in enumerate(reversed(slice_widths)):
            # 2. Walk forwards through the slice's graphs.
            objective_graphs = [all_graphs[objective_idx][S-(slice_idx+1)] for objective_idx in range(B)]
            if self._trim_fully_infinite_steps:
                K = min(graph.getEffectiveK() for graph in objective_graphs)  # If one of the graphs has very little movement potential (e.g. you applied a vocabulary constraint to it), then the others are assumed to have it too.

            v = ViterbiIntermediates(width + 1, self.objectives)  # width+1 because there is a node after the whole string.
            for n in range(width):
                clipped_K = min(K, width-n)  # There are K jumps by default, but when you're at e.g. node n == width-1 (i.e. in front of the last character), there is only 1 jump to do.
                for k in range(clipped_K):
                    offered_objective_values = tuple([o.score_combiner.combine(v.best_objectives[n][i], objective_graphs[i].get(n, k))
                                                      for i,o in enumerate(self.objectives)])
                    existing_objective_values = v.best_objectives[n+(k+1)]
                    if offered_objective_values > existing_objective_values:  # Tuple comparison
                        v.best_objectives[n+(k+1)] = offered_objective_values
                        v.backpointers[n+(k+1)]    = n

            # 3. Walk backwards over the best path.
            prev_index    = width  # bigger index
            current_index = v.backpointers[prev_index]  # smaller index
            if not self._degenerate_output:
                indices_to_token = lambda string, left_index, right_index: string[left_index:right_index]
            else:
                indices_to_token = lambda string, left_index, right_index: objective_graphs[0].getToken(left_index, right_index-left_index-1)

            while current_index != DEFAULT_VITERBI_POINTER:
                token = indices_to_token(pretoken, current_index, prev_index)
                tokens.append(token)

                prev_index    = current_index
                current_index = v.backpointers[prev_index]

        return reversed(tokens)


class ViterbiTokeniserWithVocabulary(ViterbiTokeniser, TokeniserWithVocabulary):

    def __init__(self, preprocessor: ViterbiPreprocessor, vocab: Vocab,
                 max_stepsize: int, objectives: ViterbiObjectives, degenerate: bool=False, trimmed: bool=True):
        # In multiple inheritance with two classes that are already derived, using super() causes errors.
        ViterbiTokeniser.__init__(self, preprocessor=preprocessor, max_stepsize=max_stepsize, objectives=objectives, degenerate=degenerate, trimmed=trimmed)
        TokeniserWithVocabulary.__init__(self, preprocessor=preprocessor.stage1, vocab=vocab)


from ....interfaces.artifactories import TokeniserFactory
class AutoViterbiTokeniser(TokeniserFactory):
    """
    Depending on which objectives the tokeniser is instantiated with at runtime, it should have different methods.
    """

    def __init__(self, preprocessor: ViterbiPreprocessor, max_stepsize: int,
                 objectives: ViterbiObjectives, accelerate: bool=True):
        self.preprocessor = preprocessor
        self.objectives = objectives
        self.max_stepsize = max_stepsize
        self.accelerated = accelerate

    def buildTokeniser(self) -> Union[ViterbiTokeniser, ViterbiTokeniserWithVocabulary]:
        from .objectives_postprocessors import VocabularyConstraintExact, VocabularyConstraintAtLeastAll
        vocab = None
        is_degen = False
        for i, objective in enumerate(self.objectives):
            if isinstance(objective, (VocabularyConstraintExact, VocabularyConstraintAtLeastAll)):
                vocab = objective.vocab
            if i == 0 and isinstance(objective.score_generator, ViterbiStepScoreGeneratorWithTokens):
                is_degen = True

        if vocab is not None:
            return ViterbiTokeniserWithVocabulary(
                preprocessor=self.preprocessor,
                vocab=vocab,
                max_stepsize=self.max_stepsize,
                objectives=self.objectives,
                degenerate=is_degen,
                trimmed=self.accelerated
            )
        else:
            return ViterbiTokeniser(
                preprocessor=self.preprocessor,
                max_stepsize=self.max_stepsize,
                objectives=self.objectives,
                degenerate=is_degen,
                trimmed=self.accelerated
            )
