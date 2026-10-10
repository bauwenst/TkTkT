"""
Guided score functions. These are informed by knowledge of the language, e.g. models that estimate the probability of a
token appearing at all (with or without considering context), or models that estimate the probability of a split point.
"""
from typing import MutableSequence, Sequence
from abc import abstractmethod, ABC

import numpy as np

try:
    import torch
except ImportError:
    torch = object
from math import exp, sin, pi, asin
from math import log as ln

from .framework import ViterbiStepScoreGenerator, ViterbiStepScores, INFTY
from ....util.printing import sgnprint
from ....util.iterables import cumsum

__all__ = ["BoundaryScoresChosen", "BoundaryScoresAll",
           "BoundaryPrefixLength", "BoundarySuffixLength", "BoundaryPrefixLengthExtended", "BoundarySuffixLengthExtended",
           "BoundaryPrefixAndSuffixLengthExtended",
           "IdentityPT", "PiecewisePT", "PowerMBPT", "LogPT", "LinearPT", "SineMBPT", "DoublingMBPT", "ExponentialMBPT"]


class CharacterClassifier(ABC):

    @abstractmethod
    def getPointLogProbabilities(self, pretoken: str) -> MutableSequence[float]:  # "MutableSequence" is just "anything that can be indexed with [], has order, and can be modified"
        """
        Returns a list of binary classification log probabilities that say, for each of the n characters, whether it
        should be followed by a split point.

        Note that these are log probabilities, not just probabilities. The reason is two-fold:
            - Neural models produce logits, which need only a constant subtracted to be log probabilities, rather than an expensive softmax.
            - It's more accurate to convert logarithms to probabilities than the inverse due to rounding errors.
              Also, for further computations, it's much more numerically stable to do (-5) + (-5) + (-5) than (10^(-5)) * (10^(-5)) * (10^(-5)).
        """
        pass


class SubstringClassifier(ABC):

    @abstractmethod
    def getSegmentLogProbabilities(self, pretoken: str, max_k: int) -> MutableSequence[MutableSequence[float]]:
        """
        Returns a matrix of log probabilities that say, for each substring of length k starting at each of the n characters,
        what its probability of occurring is across some domain of substrings (e.g. a subword vocabulary).

        Note: When these probabilities are normalised by an external constraint (like in ULM: across
        the entire vocabulary, type probabilities sum to 1) rather than an internal constraint (like in GRaMPa: across
        the edges exiting from a node, probabilities sum to 1), optimising paths will lead to a length bias.
        """
        pass


#############################################################################################


class ProbabilityTransform(ABC):

    def __init__(self, negate_as_complement: bool=False):
        self.negate_as_complement = negate_as_complement

    @abstractmethod
    def probabilityToScore(self, p: float) -> float:
        pass

    @abstractmethod
    def scoreToProbability(self, s: float) -> float:
        pass

    def scoreToScoreOfComplement(self, score_p: float) -> float:
        """
        Map the score S(p) to the score S(1-p).
        """
        return self.probabilityToScore(1 - self.scoreToProbability(score_p))

    def complementOfScore(self, s: float) -> float:
        return -s

    def complement(self, s: float):
        if self.negate_as_complement:
            return self.complementOfScore(s)
        else:
            return self.scoreToScoreOfComplement(s)

    def __repr__(self):
        return self.__class__.__name__


class IdentityPT(ProbabilityTransform):

    def probabilityToScore(self, p: float) -> float:
        return p

    def scoreToProbability(self, s: float) -> float:
        return s


class LogPT(ProbabilityTransform):
    """
    You may wonder if it makes sense from the POV of numerical stability to let a classifier generate log probabilities,
    transform them to probability with exp(), and then convert back to the original space with log(). Isn't it safer to
    use the original predictions, then?

    Well, if you test numpy's answer to np.log(np.exp(L)), you'll find that it only breaks down around L == -729, which
    represents a probability of 10^{-317}. At that point, some rounding error doesn't matter.

    The reason for using log probabilities could be to pile many predictions on top of each other, at which point the
    conversion would start to matter: you won't find a prediction of 10^{-300}, but you might find 20 predictions of 10^{-15}.
    The transform is only applied to individual predictions, and it is transformed values that are combined, so the transform
    never interacts with those accumulated values either.
    """
    def probabilityToScore(self, p: float) -> float:
        return np.log(p)

    def scoreToProbability(self, s: float) -> float:  # This leads to a complement that looks like ln(1-p) == ln(1 - e^(ln p)) != -ln p
        return np.exp(s)


class LinearPT(ProbabilityTransform):
    """
    In the log domain, you have disproportional punishment for wrong decisions.

    If you use a linear transform instead with symmetric reward -1 and +1, making the wrong decision can be balanced out
    by making the right decision in a comparable situation: e.g., take two points have 70% and 30% probability of being
    a boundary. You say both are boundaries, then the reward is ( 2*0.7-1 ) + ( 2*(0.3)-1 ) = 0.4 + (-0.4) = 0.
    """

    def __init__(self, a: int, b: int, negate_as_complement: bool=False):
        super().__init__(negate_as_complement)
        self.a = a
        self.b = b

    def probabilityToScore(self, p: float):
        return self.a + p*(self.b-self.a)

    def scoreToProbability(self, s: float) -> float:
        return (s - self.a)/(self.b - self.a)

    def __repr__(self):
        return self.__class__.__name__ + f"({sgnprint(self.a)},{sgnprint(self.b)})" + ("_NegComp")*self.negate_as_complement


class PiecewisePT(ProbabilityTransform):

    def __init__(self, a: int, b: int, negate_as_complement: bool=False):
        super().__init__(negate_as_complement)
        self.a = a
        self.b = b

    def probabilityToScore(self, p: float):
        return 2*(p-0.5)*self.b if p > 0.5 else 2*(0.5 - p)*self.a

    def scoreToProbability(self, s: float) -> float:
        return 0.5 + s/(2*self.b) if s > 0 else 0.5 - s/(2*self.a)

    def __repr__(self):
        return self.__class__.__name__ + f"({sgnprint(self.a)},{sgnprint(self.b)})" + ("_NegComp")*self.negate_as_complement


class SigmoidPT(ProbabilityTransform):
    """
    Implements a symmetric sigmoid around 0.5 with maximum of 1.
    Any steepness under 5 is basically just a straight line. Above 15 is very steep.
    """

    def __init__(self, steepness: float):
        super().__init__(negate_as_complement=False)  # Doesn't matter because f(1-p) == -f(p) anyway.
        self.steepness = steepness
        self.normalisation = 2 * (1 + exp(-0.5*steepness)) / (1 - exp(-0.5*steepness))

        self.inv_steepness     = 1/self.steepness
        self.inv_normalisation = 1/self.normalisation

    def probabilityToScore(self, p: float) -> float:
        return self.normalisation * (1/(1 + exp(-self.steepness*(p-0.5))) - 0.5)

    def scoreToProbability(self, s: float) -> float:
        return 0.5 - self.inv_steepness * ln(1/(s*self.inv_normalisation + 0.5) - 1)


class MultiplicativelyBalancedProbabilityTransform(ProbabilityTransform):
    """
    Defines a subset of the probability transforms f(p) that satisfy
        1. f(p) is monotonously increasing.
        2. f(p) * f(1-p) = 1
        3. lim_{p->0.5+} f(p) = lim_{p->0.5-} f(p) = 1, that is, there is no discontinuity at p = 0.5.

    ---

    An example of a function that trivially satisfies the first two conditions yet doesn't satisfy the last is
        f(p) = 1/(1-p) for p > 0.5 and f(p) = p otherwise.
    (1) The separate pieces are monotonous, and for any p1 < 0.5 < p2, note that p2 < 0.5 => 0.5 < 1-p2 => 1/0.5 < 1/(1-p2) and thus f(p1) = p1 < 0.5 < 2 < 1/(1-p2) = f(p2).
    (2a) For any p' that is < 0.5, this gives f(p') * f(1-p') = p' * 1/(1 - (1-p')) = p' * 1/p' = 1.
    (2b) For any p' that is > 0.5, this gives f(p') * f(1-p') = 1/(1-p') * (1-p') = 1.
    (3) Yet lim_{p->0.5+} f(p) = 1/(1-0.5) = 1/0.5 = 2 whereas lim_{p->0.5-} f(p) = 0.5.

    ---

    This class achieves all three conditions by taking any auxiliary function g(x) for which
        1. g(0) = 1
        2. g(p) is monotonously increasing for p in [0,0.5]
    and then taking the piecewise function
        x > 0.5: f(x) = g(x - 0.5)
        x < 0.5: f(x) = 1/g(0.5 - p)
    """

    def probabilityToScore(self, p: float) -> float:
        return 1/self.isOneAtZero(0.5 - p) if p < 0.5 else self.isOneAtZero(p - 0.5)

    def scoreToProbability(self, s: float) -> float:
        return 0.5 - self.isOneAtZero_inv(1/s) if s < 1 else 0.5 + self.isOneAtZero_inv(s)

    @abstractmethod
    def isOneAtZero(self, x: float) -> float:
        pass

    @abstractmethod
    def isOneAtZero_inv(self, y: float) -> float:
        pass


class PowerMBPT(MultiplicativelyBalancedProbabilityTransform):
    """
    Models g(x) = 1 + c*x^p.
    Also, the case p = 1 has almost the same resulting f(x) as g(x) = 1 + c*ln(1+x) and g(x) = 1 + c*tan(x), hence we
    don't have a subclass for those.
    """

    def __init__(self, power: float, scale: float, negate_as_complement: bool=False):
        super().__init__(negate_as_complement=negate_as_complement)
        assert power != 0 and scale != 0
        self.scale = scale
        self.power = power

    def isOneAtZero(self, x: float) -> float:
        return 1 + self.scale*pow(x, self.power)

    def isOneAtZero_inv(self, y: float) -> float:
        return pow((y-1)/self.scale, 1/self.power)

    def __repr__(self):
        return self.__class__.__name__ + f"(c={self.scale},p={sgnprint(self.power)})"


class DoublingMBPT(PowerMBPT):
    """
    Models g(x) = 1 + 2*x, which is one specific case of PowerMBPT that is easy to compute.
    It turns P = 100% into a score factor ×2 and turns P = 0% into a score factor ×1/2.
    """

    def __init__(self):
        super().__init__(power=1, scale=2)

    def isOneAtZero(self, x: float) -> float:
        return 1 + 2*x

    def isOneAtZero_inv(self, y: float) -> float:
        return (y-1)/2

    def __repr__(self):
        return f"PowerMBPT(c={self.scale},p={sgnprint(self.power)})"


class ExponentialMBPT(MultiplicativelyBalancedProbabilityTransform):
    """
    Models g(x) = exp(c*x).
    """

    def __init__(self, scale: float, negate_as_complement: bool=False):
        super().__init__(negate_as_complement=negate_as_complement)
        assert scale != 0
        self.scale = scale

    def isOneAtZero(self, x: float) -> float:
        return exp(self.scale*x)

    def isOneAtZero_inv(self, y: float) -> float:
        return ln(y)/self.scale

    def __repr__(self):
        return self.__class__.__name__ + f"(c={self.scale})"


class SineMBPT(MultiplicativelyBalancedProbabilityTransform):
    """
    Models g(x) = 1 + c/4 * (1 + sin(2*pi*(x - 1/4))) which has the same extrema as PowerMBPT of the same c and p = 1,
    but a different flow.
    """

    def __init__(self, scale: float, negate_as_complement: bool=False):
        super().__init__(negate_as_complement=negate_as_complement)
        assert scale != 0
        self.scale = scale

    def isOneAtZero(self, x: float) -> float:
        return 1 + 0.25*self.scale*(1 + sin(2*pi*(x - 0.25)))

    def isOneAtZero_inv(self, y: float) -> float:
        return 1/(2*pi) * asin((y-1)*4/self.scale - 1) + 0.25

    def __repr__(self):
        return self.__class__.__name__ + f"(c={self.scale})"


########################################################################################################################


class _ScoreGeneratorUsingCharacterClassifier(ViterbiStepScoreGenerator):
    """
    Stores a model that generates log(probability) values.
    The reason why that model is not a constructor argument is because the constructor of these score generators should
    be used to declare how the generator works ON TOP OF the model REGARDLESS OF which model it is. The backend model
    can vary across runtimes, whilst the parameters of the generator are fixed. "I want this tokeniser to use a (-2,+1)
    linear probability transform" is much more static than "I want this tokeniser to get its probabilities from this checkpoint".

    You will receive a NoneTypeException at some point if you forgot to set the backend.

    This class has two inheritors: one transforms the probabilities of the model and puts the result directly in the
    score grid, the other thresholds the probabilities and uses the location of these hard boundaries to generate its own scores.
    """
    def __init__(self):
        self.logprob_classifier: CharacterClassifier = None

    def setBackend(self, point_model: CharacterClassifier):
        self.logprob_classifier = point_model


class _ScoreGeneratorUsingCharacterClassifierForTransform(_ScoreGeneratorUsingCharacterClassifier):
    """
    Stores a transformation for probabilities (not log probabilities!), since it automatically converts the given model's
    log probabilities into probabilities before applying the transformation.
    """

    def __init__(self, transform: ProbabilityTransform):
        super().__init__()
        self.T = transform

    def getBoundaryScores(self, string: str) -> list[float]:  # Produces len(string) values, one for each post-character position.
        return list(map(self.T.probabilityToScore, map(exp, self.logprob_classifier.getPointLogProbabilities(string))))  # one entry for each character

    def generateGrids(self, string: str, max_k: int, widths: list[int]) -> list[ViterbiStepScores]:
        token_ends = tuple(cumsum(widths))

        all_scores  = self.getBoundaryScores(string)
        dummy_score = self.T.probabilityToScore(0.5)
        for end in token_ends:
            all_scores[end-1] = dummy_score  # Since every segmentation visits the post-last-character position, this score does not matter. It just cannot be an absorbing element (-infty for addition, 0 for multiplication). We use the score transform because potentially the user chose it to produce the neutral element of the combining operation at 50% probability (not necessarily the case, but no score is going to score 0.5 with an absorbing element).

        return [self._generateGrid(all_scores[start:end], max_k) for start,end in zip((0,) + token_ends, token_ends)]

    @abstractmethod
    def _generateGrid(self, scores: Sequence[float], max_k: int) -> ViterbiStepScores:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "(" + self.T.__repr__() + ")"


class BoundaryScoresChosen(_ScoreGeneratorUsingCharacterClassifierForTransform):
    """
    For Viterbi paths that maximise the score on the character boundaries they hit.

    If using log probabilities, hitting low probabilities is disproportionately punished. This is not the case for
    symmetric linear transforms, like [0,1] -> [-1,+1]. Note that even though such scores might resemble probabilities
    a lot, they should STILL be summed, not multiplied (as if they're still logarithms) when accumulated.

    Now, point models give a probability of each character having a split point after it or not.
    So why not use probabilities?

    We want a Viterbi path to accumulate as much of the probability mass as possible (hit all the predicted
    boundaries), but disincentivise cheating by oversegmenting. Let's say you sum across the boundaries you hit.
    Oversegmenting would then cause you to collect all boundary probability, without being punished for collecting 0s.
    Hence, it is better to use a linear transform like -1 + P*(1 - (-1)) = 2P-1: you are rewarded for stepping on a
    boundary and you are punished for stepping on a non-boundary.
    The alternative is to use multiplication, or equivalently to use logarithms, but beware that this gives outsized
    punishment to stepping on a non-boundary (a single non-boundary can make 20 boundary hits evaporate by multiplying
    by a very small number). This is also easy to see when you take the ln, where you are punished by -100 if you
    get it wrong but getting it right is an invisible +0.

    Furthermore, multiplying probabilities (summing logarithms) has some bias towards shorter token sequences.
    Picking up [START, 1, 1, 1, 0.99, END] is worse than picking up [START, 1, 1, 1, END] when you multiply, even though
    clearly the first is much better. In log space, you get -0 -0 -0 -0.000001 versus -0 -0 -0. This is why a probability
    transform and then summing instead of multiplying is worthwhile.
    """

    def _generateGrid(self, scores: Sequence[float], max_k: int) -> ViterbiStepScores:
        N = len(scores)
        grid = ViterbiStepScores(N, max_k, default=0)  # Default doesn't matter because the entire grid is filled.
        for n in range(N):
            K = min(max_k, N-n)  # Example: you are at n == 5 and there are N == 7 characters. This means you still need to include character 5 and 6 in the segmentation, using the probabilities at n+0 and n+1. Hence, there are K == N-n steps available.
            for k in range(K):
                grid.set(n, k, scores[n + k])  # Example: you are at n == 0. For k == 1, you take the second-smallest step, to 2. The probability of hitting a boundary by doing so is the probability of a boundary being after 1 == n+k.

        return grid


class BoundaryScoresAll(_ScoreGeneratorUsingCharacterClassifierForTransform):
    """
    Uses point boundaries, assumed to be independent, to compute segment probabilities.
    Should still be used with sum accumulation.

    Let's say you have suggested splits abc|de|f. The probability of the segment "abc" is computed now not as
        P(boundary after char 2 | abcdef)
    but instead as
        P(boundary after char 2 | abcdef)*P(no boundary after char 1 | abcdef)*P(no boundary after char 0 | abcdef)

    This is exactly what is calculated when you use log scores (because the sum of the logs is the log of the product of
    the probabilities, and you get numerical stability as a bonus), so mathematically speaking, this should be the most
    accurate type of score, since maximising this objective maximises the joint distribution across all possible split points.
    Of course, when constraints are involved, this is no longer true.

    There's a funky equivalence here: you could convert the CharacterClassifier into a SubstringClassifier first, and
    then use that SubstringClassifier with its usual grid generator.

    Note: for certain score functions, it is mathematically provable that they are equivalent in BoundaryScoresAll and
    in BoundaryScoresChosen, and in others it is mathematically provable that they are not equivalent but do lose all
    of their expressivity in BoundaryScoresAll rather than BoundaryScoresChosen. So choose wisely whether you want this.
    """

    def _generateGrid(self, scores: Sequence[float], max_k: int) -> ViterbiStepScores:
        N = len(scores)
        grid = ViterbiStepScores(N, max_k, default=0)
        for n in range(N):
            K = min(max_k, N-n)  # Like above.
            for k in range(K):
                cumulative_score = scores[n+k]  # Like above.
                for i in range(k):
                    cumulative_score += self.T.complement(scores[n+i])
                grid.set(n, k, cumulative_score)

        return grid


########################################################################################################################


class _ScoreGeneratorUsingCharacterClassifierForHardBoundaries(_ScoreGeneratorUsingCharacterClassifier):
    """
    Base class for all prefix/suffix score generators, which don't put predicted probabilities into the grid but rather
    some score related to the hard boundaries they indicate.
    """

    def __init__(self, punishment: float=-1, do_normalise: bool=False):
        """
        :param punishment: Default score of a step when it isn't involved with any boundary.
        :param do_normalise: Make the scores in the generated grids not exceed 1.0. For example, in prefix/suffix objectives,
                             the scores are as high as the length of the segments suggested by the hard boundaries. This means
                             more reward is given when you prioritise long tokens. Yet, arguably, you want to respect boundaries
                             for small morphemes like affices, so by normalising, you get more reward for getting 2 short affices right
                             than getting 1 long stem right.
        """
        super().__init__()
        self.punishment = -abs(punishment)  # User can give positive or negative values, doesn't matter.
        self.do_normalise = do_normalise

    # def getHardBoundaryIndices(self, string: str) -> list[int]:
    #     """
    #     Returns the INDICES of which characters are PRECEDED by a boundary with probability > 0.5.
    #     Always includes index 0 and len(string), which are imaginary boundaries.
    #     """
    #     # To reiterate how indexing works in the Viterbi framework:
    #     #   - The step score at [n,k] is the score you get when you are at the split position BEFORE character n (with n being 0-based) and take a step of k+1 characters.
    #     #   - The boundary probability at n is the probability of there being a split AFTER character n.
    #     # In other words: steps scores are indexed by inter-character positions (including two at the ends), boundary probabilities are indexed by character.
    #
    #     # If the proposed segmentation is "w|or|d", thresholding the probabilities gives a mask [1, 0, 1, 0].
    #     # Index i then says whether there is a boundary AFTER character i.
    #     boundary_after_asmask = [1 * (exp(ln) > 0.5) for ln in self.logprob_classifier.getPointLogProbabilities(string)]
    #
    #     # We turn it into "|w|or|d|" with mask [1,   1, 0, 1,   1].
    #     # Index i will now say whether there is a boundary BEFORE character i,
    #     # with an extra position at the end for a boundary behind the last character.
    #     boundary_before_asmask = [1] + boundary_after_asmask
    #     boundary_before_asmask[-1] = 1
    #
    #     # Finally, convert this to indices. In the above example: [0, 1, 3, 4].
    #     boundary_before = np.nonzero(boundary_before_asmask)[0]  # np.nonzero produces one array PER dimension. No errors are thrown if you forget the [0], but the zip() below will be empty!
    #     return boundary_before.tolist()

    def getHardIndices(self, string: str, widths: list[int]) -> list[Sequence[int]]:
        """
        First computes thresholded post-character split probabilities, overwritten by slice boundaries wherever applicable,
        then takes slices, converts the mask to (slice-local) split-point indices, and prepends the starting index 0.

        This code was tested for correctness. It works as long as sum(widths) == len(string).
        """
        M_boundary_before_char = [None] + [1 * (exp(ln) > 0.5) for ln in self.logprob_classifier.getPointLogProbabilities(string)]  # The None is intentional. It means that now a boundary after 2 characters means M[2] = 1.
        # Example: thresholding says A/BC/DEF/GHIJK/ then the mask is
        # M_boundary_before_char = [None] + [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]

        token_ends = tuple(cumsum(widths))
        for end in token_ends:
            M_boundary_before_char[end] = 1

        hard_indices = []
        for start,end in zip((0,) + token_ends, token_ends):
            partial_mask = [1] + M_boundary_before_char[start+1:end+1]
            hard_indices.append(tuple(i for i in range(len(partial_mask)) if partial_mask[i]))
        return hard_indices

    def generateGrids(self, string: str, max_k: int, widths: list[int]) -> list[ViterbiStepScores]:
        return [self._generateGrid(boundaries, width, max_k) for width, boundaries in zip(widths, self.getHardIndices(string, widths))]

    @abstractmethod
    def _generateGrid(self, boundaries: Sequence[int], width: int, max_k: int) -> ViterbiStepScores:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "_Normed"*self.do_normalise + f"(pm={self.punishment}))"


class BoundaryPrefixLength(_ScoreGeneratorUsingCharacterClassifierForHardBoundaries):
    """
    Doesn't output probability/score, but does character counting instead.

    Takes the argmax segmentation as suggested by the classifier (which we know scores 92% in all metrics)
    and attributes a score of S to every step of length S that starts on a suggested segmentation boundary and ends on
    or before the next segmentation boundary.

    For example, for the suggested segmentation
        re|anim|atie|techn|iek
    the path scores for
        re|ani|mat|i|e|tec|hniek
    would be
        2 +3  +0  +0+0+3  +0

    If the punishment is nonzero, you have an incentive against making unnecessarily many steps when you do not start on
    a proposed boundary, to try bridging the gap ASAP. (You already have incentive to use long tokens when you start on
    a boundary.)
    """

    def _generateGrid(self, boundaries: Sequence[int], width: int, max_k: int) -> ViterbiStepScores:
        scores = ViterbiStepScores(width, max_k, default=self.punishment)  # Every step that is NOT a prefix is discouraged, and more such steps are more discouraged.
        for start,end in zip(boundaries[:-1], boundaries[1:]):
            span_length = end-start
            denominator = span_length if self.do_normalise else 1

            K = min(span_length, max_k)
            for k in range(K):
                scores.set(start, k, (k+1)/denominator)  # The reason for using span_length rather than K as denominator is that we want the normalised vs. unnormalised behaviour to be the same a.f.o. changing K. Let's say you used K here, the max score would always be K/K == 1 no matter what K was, whereas the max unnormalised score would change with varying K (if K < span_length). Now, they both have a varying max with varying K.

        return scores


class BoundaryPrefixLengthExtended(_ScoreGeneratorUsingCharacterClassifierForHardBoundaries):
    """
    The same as BoundaryPrefixLength except you are also rewarded if you start on a boundary and end AFTER the next
    boundary, with the reward stagnating at that boundary.

    In BoundaryPrefixLength, the reward drops to 0 after that boundary (or the punishment), which doesn't really allow
    making the trade-off of "I can capture this boundary if I jump over the next one". With boundary probabilities, you
    can jump a gap and still harvest the score for the boundary you end up at!
    """

    def _generateGrid(self, boundaries: Sequence[int], width: int, max_k: int) -> ViterbiStepScores:
        scores = ViterbiStepScores(width, max_k, default=self.punishment)

        for n, n_where_reward_stagnates in zip(boundaries[:-1], boundaries[1:]):
            steps_with_increasing_reward = n_where_reward_stagnates - n  # == span_length above.
            denominator = steps_with_increasing_reward if self.do_normalise else 1

            K = min(max_k, width-n)
            for k in range(K):
                scores.set(n, k, min(steps_with_increasing_reward, k+1)/denominator)  # Increasing as long as k <= steps_with_increasing_reward-1, then stays at that value. E.g.: if you have |wo|rd|, the outer loop will produce (n,end) == (0,2), so taking a step of k+1 == 1 will get you reward k+1 == 1, a step of k+1 == 2 will get k+1 == 2, but a step of k+1 == 3 will still give reward end-n == 2. You hence got 2 iterations of increasing reward.

        return scores


class BoundarySuffixLength(_ScoreGeneratorUsingCharacterClassifierForHardBoundaries):
    """
    Equivalent of BoundaryPrefixLength but for suffices: you get a score proportional to how much of the END of a
    suggested segment you capture.

    If you have splits |abcde|fghi| then you get a score of 3 for a token "cde" because it ENDS on a boundary, not
    starts on it. Prefix objectives give that 0 score, and rather give you credit for any extension of that split point
    (e.g. "fgh").
    """

    def _generateGrid(self, boundaries: Sequence[int], width: int, max_k: int) -> ViterbiStepScores:
        scores = ViterbiStepScores(width, max_k, default=self.punishment)

        for start,end in zip(boundaries[:-1], boundaries[1:]):
            span_length = end - start
            denominator = span_length if self.do_normalise else 1

            K = min(max_k, span_length)
            for k in range(K):
                scores.set(end-(k+1), k, (k+1)/denominator)

        return scores


class BoundarySuffixLengthExtended(_ScoreGeneratorUsingCharacterClassifierForHardBoundaries):
    """
    Analogue of BoundaryPrefixLengthExtended.
    """

    def _generateGrid(self, boundaries: Sequence[int], width: int, max_k: int) -> ViterbiStepScores:
        scores = ViterbiStepScores(width, max_k, default=self.punishment)

        for n_where_reward_stagnates, n in zip(boundaries[:-1], boundaries[1:]):
            steps_with_increasing_reward = n - n_where_reward_stagnates
            denominator = steps_with_increasing_reward if self.do_normalise else 1

            K = min(max_k, n)  # K is the amount of steps you move back. If you are on index n, you move back 1 steps to n-1, 2 steps to n-2, ..., n steps to 0, and you can't go further.
            for k in range(K):
                scores.set(n-(k+1), k, min(steps_with_increasing_reward, k+1)/denominator)

        return scores


class BoundaryPrefixAndSuffixLength(_ScoreGeneratorUsingCharacterClassifierForHardBoundaries):
    """
    Combination of prefix and suffix scores. See BoundaryPrefixAndSuffixLengthExtended for documentation.
    This is the version without extension, i.e. you get no
    """

    def _generateGrid(self, boundaries: Sequence[int], width: int, max_k: int) -> ViterbiStepScores:
        scores = ViterbiStepScores(width, max_k, default=self.punishment)

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            span_length = end-start
            denominator = span_length if self.do_normalise else 1

            # Prefix score
            K_nodes_forward = min(max_k, span_length-1)  # Either limited by max step OR by the next boundary (which can be the edge of the graph); the step to the boundary itself is not yet scored.
            for k in range(K_nodes_forward):
                scores.set(start, k, (k+1)/denominator)

            # Suffix score
            K_nodes_back = min(max_k, span_length-1)  # Idem
            for k in range(K_nodes_back):
                scores.set(end-(k+1), k, (k+1)/denominator)

            # Exact-match score gets a slight boost so that it can be differentiated from matching a token with a prefix and suffix back-to-back.
            scores.set(start, span_length-1, (span_length + 0.01)/denominator)

        return scores


class BoundaryPrefixAndSuffixLengthExtended(_ScoreGeneratorUsingCharacterClassifierForHardBoundaries):
    """
    Combination of prefix and suffix scores.
        - When you don't start on a boundary and don't end on a boundary, you get punished.
        - When you do start on a boundary but don't end on a boundary, you get prefix score.
        - When you don't start on a boundary and do end on a boundary, you get suffix score.
        - When you do start on a boundary and do end on a boundary, there are two cases:
            - There is at least one boundary in between them: you get the sum of prefix and suffix.
            - There are no boundaries in between them: you get prefix or suffix (they are equal).
              The reason for doing this is so that every character is counted exactly once.

    You might say: in a string |ab|cdefg|, doesn't the token "bcdefg" get you the same score as "cdefg", since the
    suffix score for hitting the "g" is still the 5 you'd get for "cdefg"? Yes, HOWEVER, notice that "bcdefg" starts on
    "b" rather than "c", so you didn't get a suffix score for reaching "c".
    Thus, /a/bcdefg/ contributes 1 ("a" contains 1 starting character of "ab") + 5 ("bcdefg" contains 5 ending characters of "cdefg") = 6,
    whilst /ab/cdefg/ contributes 2 ("ab" match) + 5 ("cdefg" match) = 7.

    A slight correction is added: the decimal part of the score stores how many exact token matches have been made. This
    is to break the following two ties: we know the best segmentation gets you 7. However,
        - The segmentation /ab/cde/fg/ gets you 2 ("ab" match) + 3 ("cde" prefix of "cdefg") + 2 ("fg" suffix of "cdefg") = 7.
        - The segmentation /abcdefg/ gets you 2 ("abcdefg" contains 2 starting characters of "ab") + 5 ("abcdefg" contains 5 ending characters of "cdefg") = 7.
    In other words: you can split a good token in half or merge two tokens without seeing the difference. To counter this,
    an exact match gets an extra 0.01 as natural tiebreaker, hence the scores become
        - /ab/cdefg/ = 2.01 + 5.01 = 7.02
        - /ab/cde/fg/ = 2.01 + 3 + 2 = 7.01
        - /abcdefg/ = 2 + 5 = 7
    """

    def _generateGrid(self, boundaries: Sequence[int], width: int, max_k: int) -> ViterbiStepScores:
        scores = ViterbiStepScores(width, max_k, default=self.punishment)

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            span_length = end-start
            denominator = span_length if self.do_normalise else 1

            # Prefix score
            K_nodes_forward = min(max_k, width-start)
            for k in range(K_nodes_forward):
                scores.set(start, k, min(span_length, k+1)/denominator)

            # Suffix score
            K_nodes_back = min(max_k, end)
            for k in range(K_nodes_back):
                suffix_start = end - (k+1)
                if scores.get(suffix_start, k) == self.punishment:  # We want to use .add() to conserve existing scores, so if no score exists yet, ensure we don't add to the punishment.
                    scores.set(suffix_start, k, 0)

                # If k steps from exactly the start to the end, we know it already got span_length as the score. In that case, only add the tiebreaker.
                scores.add(start, k, min(span_length, k+1)/denominator  if suffix_start != start else  0.01/denominator)

        return scores


########################################################################################################################

from ....preparation.boundaries import BoundaryMarkerLocation, BoundaryMarker
from modest.interfaces.datasets import ModestDataset

class GoldSplits(CharacterClassifier):
    """
    Uses gold segmentations as split point suggestions. This is cheating, but it is a good baseline!

    FIXME: Has to be able to handle byte-based input too. What you receive for scoring is a pretoken about to be
           tokenised. That means for ë you will receive byte-based input and won't find it in your gold dictionary.
           preprocessor.undo() probably works.
    """

    def __init__(self, boundary_marker: BoundaryMarker, dataset: ModestDataset):
        from ....preparation.splitters import OnWhitespaceAndAddMarker
        self.pretokeniser = OnWhitespaceAndAddMarker(replacement=boundary_marker)
        self.pretoken_shift = len(self.pretokeniser.marker.substitute)*(self.pretokeniser.marker.location == BoundaryMarkerLocation.START)

        self.gold_segmentations = {obj.word: obj.segment() for obj in dataset.generate()}

    def getPointLogProbabilities(self, pretoken: str) -> MutableSequence[float]:
        labels = np.zeros(len(pretoken), dtype=np.float32)

        word, _ = self.pretokeniser.marker.isolate(pretoken)
        if word in self.gold_segmentations:
            tokens = self.gold_segmentations[word]
            split_positions = np.cumsum([len(t) for t in tokens[:-1]]) - 1
            # FIXME: Probably shouldn't add the shift to EVERY pretoken, e.g. "energie-efficiëntie" becomes ["Ġenergie", "-", "efficiëntie"] and that last one needs no shift.
            split_positions += self.pretoken_shift  # If "word" is shown to the character model as "Ġword", the suggested split indices should shift by 1.

            labels[split_positions] = 1

        return np.log(labels)


from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.models.canine.modeling_canine import CanineForTokenClassification, TokenClassifierOutput
from transformers.models.canine.tokenization_canine import CanineTokenizer
from ....util.environment import get_torch_device


class HuggingFaceForBinaryCharacterClassification(CharacterClassifier):
    """
    Classifies characters using a HuggingFace checkpoint.
    """

    def __init__(self, characterclassifier_checkpoint: str, input_kwargs: dict=None):
        self.characters_to_modelinput: CanineTokenizer = AutoTokenizer.from_pretrained(characterclassifier_checkpoint)
        self.input_kwargs = input_kwargs or dict()
        self.device = get_torch_device()

        # Sadly there is no generic "ForTokenClassification" type in HuggingFace's API nor is there any way to check that
        # the model is actually classifying tokens, so there's no real way to enforce statically that the user actually gives the correct checkpoint.
        self.model_for_tokenclassification: CanineForTokenClassification = AutoModelForTokenClassification.from_pretrained(characterclassifier_checkpoint)
        self.model_for_tokenclassification.to(self.device)  # Speeds up inference about 2x to 4x on VSC. This call is in-place, unlike for tensors. https://stackoverflow.com/a/59560101/9352077

    def getPointLogProbabilities(self, pretoken: str) -> MutableSequence[float]:
        model_input = self.characters_to_modelinput(pretoken, add_special_tokens=False, return_tensors="pt", **self.input_kwargs)
        with torch.no_grad():  # no_grad means all tensors returned don't have their gradient tracked, so you don't need to .detach() them before going to numpy.
            model_input = {k: v.to(self.device) for k,v in model_input.items()}
            prediction: TokenClassifierOutput = self.model_for_tokenclassification(**model_input)

        chars_by_classes = prediction.logits.squeeze()  # Remove batch dimension (because it has size 1).
        normalisation_constants = torch.logsumexp(chars_by_classes, dim=1)  # Note that normalisation happens not OVER characters, but PER character. It happens over two binary classes, N times.
        positive_logits = chars_by_classes[:,1]
        logprobabilities = positive_logits - normalisation_constants   # TODO: Can this not just be done with torch.nn.LogSoftmax? A softmax's denominator would be SumExp, so subtracting LogSumExp is the same as taking Log of softmax division, hence LogSoftmax.
        return logprobabilities.cpu().numpy()[:len(pretoken)]  # Always need to go to CPU before casting down to numpy. The slice is needed because models like CANINE add padding characters up to a given amount.


########################################################################################################################


class ScoreGeneratorUsingSubstringClassifier(ViterbiStepScoreGenerator):
    """
    The Viterbi score grid is 2D. A substring classifier is also 2D. Therefore, there is no extra step required to
    interpret the probabilities of the classifier into a 2D grid, unlike for character models.

    Has belated instantiation of the model for the same reason as the character classifiers.
    """

    def __init__(self):
        self.classifier: SubstringClassifier = None

    def setBackend(self, classifier: SubstringClassifier):
        self.classifier = classifier

    def generateGrids(self, string: str, max_k: int, widths: list[int]) -> list[ViterbiStepScores]:
        full_grid = np.array(self.classifier.getSegmentLogProbabilities(string, min(max_k, max(widths))))

        pretokenised_grids = []
        start = 0
        for width in widths:
            N, K = width, min(max_k, width)
            scores = ViterbiStepScores(N, K, default=-INFTY)
            scores.grid = full_grid[start:start+N, K]
            scores.grid[np.fliplr(np.tri(N, K, -1)) == 1] = -INFTY  # Close to the right edge of the grid, big steps are no longer allowed. The furthest step allowed is in the final N index, a single K index. That's what this triangle is.

            pretokenised_grids.append(scores)
            start += width

        return pretokenised_grids
